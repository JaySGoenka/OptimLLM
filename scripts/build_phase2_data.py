#!/usr/bin/env python3
"""Build OptimLLM Phase 2 datasets from Arena 55K and LMSYS-Chat-1M.

The pipeline is intentionally resumable. Large intermediate files are stored in
data/.phase2-cache, which is ignored by Git. Final artifacts are written to data/.
No cloud model or paid API is used; embeddings and labels come from Ollama.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import faiss
import numpy as np
import requests
from datasets import load_dataset
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / ".phase2-cache"

ARENA_DATASET = "lmarena-ai/arena-human-preference-55k"
LMSYS_DATASET = "lmsys/lmsys-chat-1m"
EMBED_MODEL = "nomic-embed-text"
LABEL_MODEL = "qwen3:8b"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

TASK_TYPES = {
    "coding",
    "math",
    "reasoning",
    "creative",
    "summarization",
    "simple_qa",
    "planning",
    "data_analysis",
    "translation",
}
DIFFICULTIES = {"easy", "medium", "hard"}
PRIVACY_LEVELS = {"low", "medium", "high"}
TASK_TYPE_ORDER = [
    "coding",
    "math",
    "reasoning",
    "creative",
    "summarization",
    "simple_qa",
    "planning",
    "data_analysis",
    "translation",
]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]
PRIVACY_ORDER = ["low", "medium", "high"]

LABEL_PROMPT_A = """You are a precise classification system. Classify the following user prompt.
Return only valid JSON. No explanation, no markdown, no preamble.

{
  "task_type": one of exactly: coding, math, reasoning, creative,
               summarization, simple_qa, planning, data_analysis, translation,
  "difficulty": one of exactly: easy, medium, hard,
  "privacy": one of exactly: low, medium, high
}

Definitions:
task_type:
  coding — writing, debugging, explaining, or reviewing code
  math — arithmetic, algebra, statistics, or mathematical problem solving
  reasoning — logic puzzles, argument analysis, causal inference, multi-step deduction
  creative — stories, poems, marketing copy, brainstorming, imaginative writing
  summarization — condensing or extracting information from provided text
  simple_qa — factual lookup, definitions, general knowledge questions
  planning — step-by-step plans, project breakdowns, scheduling, strategy
  data_analysis — interpreting data, identifying patterns, statistical reasoning
  translation — converting between languages

difficulty:
  easy — a small language model (1B–4B parameters) can answer this correctly
  medium — requires a 7B+ model or specialist model to answer well
  hard — requires a strong large model; a 7B model would likely fail or give
         incomplete answers

privacy:
  low — contains no personal information and is safe to send to any cloud service
  medium — contains professional context, company names, or role-specific details
           that are mildly sensitive
  high — contains personal identifiers, credentials, medical information, legal
         content, or financial details that must not leave the user's device

Prompt to classify:
\"\"\"
{prompt}
\"\"\"
"""

LABEL_PROMPT_B = """Classify this prompt for an AI routing system. Output only JSON, nothing else.
No markdown fences. No commentary.

Required output format:
{"task_type": "...", "difficulty": "...", "privacy": "..."}

task_type options: coding | math | reasoning | creative | summarization |
                   simple_qa | planning | data_analysis | translation

difficulty options:
  easy = answerable by a 1B–4B local model
  medium = needs a 7B specialist or better
  hard = needs a large capable model, small models would fail

privacy options:
  low = safe for any cloud service
  medium = contains business or professional context, handle with care
  high = contains PII, credentials, medical, legal, or financial content

Classify:
\"\"\"
{prompt}
\"\"\"
"""

LABEL_BATCH_RUBRIC_A = """You are a precise classification system. Classify every user prompt.
Return only valid JSON. No explanation, no markdown, no preamble.

task_type definitions:
coding — writing, debugging, explaining, or reviewing code
math — arithmetic, algebra, statistics, or mathematical problem solving
reasoning — logic puzzles, argument analysis, causal inference, multi-step deduction
creative — stories, poems, marketing copy, brainstorming, imaginative writing
summarization — condensing or extracting information from provided text
simple_qa — factual lookup, definitions, general knowledge questions
planning — step-by-step plans, project breakdowns, scheduling, strategy
data_analysis — interpreting data, identifying patterns, statistical reasoning
translation — converting between languages

difficulty definitions:
easy — a small language model (1B–4B parameters) can answer this correctly
medium — requires a 7B+ model or specialist model to answer well
hard — requires a strong large model; a 7B model would likely fail or be incomplete

privacy definitions:
low — no personal information; safe for any cloud service
medium — professional context, company names, or mildly sensitive role-specific details
high — personal identifiers, credentials, medical, legal, or financial information"""

LABEL_BATCH_RUBRIC_B = """Classify every prompt for an AI routing system. Output only JSON,
nothing else. No markdown fences and no commentary.

task_type options:
coding | math | reasoning | creative | summarization | simple_qa | planning |
data_analysis | translation

difficulty:
easy = answerable by a 1B–4B local model
medium = needs a 7B specialist or better
hard = needs a large capable model; small models would fail

privacy:
low = safe for any cloud service
medium = business or professional context; handle with care
high = PII, credentials, medical, legal, or financial content"""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "task_type": {"type": "string", "enum": sorted(TASK_TYPES)},
        "difficulty": {"type": "string", "enum": sorted(DIFFICULTIES)},
        "privacy": {"type": "string", "enum": sorted(PRIVACY_LEVELS)},
    },
    "required": ["task_type", "difficulty", "privacy"],
    "additionalProperties": False,
}

LABEL_BATCH_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "t": {"type": "integer", "minimum": 0, "maximum": 8},
        "d": {"type": "integer", "minimum": 0, "maximum": 2},
        "p": {"type": "integer", "minimum": 0, "maximum": 2},
    },
    "required": ["id", "t", "d", "p"],
    "additionalProperties": False,
}

PRIVACY_TERMS = re.compile(
    r"\b("
    r"diagnos(?:is|ed)|prescription|doctor|patient|medical|symptoms?|hospital|"
    r"lawyer|attorney|lawsuit|contract|legal advice|privileged|"
    r"bank|account number|routing number|salary|mortgage|credit card|tax return|"
    r"api key|password|credential|secret key|token|social security|ssn"
    r")\b",
    re.IGNORECASE,
)

MODEL_METADATA_RE = re.compile(
    r"(?im)^\s*(system|assistant|model(?:_a|_b)?|arena|chatbot)\s*:\s*.*$"
)
WHITESPACE_RE = re.compile(r"\s+")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str) -> str:
    normalized = WHITESPACE_RE.sub(" ", text.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def sanitize_prompt(text: str) -> str:
    text = MODEL_METADATA_RE.sub("", text)
    text = text.replace("\x00", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def message_role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("role") or message.get("from") or "").lower()


def message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    value = message.get("content", message.get("value", message.get("text", "")))
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return ""


def extract_arena_user_turns(prompt: Any) -> str:
    if isinstance(prompt, str):
        return sanitize_prompt(prompt)
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for item in prompt:
        role = message_role(item)
        if isinstance(item, str) or role in {"", "user", "human"}:
            text = sanitize_prompt(message_text(item))
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def extract_first_user_turn(conversation: Any) -> str:
    if not isinstance(conversation, list):
        return ""
    for item in conversation:
        role = message_role(item)
        if role in {"user", "human"} or (not role and isinstance(item, dict)):
            text = sanitize_prompt(message_text(item))
            if text:
                return text
    return ""


def moderation_flagged(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if value.get("flagged") is True:
            return True
        categories = value.get("categories")
        return isinstance(categories, dict) and any(v is True for v in categories.values())
    if isinstance(value, list):
        return any(moderation_flagged(item) for item in value)
    return False


def normalized_language(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def is_english(value: Any) -> bool:
    return normalized_language(value) in {"en", "eng", "english", "en-us", "en-gb"}


def classify_arena_model(model: str) -> tuple[str, str]:
    """Conservative strong/weak/skip mapping for Arena 55K model aliases."""
    name = model.strip().lower()

    strong_patterns = [
        (r"gpt-4", "GPT-4 family"),
        (r"claude-(?:2|3|opus|sonnet)", "Claude 2+ family"),
        (r"claude-1\.3", "Claude 1.3 high-capability model"),
        (r"gemini-(?:pro|1\.5|ultra)", "Gemini Pro/Ultra family"),
        (r"mistral-medium", "Mistral Medium"),
        (r"palm-2", "PaLM 2"),
        (r"llama[-_ ]?3.*70b|llama[-_ ]?2.*70b", "Llama 70B+"),
        (r"mixtral.*8x22b", "Mixtral 8x22B"),
        (r"qwen.*(?:72b|110b)", "Qwen 72B+"),
        (r"command-r-plus", "Command R+"),
        (r"yi-(?:large|34b)", "Yi large-capability family"),
        (r"wizardlm-70b", "WizardLM 70B"),
        (r"pplx-70b", "PPLX 70B"),
        (r"tulu-2-dpo-70b", "Tulu 2 DPO 70B"),
        (r"deepseek-llm-67b", "DeepSeek LLM 67B"),
        (r"falcon-180b", "Falcon 180B"),
    ]
    weak_patterns = [
        (r"gpt-3\.5", "GPT-3.5 family"),
        (r"claude-(?:instant-1|1)$", "Claude 1/Instant"),
        (r"llama[-_ ]?2.*(?:7b|13b)", "Llama 2 7B/13B"),
        (r"llama-13b", "Llama 13B"),
        (r"llama[-_ ]?3.*8b", "Llama 3 8B"),
        (r"mistral.*7b", "Mistral 7B"),
        (r"mixtral-8x7b", "Mixtral 8x7B"),
        (r"vicuna|alpaca|koala", "legacy small open model"),
        (r"qwen.*(?:1\.8b|4b|7b)", "Qwen under 14B"),
        (r"chatglm|rwkv|oasst|fastchat-t5", "legacy lower-capability open model"),
        (r"openchat|starling|zephyr|solar-10\.7b", "7B-11B open model"),
        (r"gemma.*(?:2b|7b)", "Gemma 2B/7B"),
        (r"wizardlm.*(?:7b|13b)|wizard-vicuna", "WizardLM/Vicuna small model"),
        (r"pplx-7b|codellama.*(?:7b|13b)", "small specialist model"),
        (r"mpt-7b|dolly-v2-12b|stablelm-tuned-alpha-7b", "legacy 7B-12B open model"),
    ]
    for pattern, reason in strong_patterns:
        if re.search(pattern, name):
            return "strong", reason
    for pattern, reason in weak_patterns:
        if re.search(pattern, name):
            return "weak", reason
    return "skip", "ambiguous capability or insufficiently comparable checkpoint"


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def model_names(self) -> set[str]:
        response = self.session.get(f"{self.base_url}/api/tags", timeout=30)
        response.raise_for_status()
        return {model["name"] for model in response.json().get("models", [])}

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = list(texts[start : start + batch_size])
            chunks.append(self._embed_batch_with_retry(batch))
        matrix = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=np.float32)
        faiss.normalize_L2(matrix)
        return matrix

    def _embed_batch_with_retry(
        self, texts: Sequence[str], attempts: int = 3
    ) -> np.ndarray:
        last_error: requests.RequestException | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": EMBED_MODEL,
                        "input": list(texts),
                        "truncate": True,
                        "keep_alive": "30m",
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return np.asarray(response.json()["embeddings"], dtype=np.float32)
            except requests.RequestException as error:
                last_error = error
                if attempt < attempts:
                    time.sleep(2**attempt)
        if len(texts) > 1:
            midpoint = len(texts) // 2
            left = self._embed_batch_with_retry(texts[:midpoint], attempts)
            right = self._embed_batch_with_retry(texts[midpoint:], attempts)
            return np.vstack([left, right])
        raise RuntimeError(f"Unable to embed prompt after retries: {last_error}") from last_error

    def classify(self, prompt: str, version: str) -> dict[str, str] | None:
        template = LABEL_PROMPT_A if version == "A" else LABEL_PROMPT_B
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": LABEL_MODEL,
                "messages": [{"role": "user", "content": template.replace("{prompt}", prompt)}],
                "stream": False,
                "think": False,
                "format": LABEL_SCHEMA,
                "keep_alive": "60m",
                "options": {"temperature": 0, "num_predict": 80, "seed": 17 if version == "A" else 29},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if not match:
                return None
            try:
                value = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if (
            set(value) == {"task_type", "difficulty", "privacy"}
            and value["task_type"] in TASK_TYPES
            and value["difficulty"] in DIFFICULTIES
            and value["privacy"] in PRIVACY_LEVELS
        ):
            return value
        return None

    def classify_batch(
        self, prompts: Sequence[str], version: str
    ) -> list[dict[str, str] | None]:
        if not prompts:
            return []
        rubric = LABEL_BATCH_RUBRIC_A if version == "A" else LABEL_BATCH_RUBRIC_B
        prompt_blocks = "\n\n".join(
            f'<prompt id="{index}">\n{prompt}\n</prompt>'
            for index, prompt in enumerate(prompts)
        )
        batch_prompt = (
            f"{rubric}\n\n"
            "Classify every numbered prompt below independently. Use this compact output "
            "encoding: t is task_type (0=coding, 1=math, 2=reasoning, 3=creative, "
            "4=summarization, 5=simple_qa, 6=planning, 7=data_analysis, 8=translation); "
            "d is difficulty (0=easy, 1=medium, 2=hard); p is privacy "
            "(0=low, 1=medium, 2=high). Return one JSON array item per prompt in the same "
            "order, shaped exactly as {\"id\":0,\"t\":0,\"d\":0,\"p\":0}. Include every id "
            "and do not omit or merge prompts.\n\n"
            f"{prompt_blocks}"
        )
        schema = {
            "type": "array",
            "items": LABEL_BATCH_ITEM_SCHEMA,
            "minItems": len(prompts),
            "maxItems": len(prompts),
        }
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": LABEL_MODEL,
                "messages": [{"role": "user", "content": batch_prompt}],
                "stream": False,
                "think": False,
                "format": schema,
                "keep_alive": "60m",
                "options": {
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": max(120, len(prompts) * 18),
                    "seed": 17 if version == "A" else 29,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        try:
            values = json.loads(content)
        except json.JSONDecodeError:
            return [None] * len(prompts)
        if not isinstance(values, list) or len(values) != len(prompts):
            return [None] * len(prompts)
        results: list[dict[str, str] | None] = [None] * len(prompts)
        for value in values:
            if (
                isinstance(value, dict)
                and isinstance(value.get("id"), int)
                and 0 <= value["id"] < len(prompts)
                and isinstance(value.get("t"), int)
                and 0 <= value["t"] < len(TASK_TYPE_ORDER)
                and isinstance(value.get("d"), int)
                and 0 <= value["d"] < len(DIFFICULTY_ORDER)
                and isinstance(value.get("p"), int)
                and 0 <= value["p"] < len(PRIVACY_ORDER)
            ):
                results[value["id"]] = {
                    "task_type": TASK_TYPE_ORDER[value["t"]],
                    "difficulty": DIFFICULTY_ORDER[value["d"]],
                    "privacy": PRIVACY_ORDER[value["p"]],
                }
        return results


def save_embedding_cache(name: str, prompts: Sequence[str], client: OllamaClient) -> np.ndarray:
    embeddings_path = CACHE_DIR / f"{name}-embeddings.npy"
    hashes_path = CACHE_DIR / f"{name}-embedding-hashes.json"
    hashes = [stable_hash(prompt) for prompt in prompts]
    if embeddings_path.exists() and hashes_path.exists():
        if json.loads(hashes_path.read_text()) == hashes:
            return np.load(embeddings_path)
    embeddings = client.embed(prompts)
    np.save(embeddings_path, embeddings)
    write_json(hashes_path, hashes)
    return embeddings


def deduplicate_by_similarity(
    rows: Sequence[dict[str, Any]], embeddings: np.ndarray, threshold: float
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    if not rows:
        return [], embeddings, 0
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    limits, _, neighbors = index.range_search(embeddings, threshold)
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for row_index in range(len(rows)):
        for neighbor in neighbors[limits[row_index] : limits[row_index + 1]]:
            neighbor = int(neighbor)
            if neighbor < row_index:
                union(row_index, neighbor)

    representatives: dict[int, int] = {}
    for index_value in range(len(rows)):
        root = find(index_value)
        representatives.setdefault(root, index_value)
    kept_indices = sorted(representatives.values())
    removed = len(rows) - len(kept_indices)
    return [rows[index_value] for index_value in kept_indices], embeddings[kept_indices], removed


def arena_stage(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = OllamaClient(args.ollama_url)
    dataset = load_dataset(ARENA_DATASET, split="train", streaming=True)
    model_counts: Counter[str] = Counter()
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for row in tqdm(dataset, desc="Arena battles"):
        counters["total_battles"] += 1
        model_a, model_b = str(row["model_a"]), str(row["model_b"])
        model_counts.update([model_a, model_b])
        tier_a, _ = classify_arena_model(model_a)
        tier_b, _ = classify_arena_model(model_b)
        if "skip" in {tier_a, tier_b}:
            counters["dropped_skip_model"] += 1
            continue
        if tier_a == tier_b:
            counters["dropped_same_tier"] += 1
            continue
        is_tie = bool(row.get("winner_tie"))
        winner_a, winner_b = bool(row.get("winner_model_a")), bool(row.get("winner_model_b"))
        if not is_tie and winner_a == winner_b:
            counters["dropped_unclear_winner"] += 1
            continue
        prompt = extract_arena_user_turns(row.get("prompt"))
        if not prompt:
            counters["dropped_empty_prompt"] += 1
            continue
        prompt_hash = stable_hash(prompt)
        if prompt_hash in seen_hashes:
            counters["dropped_exact_duplicate"] += 1
            continue
        seen_hashes.add(prompt_hash)
        strong_is_a = tier_a == "strong"
        # A cross-tier tie means the weaker model was sufficient for the prompt,
        # so treat it as a weak-tier win for routing-policy training.
        strong_wins = False if is_tie else (winner_a if strong_is_a else winner_b)
        if is_tie:
            counters["ties_labeled_weak_win"] += 1
        candidates.append(
            {
                "prompt": prompt,
                "strong_wins": bool(strong_wins),
                "source": "arena_55k",
                "language": "en",
                "_label_origin": "tie_as_weak_win" if is_tie else "observed_winner",
            }
        )

    tier_map = {}
    for model in sorted(model_counts):
        tier, reason = classify_arena_model(model)
        tier_map[model] = {"tier": tier, "reason": reason, "battle_appearances": model_counts[model]}
    write_json(
        DATA_DIR / "arena-model-tier-map.json",
        {
            "schema_version": 1,
            "dataset": ARENA_DATASET,
            "generated_at": time.strftime("%Y-%m-%d"),
            "models": tier_map,
        },
    )
    counters["after_filter_and_exact_dedup"] = len(candidates)

    embeddings = save_embedding_cache("arena", [row["prompt"] for row in candidates], client)
    final_rows, final_embeddings, removed = deduplicate_by_similarity(
        candidates, embeddings, args.arena_similarity
    )
    counters["dropped_near_duplicate"] = removed
    counters["final_records"] = len(final_rows)
    counters["final_ties_labeled_weak_win"] = sum(
        row["_label_origin"] == "tie_as_weak_win" for row in final_rows
    )
    output_rows = [
        {key: value for key, value in row.items() if key != "_label_origin"} for row in final_rows
    ]
    write_jsonl(DATA_DIR / "arena-preference.jsonl", output_rows)
    np.save(CACHE_DIR / "arena-final-embeddings.npy", final_embeddings)
    write_json(CACHE_DIR / "arena-funnel.json", dict(counters))
    print(json.dumps(dict(counters), indent=2))


def hash_reservoir_add(
    reservoir: list[tuple[int, int, dict[str, Any]]],
    row: dict[str, Any],
    capacity: int,
) -> None:
    score = int(stable_hash(row["prompt"])[:16], 16)
    sequence = int(stable_hash(row["prompt"])[16:32], 16)
    if len(reservoir) < capacity:
        heapq.heappush(reservoir, (-score, sequence, row))
        return
    largest_kept_score = -reservoir[0][0]
    if score >= largest_kept_score:
        return
    heapq.heapreplace(reservoir, (-score, sequence, row))


def lmsys_sample_stage(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = OllamaClient(args.ollama_url)
    dataset = load_dataset(LMSYS_DATASET, split="train", streaming=True)
    counters: Counter[str] = Counter()
    exact_hashes: set[str] = set()
    reservoir: list[tuple[int, int, dict[str, Any]]] = []

    for row in tqdm(dataset, desc="LMSYS streaming scan", total=args.lmsys_scan_limit or None):
        counters["rows_scanned"] += 1
        if args.lmsys_scan_limit and counters["rows_scanned"] > args.lmsys_scan_limit:
            break
        prompt = extract_first_user_turn(row.get("conversation"))
        if len(prompt) < 20:
            counters["dropped_under_20_chars"] += 1
            continue
        if len(prompt) > 2000:
            counters["dropped_over_2000_chars"] += 1
            continue
        if not is_english(row.get("language")):
            counters["dropped_non_english"] += 1
            continue
        if moderation_flagged(row.get("openai_moderation")):
            counters["dropped_moderation_flagged"] += 1
            continue
        prompt_hash = stable_hash(prompt)
        if prompt_hash in exact_hashes:
            counters["dropped_exact_duplicate"] += 1
            continue
        exact_hashes.add(prompt_hash)
        counters["eligible_unique"] += 1
        hash_reservoir_add(
            reservoir,
            {"prompt": prompt, "source": "lmsys_chat_1m", "language": "en"},
            args.lmsys_candidate_pool,
        )

    candidates = [row for _, _, row in sorted(reservoir, key=lambda item: -item[0])]
    counters["candidate_pool"] = len(candidates)
    embeddings = save_embedding_cache("lmsys-candidates", [row["prompt"] for row in candidates], client)

    cluster_count = min(args.lmsys_clusters, max(2, len(candidates) // 50))
    kmeans = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=args.seed,
        batch_size=2048,
        n_init="auto",
        max_iter=200,
    )
    labels = kmeans.fit_predict(embeddings)
    by_cluster: dict[int, list[int]] = defaultdict(list)
    for index_value, cluster in enumerate(labels):
        by_cluster[int(cluster)].append(index_value)

    target = min(args.lmsys_label_target, len(candidates))
    cluster_cap = max(1, math.floor(target * 0.05))
    selected: list[int] = []
    cluster_allocations: dict[int, int] = {}
    remaining = target
    cluster_sizes = {cluster: len(indices) for cluster, indices in by_cluster.items()}
    total_size = sum(cluster_sizes.values())

    for cluster, indices in sorted(by_cluster.items()):
        allocation = min(cluster_cap, len(indices), math.floor(target * len(indices) / total_size))
        cluster_allocations[cluster] = allocation
        remaining -= allocation
    while remaining > 0:
        progressed = False
        for cluster in sorted(by_cluster, key=lambda value: (-cluster_sizes[value], value)):
            if cluster_allocations[cluster] < min(cluster_cap, cluster_sizes[cluster]):
                cluster_allocations[cluster] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    for cluster, indices in sorted(by_cluster.items()):
        length_bins: dict[int, list[int]] = defaultdict(list)
        lengths = np.asarray([len(candidates[index_value]["prompt"]) for index_value in indices])
        boundaries = np.quantile(lengths, [0.25, 0.5, 0.75]) if len(lengths) >= 4 else []
        for index_value in indices:
            length_bin = int(np.searchsorted(boundaries, len(candidates[index_value]["prompt"])))
            length_bins[length_bin].append(index_value)
        for values in length_bins.values():
            values.sort(key=lambda index_value: stable_hash(candidates[index_value]["prompt"]))
        ordered: list[int] = []
        while any(length_bins.values()):
            for length_bin in sorted(length_bins):
                if length_bins[length_bin]:
                    ordered.append(length_bins[length_bin].pop(0))
        selected.extend(ordered[: cluster_allocations[cluster]])

    selected = sorted(selected, key=lambda index_value: stable_hash(candidates[index_value]["prompt"]))
    sampled = [candidates[index_value] | {"cluster_id": int(labels[index_value])} for index_value in selected]
    sampled_embeddings = embeddings[selected]
    counters["sampled_for_labeling"] = len(sampled)
    counters["clusters"] = cluster_count
    counters["largest_cluster_contribution"] = max(cluster_allocations.values(), default=0)
    write_jsonl(CACHE_DIR / "lmsys-sample.jsonl", sampled)
    np.save(CACHE_DIR / "lmsys-sample-embeddings.npy", sampled_embeddings)
    write_json(CACHE_DIR / "lmsys-filter-funnel.json", dict(counters))
    write_json(
        CACHE_DIR / "lmsys-cluster-summary.json",
        {
            "cluster_sizes": cluster_sizes,
            "selected_per_cluster": cluster_allocations,
            "target": target,
            "cap_per_cluster": cluster_cap,
        },
    )
    print(json.dumps(dict(counters), indent=2))


def resolve_double_pass_labels(
    label_a: dict[str, str] | None,
    label_b: dict[str, str] | None,
) -> dict[str, str] | None:
    """Resolve valid labels when both passes agree on the primary task type."""
    if not label_a or not label_b or label_a["task_type"] != label_b["task_type"]:
        return None
    difficulty = min(
        (label_a["difficulty"], label_b["difficulty"]),
        key=DIFFICULTY_ORDER.index,
    )
    privacy = max(
        (label_a["privacy"], label_b["privacy"]),
        key=PRIVACY_ORDER.index,
    )
    return {
        "task_type": label_a["task_type"],
        "difficulty": difficulty,
        "privacy": privacy,
    }


def label_stage(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = OllamaClient(args.ollama_url)
    sample_path = CACHE_DIR / "lmsys-sample.jsonl"
    if not sample_path.exists():
        raise SystemExit("Run lmsys-sample before label.")
    sample = list(read_jsonl(sample_path))
    checkpoint = CACHE_DIR / "lmsys-label-checkpoint.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(checkpoint):
        completed[row["prompt_hash"]] = row

    pending = []
    for row in sample:
        prior = completed.get(stable_hash(row["prompt"]))
        if not prior or prior.get("label_a") is None or prior.get("label_b") is None:
            pending.append(row)
    if args.label_limit:
        pending = pending[: args.label_limit]
    print(f"Label checkpoint: {len(completed)} complete; {len(pending)} pending this run.")

    batches: list[list[dict[str, Any]]] = []
    current_batch: list[dict[str, Any]] = []
    current_chars = 0
    for row in pending:
        prompt_chars = len(row["prompt"])
        if current_batch and (
            len(current_batch) >= args.label_batch_size
            or current_chars + prompt_chars > args.label_batch_chars
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0
        current_batch.append(row)
        current_chars += prompt_chars
    if current_batch:
        batches.append(current_batch)

    def classify_batch_pair(
        batch: Sequence[dict[str, Any]],
    ) -> tuple[Sequence[dict[str, Any]], list[dict[str, str] | None], list[dict[str, str] | None], str]:
        try:
            prompts = [row["prompt"] for row in batch]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(client.classify_batch, prompts, "A")
                future_b = executor.submit(client.classify_batch, prompts, "B")
                labels_a = future_a.result()
                labels_b = future_b.result()
        except requests.RequestException as error:
            labels_a = [None] * len(batch)
            labels_b = [None] * len(batch)
            batch_error = str(error)
        else:
            batch_error = ""
        return batch, labels_a, labels_b, batch_error

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.label_parallel_batches
    ) as executor:
        results = executor.map(classify_batch_pair, batches)
        for batch, labels_a, labels_b, batch_error in tqdm(
            results,
            total=len(batches),
            desc="Qwen3 double-pass label batches",
        ):
            for row, label_a, label_b in zip(batch, labels_a, labels_b):
                prompt_hash = stable_hash(row["prompt"])
                result: dict[str, Any] = {
                    "prompt_hash": prompt_hash,
                    "prompt": row["prompt"],
                    "cluster_id": row.get("cluster_id"),
                    "label_a": label_a,
                    "label_b": label_b,
                    "exact_agreement": bool(label_a and label_b and label_a == label_b),
                    "task_type_agreement": bool(
                        label_a
                        and label_b
                        and label_a["task_type"] == label_b["task_type"]
                    ),
                    "resolved_label": resolve_double_pass_labels(label_a, label_b),
                }
                result["agreement"] = result["resolved_label"] is not None
                if batch_error:
                    result["error"] = batch_error
                append_jsonl(checkpoint, result)
                completed[prompt_hash] = result

    valid = []
    exact_agreements = 0
    task_type_agreements = 0
    for row in sample:
        result = completed.get(stable_hash(row["prompt"]))
        if not result:
            continue
        label_a = result.get("label_a")
        label_b = result.get("label_b")
        resolved = resolve_double_pass_labels(label_a, label_b)
        if resolved:
            exact_agreement = label_a == label_b
            exact_agreements += int(exact_agreement)
            task_type_agreements += 1
            valid.append(
                {
                    "prompt": row["prompt"],
                    **resolved,
                    "source": "lmsys_chat_1m",
                    "label_agreement": exact_agreement,
                    "task_type_agreement": True,
                    "label_resolution": "lower_difficulty_higher_privacy",
                }
            )
    write_jsonl(CACHE_DIR / "lmsys-agreed-pre-dedup.jsonl", valid)
    write_json(
        CACHE_DIR / "lmsys-label-funnel.json",
        {
            "sampled_for_labeling": len(sample),
            "label_attempts_completed": len(completed),
            "exact_label_agreements": exact_agreements,
            "task_type_agreements": task_type_agreements,
            "accepted_labels": len(valid),
            "rejected_task_disagreement_or_invalid": len(completed) - len(valid),
        },
    )
    print(f"Task-agreed labels available: {len(valid)}")


def remove_cross_source_duplicates(
    lmsys_rows: Sequence[dict[str, Any]],
    lmsys_embeddings: np.ndarray,
    arena_embeddings: np.ndarray,
    threshold: float,
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    if not lmsys_rows or len(arena_embeddings) == 0:
        return list(lmsys_rows), lmsys_embeddings, 0
    index = faiss.IndexFlatIP(arena_embeddings.shape[1])
    index.add(arena_embeddings)
    similarities, _ = index.search(lmsys_embeddings, 1)
    keep_indices = [index_value for index_value, score in enumerate(similarities[:, 0]) if score <= threshold]
    return (
        [lmsys_rows[index_value] for index_value in keep_indices],
        lmsys_embeddings[keep_indices],
        len(lmsys_rows) - len(keep_indices),
    )


def privacy_candidates(args: argparse.Namespace, existing_hashes: set[str]) -> Iterator[dict[str, Any]]:
    dataset = load_dataset(LMSYS_DATASET, split="train", streaming=True)
    for row in dataset:
        prompt = extract_first_user_turn(row.get("conversation"))
        if (
            20 <= len(prompt) <= 2000
            and is_english(row.get("language"))
            and not moderation_flagged(row.get("openai_moderation"))
            and PRIVACY_TERMS.search(prompt)
            and stable_hash(prompt) not in existing_hashes
        ):
            existing_hashes.add(stable_hash(prompt))
            yield {"prompt": prompt}


def finalize_stage(args: argparse.Namespace) -> None:
    ensure_dirs()
    client = OllamaClient(args.ollama_url)
    agreed = list(read_jsonl(CACHE_DIR / "lmsys-agreed-pre-dedup.jsonl"))
    if not agreed:
        raise SystemExit("No agreed LMSYS labels found. Run label first.")
    agreed_embeddings = save_embedding_cache("lmsys-agreed", [row["prompt"] for row in agreed], client)
    arena_embeddings = np.load(CACHE_DIR / "arena-final-embeddings.npy")
    final_rows, final_embeddings, cross_removed = remove_cross_source_duplicates(
        agreed, agreed_embeddings, arena_embeddings, args.cross_similarity
    )

    # Privacy augmentation is resumable and only runs if the agreed set is below 8% high privacy.
    high_count = sum(row["privacy"] == "high" for row in final_rows)
    target_high = math.ceil((len(final_rows) - high_count) / 9)  # high / total ~= 10%
    augmentation_checkpoint = CACHE_DIR / "privacy-augmentation-checkpoint.jsonl"
    augmented = list(read_jsonl(augmentation_checkpoint))
    existing_hashes = {stable_hash(row["prompt"]) for row in final_rows}
    existing_hashes.update(stable_hash(row["prompt"]) for row in augmented)
    if final_rows and high_count / len(final_rows) < 0.08:
        needed = max(0, target_high - high_count - sum(row.get("privacy") == "high" for row in augmented))
        if needed:
            print(f"High privacy below 8%; seeking {needed} additional agreed high-privacy prompts.")
            for candidate in tqdm(privacy_candidates(args, existing_hashes), desc="Privacy augmentation"):
                label_a = client.classify(candidate["prompt"], "A")
                label_b = client.classify(candidate["prompt"], "B")
                if label_a and label_a == label_b and label_a["privacy"] == "high":
                    record = {
                        "prompt": candidate["prompt"],
                        **label_a,
                        "source": "lmsys_chat_1m",
                        "label_agreement": True,
                        "privacy_augmented": True,
                    }
                    append_jsonl(augmentation_checkpoint, record)
                    augmented.append(record)
                    needed -= 1
                    if needed <= 0:
                        break

    if augmented:
        augmentation_embeddings = save_embedding_cache(
            "privacy-augmentation", [row["prompt"] for row in augmented], client
        )
        augmented, augmentation_embeddings, removed = remove_cross_source_duplicates(
            augmented, augmentation_embeddings, arena_embeddings, args.cross_similarity
        )
        cross_removed += removed
        final_rows.extend(augmented)
        final_embeddings = np.vstack([final_embeddings, augmentation_embeddings])

    write_jsonl(DATA_DIR / "lmsys-labeled.jsonl", final_rows)
    np.save(CACHE_DIR / "lmsys-final-embeddings.npy", final_embeddings)

    arena_rows = list(read_jsonl(DATA_DIR / "arena-preference.jsonl"))
    task_counts = Counter(row["task_type"] for row in final_rows)
    difficulty_counts = Counter(row["difficulty"] for row in final_rows)
    privacy_counts = Counter(row["privacy"] for row in final_rows)
    strong_wins = sum(bool(row["strong_wins"]) for row in arena_rows)
    strong_rate = strong_wins / len(arena_rows) if arena_rows else 0
    high_rate = privacy_counts["high"] / len(final_rows) if final_rows else 0
    label_funnel = json.loads((CACHE_DIR / "lmsys-label-funnel.json").read_text())
    filter_funnel = json.loads((CACHE_DIR / "lmsys-filter-funnel.json").read_text())
    arena_funnel = json.loads((CACHE_DIR / "arena-funnel.json").read_text())

    coverage_gaps = {task: task_counts[task] for task in sorted(TASK_TYPES) if task_counts[task] < 500}
    lines = [
        "# Phase 2 Dataset Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d')}",
        "",
        "## Totals",
        "",
        f"- Arena preference records: {len(arena_rows):,}",
        f"- LMSYS labeled records: {len(final_rows):,}",
        f"- Combined records: {len(arena_rows) + len(final_rows):,}",
        "",
        "## Arena funnel",
        "",
    ]
    lines.extend(f"- {key}: {value:,}" for key, value in arena_funnel.items())
    lines.extend(
        [
            "",
            f"Strong-model win rate: {strong_rate:.2%}"
            + (" — FLAG: outside 35%-65%." if not 0.35 <= strong_rate <= 0.65 else ""),
            "",
            "## LMSYS funnel",
            "",
        ]
    )
    lines.extend(f"- {key}: {value:,}" for key, value in filter_funnel.items())
    lines.extend(f"- {key}: {value:,}" for key, value in label_funnel.items())
    lines.append(f"- removed by Arena similarity > {args.cross_similarity}: {cross_removed:,}")

    def add_distribution(title: str, counts: Counter[str]) -> None:
        lines.extend(["", f"## {title}", ""])
        total = sum(counts.values())
        for key in sorted(counts):
            lines.append(f"- {key}: {counts[key]:,} ({counts[key] / total:.2%})")

    add_distribution("Task type distribution", task_counts)
    add_distribution("Difficulty distribution", difficulty_counts)
    add_distribution("Privacy distribution", privacy_counts)
    if high_rate < 0.08:
        lines.extend(["", f"**FLAG:** High-privacy coverage is {high_rate:.2%}, below 8%."])
    if coverage_gaps:
        lines.extend(["", "## Coverage gaps", ""])
        lines.extend(f"- {task}: {count:,} records" for task, count in coverage_gaps.items())
    else:
        lines.extend(["", "## Coverage gaps", "", "No task type has fewer than 500 records."])
    (DATA_DIR / "dataset-summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(final_rows)} LMSYS records and dataset summary.")


def check_prerequisites(args: argparse.Namespace) -> None:
    client = OllamaClient(args.ollama_url)
    try:
        models = client.model_names()
    except requests.RequestException as error:
        raise SystemExit(f"Ollama is unavailable at {args.ollama_url}: {error}") from error
    available_aliases = models | {model.removesuffix(":latest") for model in models}
    missing = [model for model in (EMBED_MODEL, LABEL_MODEL) if model not in available_aliases]
    if missing:
        raise SystemExit(f"Missing Ollama models: {', '.join(missing)}")
    print(f"Ollama ready with {EMBED_MODEL} and {LABEL_MODEL}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["check", "arena", "lmsys-sample", "label", "finalize", "all"],
    )
    parser.add_argument("--ollama-url", default=OLLAMA_BASE_URL)
    parser.add_argument("--arena-similarity", type=float, default=0.95)
    parser.add_argument("--cross-similarity", type=float, default=0.92)
    parser.add_argument("--lmsys-candidate-pool", type=int, default=60000)
    parser.add_argument("--lmsys-label-target", type=int, default=30000)
    parser.add_argument("--lmsys-clusters", type=int, default=200)
    parser.add_argument("--lmsys-scan-limit", type=int, default=0)
    parser.add_argument("--label-limit", type=int, default=0)
    parser.add_argument("--label-batch-size", type=int, default=24)
    parser.add_argument("--label-batch-chars", type=int, default=20000)
    parser.add_argument("--label-parallel-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    random.seed(args.seed)
    if args.stage == "check":
        check_prerequisites(args)
        return
    if args.stage in {"arena", "all"}:
        check_prerequisites(args)
        arena_stage(args)
    if args.stage in {"lmsys-sample", "all"}:
        check_prerequisites(args)
        lmsys_sample_stage(args)
    if args.stage in {"label", "all"}:
        check_prerequisites(args)
        label_stage(args)
    if args.stage in {"finalize", "all"}:
        check_prerequisites(args)
        finalize_stage(args)


if __name__ == "__main__":
    main()
