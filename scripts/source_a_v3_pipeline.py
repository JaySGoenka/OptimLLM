#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_score import BERTScorer
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
CACHE = DATA / ".cache-v3"
LOG_PATH = DOCS / "srcA_v4.md"

ARENA_DATASET = "lmarena-ai/arena-human-preference-55k"
OLLAMA_URL = "http://127.0.0.1:11435"
EMBED_MODEL = "nomic-embed-text"

DetectorFactory.seed = 42

STRONG = {
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "claude-2.0",
    "claude-2.1",
    "gemini-pro",
    "gemini-pro-dev-api",
    "mistral-medium",
    "palm-2",
}

MEDIUM = {
    "mixtral-8x7b-instruct-v0.1",
    "llama-2-70b-chat",
    "llama2-70b-steerlm-chat",
    "wizardlm-70b",
    "tulu-2-dpo-70b",
    "yi-34b-chat",
    "deepseek-llm-67b-chat",
    "qwen1.5-72b-chat",
}

WEAK = {
    "gpt-3.5-turbo-0314",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "claude-1",
    "claude-instant-1",
    "llama-2-7b-chat",
    "llama-2-13b-chat",
    "llama-13b",
    "mistral-7b-instruct",
    "mistral-7b-instruct-v0.2",
    "vicuna-7b",
    "vicuna-13b",
    "vicuna-33b",
    "alpaca-13b",
    "koala-13b",
    "dolly-v2-12b",
    "oasst-pythia-12b",
    "fastchat-t5-3b",
    "stablelm-tuned-alpha-7b",
    "mpt-7b-chat",
    "chatglm-6b",
    "chatglm2-6b",
    "chatglm3-6b",
    "RWKV-4-Raven-14B",
    "qwen-14b-chat",
    "qwen1.5-4b-chat",
    "qwen1.5-7b-chat",
    "openchat-3.5",
    "openchat-3.5-0106",
    "openhermes-2.5-mistral-7b",
    "dolphin-2.2.1-mistral-7b",
    "nous-hermes-2-mixtral-8x7b-dpo",
    "pplx-7b-online",
    "solar-10.7b-instruct-v1.0",
    "starling-lm-7b-alpha",
    "zephyr-7b-alpha",
    "zephyr-7b-beta",
    "wizardlm-13b",
    "gpt4all-13b-snoozy",
    "guanaco-33b",
}

SKIP = {
    "falcon-180b-chat",
    "pplx-70b-online",
    "codellama-34b-instruct",
    "stripedhyena-nous-7b",
    "mpt-30b-chat",
}

TIER: dict[str, str] = {m: "strong" for m in STRONG}
TIER.update({m: "medium" for m in MEDIUM})
TIER.update({m: "weak" for m in WEAK})
TIER.update({m: "skip" for m in SKIP})

REFUSAL_PATTERNS = ("I cannot", "I'm sorry", "I'm not able", "As an AI")
CODE_RE = re.compile(
    r"\b(def|function|class|SELECT|import|from\s+\w+\s+import|for\s+\w+\s+in|while\s+|try:|except|lambda|return|console\.log|npm|pip|sql|python|javascript)\b",
    re.IGNORECASE,
)


def now() -> str:
    return datetime.now().astimezone().strftime("%B %d, %Y %H:%M:%S %Z")


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n- {now()} — {message}\n")
    print(message, flush=True)


def append_section(title: str, body: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n\n{body.rstrip()}\n")


def fail(step: str, error: BaseException) -> None:
    log(f"FAILED at {step}: {type(error).__name__}: {error}")
    raise error


def parse_json_list(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(x) for x in value)
    if not isinstance(value, str):
        return str(value)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return "\n\n".join(str(x) for x in parsed)
    except Exception:
        pass
    return value


def clear_single_winner(row: dict[str, Any]) -> bool:
    return int(row["winner_model_a"]) + int(row["winner_model_b"]) == 1 and int(row["winner_tie"]) == 0


def winner_model(row: dict[str, Any]) -> str:
    return row["model_a"] if int(row["winner_model_a"]) == 1 else row["model_b"]


def is_english(text: str) -> bool:
    try:
        return detect(text[:1000]) == "en"
    except LangDetectException:
        return False


def starts_refusal(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(p) for p in REFUSAL_PATTERNS)


def valid_python_syntax(text: str) -> bool:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = blocks or [text]
    for candidate in candidates:
        snippet = "\n".join(
            line for line in candidate.splitlines()
            if line.strip() and not line.strip().startswith(("#", ">>>", "$"))
        )
        if len(snippet) < 20:
            continue
        try:
            ast.parse(snippet)
            return True
        except SyntaxError:
            continue
    return False


def tier_matchup(model_a: str, model_b: str) -> str | None:
    ta, tb = TIER.get(model_a), TIER.get(model_b)
    if "skip" in {ta, tb} or ta is None or tb is None:
        return None
    tiers = {ta, tb}
    if tiers == {"strong", "medium"}:
        return "Tier1vTier2"
    if tiers == {"strong", "weak"}:
        return "Tier1vTier3"
    if tiers == {"medium", "weak"}:
        return "Tier2vTier3"
    return None


def stronger_model_first(model_a: str, response_a: str, model_b: str, response_b: str) -> tuple[str, str, str, str]:
    rank = {"strong": 3, "medium": 2, "weak": 1}
    if rank[TIER[model_a]] >= rank[TIER[model_b]]:
        return model_a, response_a, model_b, response_b
    return model_b, response_b, model_a, response_a


def build_tier_map(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_models = sorted(set(r["model_a"] for r in raw_rows) | set(r["model_b"] for r in raw_rows))
    missing = [m for m in raw_models if m not in TIER]
    if missing:
        raise ValueError(f"Unassigned Arena models: {missing}")

    tier_map = {
        "schema": "source_a_v4_three_tier_map",
        "created": now(),
        "tiers": {
            "strong": sorted(STRONG),
            "medium": sorted(MEDIUM),
            "weak": sorted(WEAK),
            "skip": sorted(SKIP),
        },
        "models": {
            model: {
                "tier": TIER[model],
                "reason": "Source A v4 explicit three-tier assignment",
            }
            for model in raw_models
        },
    }
    (DATA / "arena-model-tier-map.json").write_text(json.dumps(tier_map, indent=2), encoding="utf-8")

    counts = {"Tier1vTier2": 0, "Tier1vTier3": 0, "Tier2vTier3": 0, "discarded": 0}
    for row in raw_rows:
        match = tier_matchup(row["model_a"], row["model_b"])
        if match is None:
            counts["discarded"] += 1
        else:
            counts[match] += 1

    log(
        "Step A1 complete. Tier assignments: "
        f"strong={len(STRONG)}, medium={len(MEDIUM)}, weak={len(WEAK)}, skip={len(SKIP)}. "
        f"Raw battle tier counts: Tier1vTier2={counts['Tier1vTier2']}, "
        f"Tier1vTier3={counts['Tier1vTier3']}, Tier2vTier3={counts['Tier2vTier3']}, "
        f"discarded={counts['discarded']}."
    )
    return counts


def load_arena() -> list[dict[str, Any]]:
    ds = load_dataset(ARENA_DATASET, split="train")
    return [dict(r) for r in ds]


def filter_and_prepare(raw_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: dict[str, int] = {"raw": len(raw_rows)}
    rows = raw_rows

    before = len(rows)
    rows = [r for r in rows if clear_single_winner(r)]
    counts["filter_1_after_clear_winner"] = len(rows)
    counts["filter_1_dropped_ties_or_both_bad"] = before - len(rows)

    def responses_ok(row: dict[str, Any]) -> bool:
        return len(parse_json_list(row["response_a"])) >= 50 and len(parse_json_list(row["response_b"])) >= 50

    before = len(rows)
    rows = [r for r in rows if responses_ok(r)]
    counts["filter_2_after_response_length"] = len(rows)
    counts["filter_2_dropped_short_response"] = before - len(rows)

    def no_refusal(row: dict[str, Any]) -> bool:
        return not starts_refusal(parse_json_list(row["response_a"])) and not starts_refusal(parse_json_list(row["response_b"]))

    before = len(rows)
    rows = [r for r in rows if no_refusal(r)]
    counts["filter_3_after_refusal"] = len(rows)
    counts["filter_3_dropped_refusal"] = before - len(rows)

    def ratio_ok(row: dict[str, Any]) -> bool:
        a = max(1, len(parse_json_list(row["response_a"])))
        b = max(1, len(parse_json_list(row["response_b"])))
        return max(a, b) / min(a, b) <= 4.0

    before = len(rows)
    rows = [r for r in rows if ratio_ok(r)]
    counts["filter_4_after_length_ratio"] = len(rows)
    counts["filter_4_dropped_length_ratio"] = before - len(rows)

    before = len(rows)
    rows = [r for r in rows if is_english(parse_json_list(r["prompt"]))]
    counts["filter_5_after_english"] = len(rows)
    counts["filter_5_dropped_non_english"] = before - len(rows)

    before = len(rows)
    rows = [r for r in rows if tier_matchup(r["model_a"], r["model_b"]) is not None]
    counts["after_tier_matchup_filter"] = len(rows)
    counts["dropped_by_tier_matchup_filter_after_noise"] = before - len(rows)

    prepared: list[dict[str, Any]] = []
    matchup_counts = {"Tier1vTier2": 0, "Tier1vTier3": 0, "Tier2vTier3": 0}
    for row in rows:
        matchup = tier_matchup(row["model_a"], row["model_b"])
        assert matchup is not None
        model_strong, response_strong, model_weak, response_weak = stronger_model_first(
            row["model_a"],
            parse_json_list(row["response_a"]),
            row["model_b"],
            parse_json_list(row["response_b"]),
        )
        matchup_counts[matchup] += 1
        prepared.append(
            {
                "prompt": parse_json_list(row["prompt"]),
                "response_strong": response_strong,
                "response_weak": response_weak,
                "model_strong": model_strong,
                "model_weak": model_weak,
                "winner": winner_model(row),
                "tier_matchup": matchup,
            }
        )
    counts["Tier1vTier2_after_filters"] = matchup_counts["Tier1vTier2"]
    counts["Tier1vTier3_after_filters"] = matchup_counts["Tier1vTier3"]
    counts["Tier2vTier3_after_filters"] = matchup_counts["Tier2vTier3"]

    log(
        "Step A2 complete. Row counts: "
        f"raw={counts['raw']}, after clear winners={counts['filter_1_after_clear_winner']} "
        f"(dropped {counts['filter_1_dropped_ties_or_both_bad']}), "
        f"after response length={counts['filter_2_after_response_length']} "
        f"(dropped {counts['filter_2_dropped_short_response']}), "
        f"after refusal filter={counts['filter_3_after_refusal']} "
        f"(dropped {counts['filter_3_dropped_refusal']}), "
        f"after length ratio={counts['filter_4_after_length_ratio']} "
        f"(dropped {counts['filter_4_dropped_length_ratio']}), "
        f"after English={counts['filter_5_after_english']} "
        f"(dropped {counts['filter_5_dropped_non_english']}), "
        f"after tier matchup filter={counts['after_tier_matchup_filter']} "
        f"(dropped {counts['dropped_by_tier_matchup_filter_after_noise']})."
    )
    return prepared, counts


def truncate_for_bertscore(text: str, max_chars: int = 3000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def truncate_for_embedding(text: str, max_chars: int = 4000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def compute_scores(rows: list[dict[str, Any]], batch_size: int = 4) -> list[dict[str, Any]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    checkpoint = CACHE / "bertscore-f1-checkpoint-v4.npy"
    strong_responses = [truncate_for_bertscore(r["response_strong"]) for r in rows]
    weak_responses = [truncate_for_bertscore(r["response_weak"]) for r in rows]
    f1s: list[float | None] = [None] * len(rows)

    def score_key(row: dict[str, Any]) -> tuple[str, str, str]:
        return (row["prompt"], row["model_strong"], row["model_weak"])

    reused = 0
    prior_v3 = DATA / "arena-preference-v3.jsonl"
    if prior_v3.exists():
        prior_scores: dict[tuple[str, str, str], float] = {}
        with prior_v3.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    prior = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prior_scores[(prior["prompt"], prior["model_strong"], prior["model_weak"])] = float(prior["similarity"])
        for i, row in enumerate(rows):
            cached = prior_scores.get(score_key(row))
            if cached is not None:
                f1s[i] = cached
                reused += 1
        log(f"Step A3 reused {reused} BERTScore similarities from v3 JSONL by key.")

    if checkpoint.exists():
        cached_v4 = np.load(checkpoint, allow_pickle=False)
        if cached_v4.shape[0] == len(rows):
            for i, value in enumerate(cached_v4):
                if not np.isnan(value):
                    f1s[i] = float(value)
            log(f"Step A3 resumed v4 BERTScore checkpoint with {sum(v is not None for v in f1s)} completed rows.")

    missing = [i for i, value in enumerate(f1s) if value is None]
    if not missing:
        log(f"Step A3 using complete keyed BERTScore cache with {len(rows)} rows; scorer construction skipped.")
    else:
        log(f"Step A3 computing BERTScore for {len(missing)} new or uncached rows.")
        scorer = BERTScorer(
            model_type="distilbert-base-uncased",
            lang="en",
            rescale_with_baseline=False,
            batch_size=batch_size,
        )

        for start in range(0, len(missing), batch_size):
            batch_indices = missing[start : start + batch_size]
            _, _, f = scorer.score(
                [strong_responses[i] for i in batch_indices],
                [weak_responses[i] for i in batch_indices],
                verbose=False,
                batch_size=batch_size,
            )
            for idx, value in zip(batch_indices, f.cpu().numpy()):
                f1s[idx] = float(value)
            checkpoint_values = np.asarray([np.nan if v is None else v for v in f1s], dtype=np.float32)
            np.save(checkpoint, checkpoint_values)
            completed = sum(v is not None for v in f1s)
            if completed % 500 < batch_size or completed == len(rows):
                log(f"Step A3 BERTScore progress: completed {completed} of {len(rows)} rows.")

    raw_scores = []
    boost_count = 0
    for row, similarity in zip(rows, f1s):
        quality_gap = 1.0 - similarity
        lower_tier_won = row["winner"] == row["model_weak"]
        outcome_weight = 0.7 if lower_tier_won else 1.0
        raw = quality_gap * outcome_weight
        boost = False
        if CODE_RE.search(row["prompt"]):
            if valid_python_syntax(row["response_strong"]) and not valid_python_syntax(row["response_weak"]):
                raw *= 1.3
                boost = True
                boost_count += 1
        tier_scale = {"Tier1vTier2": 0.9, "Tier1vTier3": 1.0, "Tier2vTier3": 0.7}
        raw *= tier_scale[row["tier_matchup"]]
        row["similarity"] = similarity
        row["outcome_weight"] = outcome_weight
        row["structural_boost_applied"] = boost
        row["raw_score"] = raw
        raw_scores.append(raw)

    raw_arr = np.asarray(raw_scores, dtype=np.float64)
    min_raw, max_raw = float(raw_arr.min()), float(raw_arr.max())
    if math.isclose(max_raw, min_raw):
        normalized = np.full_like(raw_arr, 0.5)
    else:
        normalized = 0.05 + ((raw_arr - min_raw) / (max_raw - min_raw)) * 0.90

    output_rows = []
    for row, score in zip(rows, normalized):
        output_rows.append(
            {
                "prompt": row["prompt"],
                "routing_score": float(score),
                "model_strong": row["model_strong"],
                "model_weak": row["model_weak"],
                "similarity": float(row["similarity"]),
                "outcome_weight": float(row["outcome_weight"]),
                "tier_matchup": row["tier_matchup"],
                "structural_boost_applied": bool(row["structural_boost_applied"]),
            }
        )

    similarities = np.asarray([float(v) for v in f1s], dtype=np.float64)
    scores = np.asarray([r["routing_score"] for r in output_rows], dtype=np.float64)
    with (DATA / "arena-preference-v4.jsonl").open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log(
        "Step A3 complete. "
        f"BERTScore similarity mean={similarities.mean():.6f}, std={similarities.std():.6f}, "
        f"min={similarities.min():.6f}, max={similarities.max():.6f}. "
        f"routing_score mean={scores.mean():.6f}, std={scores.std():.6f}, "
        f"min={scores.min():.6f}, max={scores.max():.6f}. "
        f"structural_boost_applied={boost_count} rows ({(boost_count / len(rows) * 100):.2f}%). "
        f"Final row count={len(output_rows)}."
    )
    return output_rows


def verify_ollama() -> None:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    log(f"Step A4 Ollama reachable at {OLLAMA_URL}; available tags response received.")


def embed_batch(prompts: list[str]) -> np.ndarray:
    embeddings = []
    for prompt in prompts:
        last_error: Exception | None = None
        variants = [prompt]
        for fallback_chars in (3000, 2000, 1000):
            fallback = truncate_for_embedding(prompt, fallback_chars)
            if fallback != variants[-1]:
                variants.append(fallback)
        for variant_idx, variant in enumerate(variants):
            for attempt in range(3):
                try:
                    response = requests.post(
                        f"{OLLAMA_URL}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": variant},
                        timeout=120,
                    )
                    if response.status_code == 500 and "context length" in response.text and variant_idx < len(variants) - 1:
                        last_error = requests.HTTPError(response.text, response=response)
                        break
                    response.raise_for_status()
                    embeddings.append(response.json()["embedding"])
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(5)
            if last_error is None:
                break
        if last_error is not None:
            raise last_error
    return np.asarray(embeddings, dtype=np.float32)


def embed_prompts(rows: list[dict[str, Any]], batch_size: int = 50) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    checkpoint = CACHE / "embeddings-checkpoint-v4.npy"
    final_path = CACHE / "arena-v4-embeddings.npy"
    prior_v3_path = CACHE / "arena-v3-embeddings.npy"

    if final_path.exists():
        embeddings = np.load(final_path)
        if embeddings.shape[0] == len(rows):
            log(f"Step A4 embeddings loaded from final cache: {embeddings.shape}.")
            return embeddings.astype(np.float32)

    embeddings = np.full((len(rows), 768), np.nan, dtype=np.float32)
    if checkpoint.exists():
        existing = np.load(checkpoint)
        if existing.shape == embeddings.shape:
            embeddings = existing.astype(np.float32)
            completed = int(np.isfinite(embeddings[:, 0]).sum())
            log(f"Step A4 resumed v4 embeddings checkpoint with {completed} completed rows.")

    completed_before_prior = int(np.isfinite(embeddings[:, 0]).sum())
    if prior_v3_path.exists():
        prior_embeddings = np.load(prior_v3_path).astype(np.float32)
        old_indices = [i for i, row in enumerate(rows) if row["tier_matchup"] in {"Tier1vTier2", "Tier1vTier3"}]
        if len(old_indices) <= prior_embeddings.shape[0]:
            for old_pos, row_idx in enumerate(old_indices):
                if not np.isfinite(embeddings[row_idx, 0]):
                    embeddings[row_idx] = prior_embeddings[old_pos]
            completed_after_prior = int(np.isfinite(embeddings[:, 0]).sum())
            log(
                f"Step A4 reused {completed_after_prior - completed_before_prior} embeddings from v3 cache "
                f"for unchanged Tier1 rows; {len(rows) - completed_after_prior} rows remain."
            )

    missing = [i for i in range(len(rows)) if not np.isfinite(embeddings[i, 0])]
    for start in range(0, len(missing), batch_size):
        batch_indices = missing[start : start + batch_size]
        batch_num = start // batch_size + 1
        try:
            batch = embed_batch([truncate_for_embedding(rows[i]["prompt"]) for i in batch_indices])
        except Exception as e:
            completed = int(np.isfinite(embeddings[:, 0]).sum())
            raise RuntimeError(f"embedding batch {batch_num} failed after 3 retries; completed={completed}") from e
        for row_idx, vector in zip(batch_indices, batch):
            embeddings[row_idx] = vector
        np.save(checkpoint, embeddings)
        completed = int(np.isfinite(embeddings[:, 0]).sum())
        log(f"Step A4 v4 embedding checkpoint saved after batch {batch_num}; completed={completed} of {len(rows)}.")

    np.save(final_path, embeddings)
    return embeddings


def cluster_filter(rows: list[dict[str, Any]], embeddings: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    km = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=1024, n_init="auto")
    cluster_ids = km.fit_predict(embeddings)
    scores = np.asarray([r["routing_score"] for r in rows], dtype=np.float64)

    kept_clusters = set()
    cluster_stats: dict[int, dict[str, float]] = {}
    for cid in range(500):
        idx = np.where(cluster_ids == cid)[0]
        if len(idx) == 0:
            mean, std = 0.0, 1.0
        else:
            mean, std = float(scores[idx].mean()), float(scores[idx].std())
        cluster_stats[cid] = {"mean": mean, "std": std}
        if len(idx) > 0 and std <= 0.25:
            kept_clusters.add(cid)

    kept_rows: list[dict[str, Any]] = []
    kept_embeddings = []
    for i, row in enumerate(rows):
        cid = int(cluster_ids[i])
        if cid not in kept_clusters:
            continue
        std = cluster_stats[cid]["std"]
        updated = dict(row)
        updated["cluster_id"] = cid
        updated["confidence_weight"] = float(1.0 - (std / 0.25))
        kept_rows.append(updated)
        kept_embeddings.append(embeddings[i])

    kept_embeddings_arr = np.asarray(kept_embeddings, dtype=np.float32)
    with (DATA / "arena-preference-v4.jsonl").open("w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    kept_weights = np.asarray([r["confidence_weight"] for r in kept_rows], dtype=np.float64)
    stats = {
        "total_clusters": 500,
        "clusters_kept": len(kept_clusters),
        "clusters_discarded": 500 - len(kept_clusters),
        "mean_confidence_weight": float(kept_weights.mean()) if len(kept_weights) else 0.0,
        "rows_before_cluster_filter": len(rows),
        "rows_after_cluster_filter": len(kept_rows),
    }
    log(
        "Step A4 complete. "
        f"Total clusters=500, kept={stats['clusters_kept']}, discarded={stats['clusters_discarded']}, "
        f"mean_confidence_weight={stats['mean_confidence_weight']:.6f}, "
        f"final row count after cluster filter={stats['rows_after_cluster_filter']}."
    )
    return kept_rows, kept_embeddings_arr, stats


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    y = y / np.maximum(norms, 1e-12)
    return y


def split_by_cluster(rows: list[dict[str, Any]], embeddings: np.ndarray) -> dict[str, Any]:
    clusters = sorted(set(int(r["cluster_id"]) for r in rows))
    cluster_scores = {
        c: float(np.mean([r["routing_score"] for r in rows if int(r["cluster_id"]) == c]))
        for c in clusters
    }
    bins = np.asarray([min(4, int(cluster_scores[c] * 5)) for c in clusters])
    try:
        train_clusters, temp_clusters = train_test_split(clusters, test_size=0.2, random_state=42, stratify=bins)
        temp_bins = np.asarray([min(4, int(cluster_scores[c] * 5)) for c in temp_clusters])
        val_clusters, test_clusters = train_test_split(temp_clusters, test_size=0.5, random_state=42, stratify=temp_bins)
    except ValueError:
        train_clusters, temp_clusters = train_test_split(clusters, test_size=0.2, random_state=42)
        val_clusters, test_clusters = train_test_split(temp_clusters, test_size=0.5, random_state=42)

    split_clusters = {
        "train": set(train_clusters),
        "val": set(val_clusters),
        "test": set(test_clusters),
    }
    indices = {
        split: np.asarray([i for i, r in enumerate(rows) if int(r["cluster_id"]) in cs], dtype=np.int64)
        for split, cs in split_clusters.items()
    }
    assert set(indices["train"]).isdisjoint(set(indices["val"]))
    assert set(indices["train"]).isdisjoint(set(indices["test"]))
    assert set(indices["val"]).isdisjoint(set(indices["test"]))
    return {"clusters": split_clusters, "indices": indices}


class RoutingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], embeddings: np.ndarray, indices: np.ndarray, model_to_id: dict[str, int]):
        self.x = torch.tensor(embeddings[indices], dtype=torch.float32)
        self.y = torch.tensor([rows[i]["routing_score"] for i in indices], dtype=torch.float32)
        self.w = torch.tensor([rows[i]["confidence_weight"] for i in indices], dtype=torch.float32)
        self.s = torch.tensor([model_to_id[rows[i]["model_strong"]] for i in indices], dtype=torch.long)
        self.m = torch.tensor([model_to_id[rows[i]["model_weak"]] for i in indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[idx], self.s[idx], self.m[idx], self.y[idx], self.w[idx]


class MFRouter(nn.Module):
    def __init__(self, num_models: int):
        super().__init__()
        self.proj = nn.Linear(768, 64)
        self.cap = nn.Embedding(num_models, 64)

    def forward(self, x: torch.Tensor, strong_id: torch.Tensor, weak_id: torch.Tensor) -> torch.Tensor:
        prompt_latent = self.proj(x)
        diff = self.cap(strong_id) - self.cap(weak_id)
        logits = torch.sum(prompt_latent * diff, dim=1)
        return torch.sigmoid(logits)


def evaluate_bce(model: MFRouter, loader: DataLoader) -> float:
    model.eval()
    losses = []
    weights = []
    with torch.no_grad():
        for x, s, m, y, w in loader:
            pred = model(x, s, m)
            loss = F.binary_cross_entropy(pred, y, weight=w, reduction="sum")
            losses.append(loss.item())
            weights.append(w.sum().item())
    return float(sum(losses) / max(sum(weights), 1e-9))


def train_mf(rows: list[dict[str, Any]], embeddings: np.ndarray, split: dict[str, Any]) -> tuple[MFRouter, dict[str, Any], dict[str, int]]:
    models = sorted(set(r["model_strong"] for r in rows) | set(r["model_weak"] for r in rows))
    model_to_id = {m: i for i, m in enumerate(models)}
    train_ds = RoutingDataset(rows, embeddings, split["indices"]["train"], model_to_id)
    val_ds = RoutingDataset(rows, embeddings, split["indices"]["val"], model_to_id)
    train_scores = np.array([rows[i]["routing_score"] for i in split["indices"]["train"]])
    sample_weights = np.where(train_scores > 0.5, 3.0, 1.0)
    sample_weights = sample_weights / sample_weights.sum()
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    model = MFRouter(num_models=len(models))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    epoch_losses: dict[int, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    logged_epochs = {1, 5, 10, 20, 30, 40, 50}
    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        for x, s, m, y, w in train_loader:
            opt.zero_grad()
            pred = model(x, s, m)
            loss = F.binary_cross_entropy(pred, y, weight=w, reduction="sum") / torch.clamp(w.sum(), min=1e-9)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * float(w.sum().item())
            total_weight += float(w.sum().item())
        epoch_loss = total_loss / max(total_weight, 1e-9)
        val_loss = evaluate_bce(model, val_loader)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch in logged_epochs:
            epoch_losses[epoch] = epoch_loss
            log(f"Phase B MF epoch {epoch} weighted BCE train loss={epoch_loss:.6f}, val loss={val_loss:.6f}.")
        if epochs_without_improvement >= 5:
            log(f"Phase B MF early stopped at epoch {epoch}; best validation BCE={best_val_loss:.6f}.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_bce = evaluate_bce(model, DataLoader(train_ds, batch_size=256, shuffle=False))
    val_bce = evaluate_bce(model, val_loader)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_to_id": model_to_id,
            "embedding_dim": 768,
            "latent_dim": 64,
        },
        MODELS / "local-cloud-classifier-v4-mf.pt",
    )
    stats = {"epoch_losses": epoch_losses, "train_bce": train_bce, "val_bce": val_bce, "best_val_bce": best_val_loss}
    return model, stats, model_to_id


def predict_mf(model: MFRouter, rows: list[dict[str, Any]], embeddings: np.ndarray, indices: np.ndarray, model_to_id: dict[str, int]) -> np.ndarray:
    ds = RoutingDataset(rows, embeddings, indices, model_to_id)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, s, m, _, _ in loader:
            preds.extend(float(v) for v in model(x, s, m).numpy())
    return np.asarray(preds, dtype=np.float64)


def metrics_for(y_true_score: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    y_binary = (y_true_score > 0.5).astype(int)
    out: dict[str, float] = {}
    corr = spearmanr(y_true_score, preds).correlation
    out["spearman"] = float(0.0 if np.isnan(corr) else corr)
    out["roc_auc"] = float(roc_auc_score(y_binary, preds)) if len(set(y_binary)) > 1 else float("nan")
    out["brier"] = float(brier_score_loss(y_binary, np.clip(preds, 0.0, 1.0)))
    for threshold in (0.40, 0.50, 0.60):
        pred_bin = (preds >= threshold).astype(int)
        out[f"precision_at_{threshold:.2f}"] = float(precision_score(y_binary, pred_bin, zero_division=0))
        out[f"recall_at_{threshold:.2f}"] = float(recall_score(y_binary, pred_bin, zero_division=0))
    return out


def train_ensemble(rows: list[dict[str, Any]], embeddings: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    split = split_by_cluster(rows, embeddings)
    mf_model, mf_stats, model_to_id = train_mf(rows, embeddings, split)
    log(
        "Phase B complete. "
        f"MF train BCE={mf_stats['train_bce']:.6f}, val BCE={mf_stats['val_bce']:.6f}, "
        f"best val BCE={mf_stats['best_val_bce']:.6f}."
    )
    wrapper = {
        "type": "source_a_v4_mf_routing_score",
        "created": now(),
        "mf_model_path": "models/local-cloud-classifier-v4-mf.pt",
        "model_to_id": model_to_id,
        "formula": "final_score = score_mf",
        "embedding_model": EMBED_MODEL,
        "ollama_url": OLLAMA_URL,
    }
    with (MODELS / "local-cloud-classifier-v4-ensemble.pkl").open("wb") as f:
        pickle.dump(wrapper, f)
    return {"split": split, "mf_model": mf_model, "model_to_id": model_to_id}, {"mf": mf_stats}


def evaluate_arena(rows: list[dict[str, Any]], embeddings: np.ndarray, trained: dict[str, Any]) -> dict[str, Any]:
    test_idx = trained["split"]["indices"]["test"]
    y = np.asarray([rows[i]["routing_score"] for i in test_idx], dtype=np.float64)
    mf_pred = predict_mf(trained["mf_model"], rows, embeddings, test_idx, trained["model_to_id"])
    results = {
        "mf": metrics_for(y, mf_pred),
        "test_size": int(len(test_idx)),
        "test_positive_rate_score_gt_0_5": float((y > 0.5).mean()),
    }
    return results


def ollama_tags() -> list[str]:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    response.raise_for_status()
    data = response.json()
    return [m.get("name", "") for m in data.get("models", []) if m.get("name")]


def choose_weak_model(tags: list[str]) -> str:
    preferred = ["qwen3:8b", "qwen2.5:7b", "llama3.1:8b", "mistral:7b", "llama3:8b"]
    for model in preferred:
        if model in tags:
            return model
    if not tags:
        raise RuntimeError("No Ollama models available for MMLU weak-model evaluation.")
    return tags[0]


def extract_answer(text: str) -> str | None:
    text = text.strip()
    patterns = [
        r"\banswer\s*(?:is|:)\s*([ABCD])\b",
        r"\b([ABCD])\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def mmlu_prompt(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if not isinstance(choices, list):
        choices = [row.get("A"), row.get("B"), row.get("C"), row.get("D")]
    letters = ["A", "B", "C", "D"]
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))
    return (
        "Answer this multiple choice question. Return only one letter: A, B, C, or D.\n\n"
        f"Question: {row.get('question')}\n{options}\n\nAnswer:"
    )


def ask_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_predict": 8}},
        timeout=180,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def predict_ensemble_for_prompts(prompts: list[str], trained: dict[str, Any], default_strong: str, default_weak: str) -> np.ndarray:
    temp_rows = [
        {
            "prompt": p,
            "routing_score": 0.5,
            "confidence_weight": 1.0,
            "model_strong": default_strong,
            "model_weak": default_weak,
        }
        for p in prompts
    ]
    embeddings = embed_batch([truncate_for_embedding(p) for p in prompts])
    model_to_id = trained["model_to_id"]
    if default_strong not in model_to_id or default_weak not in model_to_id:
        raise RuntimeError("Default strong/weak model IDs not available in MF model mapping.")
    idx = np.arange(len(temp_rows), dtype=np.int64)
    return predict_mf(trained["mf_model"], temp_rows, embeddings, idx, model_to_id)


def evaluate_mmlu(trained: dict[str, Any], sample_size: int = 500) -> dict[str, Any]:
    sanity_prompts = [
        "What is 2+2?",
        "Define photosynthesis in one sentence.",
        "Explain quantum entanglement to a high school student.",
        "Write a Python function to merge two sorted lists.",
        "Write a Rust async TCP server.",
        "Design a distributed rate limiter for a multi-region API.",
        "Prove that there are infinitely many prime numbers.",
        "Debug a SQL query with a window function and duplicate rows.",
        "Summarize the tradeoffs between Paxos and Raft.",
        "Create a secure OAuth2 flow for a mobile banking app.",
    ]
    sanity_scores = predict_ensemble_for_prompts(sanity_prompts, trained, "gpt-4-0613", "gpt-3.5-turbo-0613")
    if float(np.max(sanity_scores)) <= 0.5:
        log(
            "FAILED pre-MMLU sanity check: all 10 fixed prompt scores were <= 0.5; "
            f"scores={[round(float(s), 6) for s in sanity_scores]}. Skipping MMLU."
        )
        return {
            "sample_size": 0,
            "weak_model": None,
            "skipped": True,
            "skip_reason": "all sanity-check routing scores <= 0.5",
            "sanity_scores": [float(s) for s in sanity_scores],
            "chosen_threshold": None,
            "routed_accuracy": 0.0,
            "over_routing_rate": 0.0,
            "under_routing_rate": 1.0,
            "details": [],
        }
    log(f"Pre-MMLU sanity check passed; max score={float(np.max(sanity_scores)):.6f}.")

    tags = ollama_tags()
    weak_model = choose_weak_model(tags)
    ds = load_dataset("cais/mmlu", "all", split="validation")
    n = min(sample_size, len(ds))
    rng = random.Random(42)
    indices = rng.sample(range(len(ds)), n)
    rows = [dict(ds[i]) for i in indices]
    prompts = [mmlu_prompt(r) for r in rows]
    scores = predict_ensemble_for_prompts(prompts, trained, "gpt-4-0613", "gpt-3.5-turbo-0613")

    local_correct_flags = []
    raw_details = []
    letters = ["A", "B", "C", "D"]
    for i, row in enumerate(rows):
        answer_idx = int(row["answer"])
        answer_letter = letters[answer_idx]
        response = ask_ollama(weak_model, prompts[i])
        local_answer = extract_answer(response)
        local_correct = local_answer == answer_letter
        local_correct_flags.append(bool(local_correct))
        raw_details.append(
            {
                "question": row["question"],
                "answer": answer_letter,
                "weak_model": weak_model,
                "weak_response": response,
                "weak_answer": local_answer,
                "weak_correct": bool(local_correct),
                "routing_score": float(scores[i]),
            }
        )
        if (i + 1) % 50 == 0:
            log(f"Phase C MMLU progress: evaluated {i + 1} of {n} questions with weak model `{weak_model}`.")

    sweep_n = min(100, n)
    eval_start = sweep_n
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    sweep_results = {}
    for threshold in thresholds:
        correct = 0
        for i in range(sweep_n):
            chose_cloud = float(scores[i]) > threshold
            correct += int(True if chose_cloud else local_correct_flags[i])
        sweep_results[str(threshold)] = correct / max(sweep_n, 1)
    chosen_threshold = max(thresholds, key=lambda t: (sweep_results[str(t)], -t))

    correct_routed = 0
    local_sufficient_cloud = 0
    cloud_needed_local = 0
    details = []
    eval_indices = range(eval_start, n)
    for i in eval_indices:
        chose_cloud = float(scores[i]) > chosen_threshold
        routed_correct = True if chose_cloud else local_correct_flags[i]
        correct_routed += int(routed_correct)
        local_sufficient_cloud += int(local_correct_flags[i] and chose_cloud)
        cloud_needed_local += int((not local_correct_flags[i]) and (not chose_cloud))
        detail = dict(raw_details[i])
        detail["router_chose_cloud"] = bool(chose_cloud)
        detail["routed_correct"] = bool(routed_correct)
        detail["threshold"] = float(chosen_threshold)
        details.append(detail)

    eval_n = max(n - eval_start, 1)
    return {
        "sample_size": n,
        "threshold_sweep_size": sweep_n,
        "evaluation_size": n - eval_start,
        "weak_model": weak_model,
        "skipped": False,
        "sanity_scores": [float(s) for s in sanity_scores],
        "chosen_threshold": float(chosen_threshold),
        "threshold_sweep": sweep_results,
        "routed_accuracy": correct_routed / eval_n,
        "over_routing_rate": local_sufficient_cloud / eval_n,
        "under_routing_rate": cloud_needed_local / eval_n,
        "details": details,
    }


def format_metric_table(results: dict[str, Any]) -> str:
    names = [
        ("Spearman correlation", "spearman"),
        ("ROC-AUC (score > 0.5 = cloud)", "roc_auc"),
        ("Brier score", "brier"),
        ("Precision @ threshold 0.40", "precision_at_0.40"),
        ("Recall @ threshold 0.40", "recall_at_0.40"),
        ("Precision @ threshold 0.50", "precision_at_0.50"),
        ("Recall @ threshold 0.50", "recall_at_0.50"),
        ("Precision @ threshold 0.60", "precision_at_0.60"),
        ("Recall @ threshold 0.60", "recall_at_0.60"),
    ]
    lines = ["| Metric | MF |", "|---|---:|"]
    for label, key in names:
        lines.append(f"| {label} | {results['mf'][key]:.6f} |")
    return "\n".join(lines)


def append_final_docs(dataset_counts: dict[str, int], cluster_stats: dict[str, Any], arena_results: dict[str, Any], mmlu_results: dict[str, Any]) -> None:
    with (DATA / "arena-preference-v4.jsonl").open(encoding="utf-8") as f:
        scores = np.asarray([json.loads(line)["routing_score"] for line in f if line.strip()])
    table = format_metric_table(arena_results)
    v4_auc = arena_results["mf"]["roc_auc"]
    status = "Complete" if (
        v4_auc > 0.65
        and max(mmlu_results.get("sanity_scores", [0.0])) > 0.5
        and mmlu_results.get("routed_accuracy", 0.0) > 0.50
        and mmlu_results.get("under_routing_rate", 1.0) < 0.80
    ) else "Target not met"
    assessment1 = (
        f"The v4 MF-only model {'met' if v4_auc > 0.65 else 'did not meet'} the Arena ROC-AUC target: "
        f"v4 ROC-AUC is {v4_auc:.6f}, compared with v1 0.5203 and v2 0.6296. "
        "The SW component was removed, so the reported score is the direct MF routing signal."
    )
    assessment2 = (
        f"The MMLU sample used local model `{mmlu_results['weak_model']}` and produced routed accuracy "
        f"{mmlu_results['routed_accuracy'] * 100:.2f}%, over-routing {mmlu_results['over_routing_rate'] * 100:.2f}%, "
        f"and under-routing {mmlu_results['under_routing_rate'] * 100:.2f}% at threshold {mmlu_results.get('chosen_threshold')}. "
        "This out-of-distribution result indicates whether Arena preference data transfers to exam-style questions; if under-routing remains material or ROC-AUC is below target, Source A alone is not sufficient and Source B augmentation remains required before production routing."
    )
    final = f"""---
## Final summary

**Completed:** {now()}
**Status:** {status}

**Dataset**
- Raw battles loaded: {dataset_counts['raw']}
- Surviving after all filters: {dataset_counts['after_tier_matchup_filter']}
- Surviving after cluster filter: {cluster_stats['rows_after_cluster_filter']}
- Tier1vTier2 battles: {dataset_counts['Tier1vTier2_after_filters']}
- Tier1vTier3 battles: {dataset_counts['Tier1vTier3_after_filters']}
- Tier2vTier3 battles: {dataset_counts['Tier2vTier3_after_filters']}

**Routing score distribution**
- Mean: {scores.mean():.6f}   Std: {scores.std():.6f}   Min: {scores.min():.6f}   Max: {scores.max():.6f}

**Model performance**
{table}

**MMLU routing accuracy:** {mmlu_results['routed_accuracy'] * 100:.2f}%
**MMLU chosen threshold:** {mmlu_results.get('chosen_threshold')}
**Over-routing rate:** {mmlu_results['over_routing_rate'] * 100:.2f}%
**Under-routing rate:** {mmlu_results['under_routing_rate'] * 100:.2f}%

**vs baselines**
- v1 ROC-AUC: 0.5203  →  v4: {v4_auc:.6f}
- v2 ROC-AUC: 0.6296  →  v4: {v4_auc:.6f}

**Artifacts**
- data/arena-preference-v4.jsonl
- models/local-cloud-classifier-v4-mf.pt
- models/local-cloud-classifier-v4-ensemble.pkl
- data/local-cloud-classifier-v4-evaluation.json

**Assessment**
{assessment1}

{assessment2}
---"""
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write("\n" + final + "\n")
    with (DOCS / "progress.md").open("a", encoding="utf-8") as f:
        f.write(f"\nSource A v4 MF routing score fix — {status} — ROC-AUC: {v4_auc:.6f} — see docs/srcA_v4.md\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-mmlu", action="store_true")
    parser.add_argument("--a3-test-rows", type=int, default=0)
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        raw_rows = load_arena()
        build_tier_map(raw_rows)
    except Exception as e:
        fail("Step A1", e)

    try:
        prepared, dataset_counts = filter_and_prepare(raw_rows)
        if args.a3_test_rows:
            prepared = prepared[: args.a3_test_rows]
            dataset_counts["after_tier_matchup_filter"] = len(prepared)
            dataset_counts["Tier1vTier2_after_filters"] = sum(1 for r in prepared if r["tier_matchup"] == "Tier1vTier2")
            dataset_counts["Tier1vTier3_after_filters"] = sum(1 for r in prepared if r["tier_matchup"] == "Tier1vTier3")
            dataset_counts["Tier2vTier3_after_filters"] = sum(1 for r in prepared if r["tier_matchup"] == "Tier2vTier3")
            log(f"Step A3 real-data validation mode enabled with {len(prepared)} rows.")
    except Exception as e:
        fail("Step A2", e)

    try:
        scored = compute_scores(prepared)
        if args.a3_test_rows:
            log(f"Step A3 real-data validation completed successfully for {len(scored)} rows; stopping before Step A4 by design.")
            return
    except Exception as e:
        fail("Step A3", e)

    try:
        verify_ollama()
        embeddings = embed_prompts(scored)
        kept_rows, kept_embeddings, cluster_stats = cluster_filter(scored, embeddings)
    except Exception as e:
        fail("Step A4", e)

    try:
        trained, train_stats = train_ensemble(kept_rows, kept_embeddings)
    except Exception as e:
        fail("Phase B", e)

    try:
        arena_results = evaluate_arena(kept_rows, kept_embeddings, trained)
        table = format_metric_table(arena_results)
        append_section(
            "Phase C — Arena held-out test evaluation",
            f"{table}\n\nCompared against v1 ROC-AUC 0.5203 and v2 ROC-AUC 0.6296. Test size: {arena_results['test_size']}. Test positive rate: {arena_results['test_positive_rate_score_gt_0_5']:.6f}.",
        )
        if args.skip_mmlu:
            raise RuntimeError("MMLU evaluation skipped by explicit --skip-mmlu flag; this is not allowed for final v4 completion.")
        mmlu_results = evaluate_mmlu(trained)
        append_section(
            "Phase C — MMLU validation evaluation",
            f"Sample size: {mmlu_results['sample_size']}\n\nWeak model: `{mmlu_results['weak_model']}`\n\nChosen threshold: {mmlu_results.get('chosen_threshold')}\n\nRouted accuracy: {mmlu_results['routed_accuracy'] * 100:.2f}%\n\nOver-routing rate: {mmlu_results['over_routing_rate'] * 100:.2f}%\n\nUnder-routing rate: {mmlu_results['under_routing_rate'] * 100:.2f}%",
        )
        eval_out = {
            "arena": arena_results,
            "mmlu": mmlu_results,
            "training": train_stats,
            "dataset_counts": dataset_counts,
            "cluster_stats": cluster_stats,
        }
        (DATA / "local-cloud-classifier-v4-evaluation.json").write_text(json.dumps(eval_out, indent=2), encoding="utf-8")
        append_final_docs(dataset_counts, cluster_stats, arena_results, mmlu_results)
    except Exception as e:
        fail("Phase C", e)


if __name__ == "__main__":
    main()
