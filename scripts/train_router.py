#!/usr/bin/env python3
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = PROJECT_ROOT / "data" / "router-training.json"
EVAL_PATH = PROJECT_ROOT / "data" / "router-eval.json"
MODEL_PATH = PROJECT_ROOT / "data" / "router-model.json"
TARGETS = ["task_type", "difficulty", "privacy", "route_class"]
TRAIN_TEST_SEED = 42
TEST_RATIO = 0.2
MIN_TRAINING_EXAMPLES = 500
MIN_TOKEN_COUNT = 2
ALPHA = 0.7
CHAR_NGRAM_RANGE = (4, 5)
CENTROID_WEIGHT = 1.0
NAIVE_BAYES_WEIGHT = 0.0

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "me", "my", "of", "on", "or", "that", "the",
    "this", "to", "with"
}
PRIVACY_TERMS = [
    "private", "confidential", "secret", "password", "api key", "token", "ssn",
    "social security", "medical", "diagnosis", "bank", "credit card", "personal",
    "address", "phone number", "email", "legal", "contract", "customer", "journal",
    "employee", "salary", "payroll", "invoice", "tax", "passport", "driver license",
    "patient", "therapy", "insurance", "proprietary", "internal", "nda",
]
CODING_TERMS = [
    "code", "function", "debug", "bug", "javascript", "typescript", "python",
    "react", "node", "api", "sql", "stack trace", "refactor", "component",
    "repository", "endpoint", "schema", "database", "unit test", "pull request",
    "typescript", "frontend", "backend", "middleware", "exception", "traceback",
]
COMPLEX_TERMS = [
    "architecture", "multi-step", "deep", "detailed", "optimize", "tradeoff",
    "compare", "reason", "reasoning", "proof", "math", "logic", "design",
    "strategy", "analyze", "implement end to end", "scalable", "distributed",
    "root cause", "performance", "migration", "system design", "security review",
    "research paper", "long transcript", "requirements", "roadmap", "dependencies",
]
SUMMARY_TERMS = ["summarize", "summary", "rewrite", "shorter", "condense", "extract action items", "meeting notes", "transcript"]
TRANSLATION_TERMS = ["translate", "translation", "spanish", "french", "german", "italian", "preserve terminology"]
CREATIVE_TERMS = ["write", "draft", "brainstorm", "story", "poem", "tagline", "marketing", "copy", "announcement", "subject line"]
DATA_TERMS = ["csv", "data", "dataset", "table", "metrics", "trends", "anomalies", "cohort", "sentiment", "classify", "categorize"]
PLANNING_TERMS = ["plan", "roadmap", "milestones", "strategy", "dependencies", "launch", "schedule", "checklist", "migration plan"]
MATH_TERMS = ["math", "algebra", "proof", "equation", "probability", "statistics", "calculation", "logic puzzle", "optimization problem"]
REASONING_TERMS = ["reason", "analyze", "compare", "evaluate", "tradeoff", "pros and cons", "risks", "weak points", "argument"]


def main():
    training_data = json.loads(TRAINING_PATH.read_text())
    eval_data = json.loads(EVAL_PATH.read_text())
    validate_training_data(training_data)
    validate_training_data(eval_data)

    base_examples = training_data["examples"]
    augmented_examples = augment_examples()
    validate_examples({**training_data, "examples": base_examples + augmented_examples})

    if len(base_examples) + len(augmented_examples) < MIN_TRAINING_EXAMPLES:
        raise ValueError(f"Router training set must include at least {MIN_TRAINING_EXAMPLES} examples.")

    examples = [
        {**example, "tokens": tokenize(example["text"])}
        for example in base_examples + augmented_examples
    ]
    eval_examples = [
        {**example, "tokens": tokenize(example["text"])}
        for example in eval_data["examples"]
    ]
    full_vocabulary = build_vocabulary(examples)
    holdout_metrics = {}
    split_summary = {}

    for target in TARGETS:
        target_train_examples, target_test_examples = split_examples(examples, target)
        target_train_vocabulary = build_vocabulary(target_train_examples)
        evaluation_classifier = train_naive_bayes(
            target_train_examples,
            target_train_vocabulary,
            target,
            training_data["labels"][target],
        )
        evaluation_centroid = train_centroid(
            target_train_examples,
            target_train_vocabulary,
            target,
            training_data["labels"][target],
        )
        holdout_metrics[target] = {
            "naive_bayes": evaluate_target_naive_bayes(target_test_examples, evaluation_classifier, target),
            "centroid": evaluate_target_centroid(target_test_examples, evaluation_centroid, target),
            "hybrid": evaluate_target_hybrid(target_test_examples, evaluation_classifier, evaluation_centroid, target),
        }
        split_summary[target] = {
            "train_examples": len(target_train_examples),
            "test_examples": len(target_test_examples),
        }

    final_classifiers = {
        target: train_naive_bayes(
            examples,
            full_vocabulary,
            target,
            training_data["labels"][target],
        )
        for target in TARGETS
    }
    final_centroid_classifiers = {
        target: train_centroid(
            examples,
            full_vocabulary,
            target,
            training_data["labels"][target],
        )
        for target in TARGETS
    }

    model = {
        "schema_version": 3,
        "model_type": "python_hybrid_naive_bayes_centroid",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": len(examples),
        "base_training_examples": len(base_examples),
        "augmented_training_examples": len(augmented_examples),
        "eval_examples": len(eval_examples),
        "training_script": "scripts/train_router.py",
        "preprocessing": {
            "lowercase": True,
            "token_pattern": "alphanumeric_plus_hash_dot",
            "word_ngrams": [1, 2],
            "char_ngrams": list(CHAR_NGRAM_RANGE),
            "engineered_features": ["privacy_terms", "coding_terms", "complex_terms", "prompt_length"],
            "stop_words": sorted(STOP_WORDS),
            "min_token_count": MIN_TOKEN_COUNT,
            "alpha": ALPHA,
        },
        "ensemble": {
            "naive_bayes_weight": NAIVE_BAYES_WEIGHT,
            "centroid_weight": CENTROID_WEIGHT,
        },
        "split": {
            "seed": TRAIN_TEST_SEED,
            "test_ratio": TEST_RATIO,
            "by_target": split_summary,
        },
        "vocabulary": full_vocabulary,
        "targets": TARGETS,
        "classifiers": final_classifiers if NAIVE_BAYES_WEIGHT > 0 else None,
        "centroid_classifiers": final_centroid_classifiers,
        "metrics": {
            "train": evaluate(examples, final_classifiers, final_centroid_classifiers),
            "holdout": holdout_metrics,
            "eval": evaluate(eval_examples, final_classifiers, final_centroid_classifiers),
        },
        "top_tokens": {
            target: top_tokens(final_classifiers[target], limit=8)
            for target in TARGETS
        },
    }

    MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n")
    print_summary(model)


def validate_training_data(training_data):
    examples = training_data.get("examples")
    labels = training_data.get("labels")

    if not isinstance(examples, list) or not examples:
        raise ValueError("router-training.json must include non-empty examples.")

    for target in TARGETS:
        if not isinstance(labels.get(target), list):
            raise ValueError(f"router-training.json is missing labels.{target}.")

    validate_examples(training_data)


def validate_examples(training_data):
    examples = training_data["examples"]
    labels = training_data["labels"]

    for index, example in enumerate(examples):
        if not isinstance(example.get("text"), str) or not example["text"].strip():
            raise ValueError(f"Example {index} is missing text.")

        for target in TARGETS:
            if example.get(target) not in labels[target]:
                raise ValueError(f"Example {index} has invalid {target}: {example.get(target)}")


def augment_examples():
    examples = []

    def add(texts, task_type, difficulty, privacy, route_class):
        for text in texts:
            examples.append({
                "text": text,
                "task_type": task_type,
                "difficulty": difficulty,
                "privacy": privacy,
                "route_class": route_class,
            })

    add([
        "Define cache invalidation in one short sentence.",
        "What is JSON and why is it used?",
        "Explain latency to a non technical person.",
        "List three benefits of using keyboard shortcuts.",
        "Give me a quick explanation of what DNS does.",
        "What does HTTP status code 404 mean?",
        "Explain the difference between RAM and storage.",
        "Give a simple answer to this general knowledge question.",
    ], "simple_qa", "easy", "low", "local_tiny")

    add([
        "Summarize this public paragraph into one sentence.",
        "Rewrite this short announcement so it is clearer.",
        "Turn these public notes into three bullets.",
        "Make this short product description more concise.",
        "Extract the main idea from this brief article.",
        "Clean up this harmless message for grammar.",
    ], "summarization", "easy", "low", "local_tiny")

    add([
        "Summarize this internal team update with employee names.",
        "Rewrite this customer email response professionally.",
        "Condense these meeting notes with account details.",
        "Extract action items from this private staff discussion.",
        "Summarize my personal notes about a family issue.",
        "Make this confidential memo shorter.",
    ], "summarization", "medium", "high", "local_general")

    add([
        "Classify these survey responses by sentiment.",
        "Group these support tickets by topic and priority.",
        "Find anomalies in this small table of numbers.",
        "Categorize these customer comments into themes.",
        "Compare these monthly metrics and call out changes.",
        "Extract entities and dates from these notes.",
    ], "data_analysis", "medium", "medium", "local_general")

    add([
        "Analyze this CSV and recommend statistical tests.",
        "Find trends, outliers, and correlations in this public dataset.",
        "Build a detailed interpretation of this experiment table.",
        "Compare multiple cohorts and explain likely drivers.",
        "Evaluate this A/B test and list limitations.",
        "Inspect this analytics export and produce findings.",
    ], "data_analysis", "hard", "low", "cloud_strong")

    add([
        "Translate this short public message to Spanish.",
        "Translate this casual sentence to French.",
        "Translate this basic instruction into German.",
        "Convert this greeting into Italian.",
        "Translate this short product label.",
    ], "translation", "easy", "low", "local_tiny")

    add([
        "Translate this legal email that includes my address.",
        "Translate this private medical note into plain English.",
        "Translate this confidential contract clause to French.",
        "Translate this customer complaint with personal details.",
        "Translate this internal HR message carefully.",
    ], "translation", "medium", "high", "local_general")

    add([
        "Brainstorm ten names for a simple todo app.",
        "Write a short thank you note.",
        "Create three friendly subject lines.",
        "Draft a small birthday message.",
        "Write a two sentence product blurb.",
    ], "creative", "easy", "low", "local_tiny")

    add([
        "Write a polished blog introduction with a professional tone.",
        "Create several marketing taglines with distinct voices.",
        "Draft a thoughtful product announcement.",
        "Improve this landing page copy for clarity and energy.",
        "Write a short story scene with dialogue.",
    ], "creative", "medium", "low", "cloud_fast")

    add([
        "Write a JavaScript debounce function with comments.",
        "Fix this React useEffect dependency bug.",
        "Explain this Python traceback and suggest a patch.",
        "Generate unit tests for this TypeScript helper.",
        "Refactor this API handler for readability.",
        "Write a SQL query to group orders by month.",
        "Debug this Node middleware that returns 500.",
        "Create a small React form component.",
        "Explain why this async function is returning undefined.",
        "Review this pull request diff for obvious bugs.",
    ], "coding", "medium", "low", "local_coder")

    add([
        "Fix this private payroll script locally.",
        "Debug this internal API endpoint with customer data.",
        "Review this proprietary TypeScript module without cloud.",
        "Explain this stack trace from our private repository.",
        "Generate tests for this confidential billing function.",
        "Find the issue in this local-only database migration.",
    ], "coding", "medium", "high", "local_coder")

    add([
        "Find the root cause across these three services.",
        "Design a migration from REST to GraphQL for a large codebase.",
        "Review the architecture of this public repository.",
        "Optimize this distributed job pipeline and explain tradeoffs.",
        "Plan an end to end refactor of this frontend application.",
        "Compare two database schema designs for scale.",
        "Analyze this complex production incident timeline.",
    ], "coding", "hard", "low", "cloud_strong")

    add([
        "Solve this algebra problem step by step.",
        "Check this probability calculation for mistakes.",
        "Explain the logic puzzle solution carefully.",
        "Walk through this medium difficulty proof.",
        "Reason about this scheduling constraint problem.",
    ], "math", "medium", "low", "local_reasoning")

    add([
        "Solve this advanced proof and identify hidden assumptions.",
        "Work through this hard optimization problem.",
        "Analyze this complex logic argument in detail.",
        "Compare multiple mathematical approaches and choose one.",
        "Solve this difficult statistics problem with caveats.",
    ], "math", "hard", "low", "cloud_strong")

    add([
        "Reason through this private family budget question.",
        "Analyze this confidential legal argument locally.",
        "Evaluate this sensitive medical decision note.",
        "Review this personal financial scenario without cloud.",
        "Think through this private employment negotiation.",
    ], "reasoning", "hard", "high", "local_reasoning")

    add([
        "Compare two public product strategies.",
        "Explain tradeoffs between speed and quality for this project.",
        "Analyze whether this business idea is likely to work.",
        "Evaluate the pros and cons of this technical decision.",
        "Reason through possible causes of this operational problem.",
    ], "reasoning", "hard", "low", "cloud_strong")

    add([
        "Make a simple weekly meal plan.",
        "Organize these errands into a daily schedule.",
        "Create a basic checklist for moving apartments.",
        "Plan a short study routine.",
        "Outline a simple weekend project.",
    ], "planning", "medium", "low", "cloud_fast")

    add([
        "Design a multi quarter launch plan with risks and dependencies.",
        "Create a migration roadmap for a large SaaS product.",
        "Plan a complex architecture rollout across teams.",
        "Build a detailed project strategy with milestones and tradeoffs.",
        "Design a scalable authentication architecture.",
    ], "planning", "hard", "low", "cloud_strong")

    add([
        "Summarize this very long transcript with decisions and owners.",
        "Analyze this long research paper and list methods and limitations.",
        "Extract requirements from a long technical specification.",
        "Compare every section of this long public policy document.",
        "Turn this long implementation document into a roadmap.",
        "Synthesize a long product requirements document.",
    ], "summarization", "hard", "low", "cloud_long_context")

    add([
        "Review this contract with my name, address, and payment details.",
        "Explain this diagnosis report and medication note.",
        "Summarize this bank statement and categorize spending.",
        "Analyze this customer export with emails and phone numbers.",
        "Help with this confidential employee performance review.",
        "What should I do after leaking this API key?",
        "Redact sensitive information from this private document.",
    ], "reasoning", "medium", "high", "local_general")

    examples.extend(generate_matrix_examples())
    return dedupe_examples(examples)


def generate_matrix_examples():
    generated = []

    def add_group(templates, subjects, limit, task_type, difficulty, privacy, route_class):
        count = 0

        for template in templates:
            for subject in subjects:
                generated.append({
                    "text": template.format(subject=subject),
                    "task_type": task_type,
                    "difficulty": difficulty,
                    "privacy": privacy,
                    "route_class": route_class,
                })
                count += 1

                if count >= limit:
                    return

    simple_subjects = [
        "DNS", "JSON", "OAuth", "latency", "cache invalidation", "HTTP 404",
        "environment variables", "RAM", "vector databases", "rate limits",
    ]
    add_group([
        "Define {subject} in one short sentence.",
        "Explain {subject} for a beginner.",
        "Give me a quick answer about {subject}.",
        "List three basic facts about {subject}.",
    ], simple_subjects, 36, "simple_qa", "easy", "low", "local_tiny")

    summary_public_subjects = [
        "a product update", "a public blog post", "a release note", "a short article",
        "a help center page", "a public changelog", "a conference abstract", "a newsletter",
    ]
    add_group([
        "Summarize {subject} into three bullets.",
        "Rewrite {subject} to be clearer and shorter.",
        "Extract the main point from {subject}.",
        "Condense {subject} into one paragraph.",
    ], summary_public_subjects, 32, "summarization", "easy", "low", "local_tiny")

    summary_private_subjects = [
        "an internal employee memo", "customer meeting notes", "a confidential sales call",
        "a private journal entry", "a staff performance note", "a support case with emails",
        "a legal intake note", "a patient follow up note",
    ]
    add_group([
        "Summarize {subject} without using cloud models.",
        "Extract action items from {subject}.",
        "Rewrite {subject} more professionally.",
        "Condense {subject} while preserving sensitive details.",
    ], summary_private_subjects, 34, "summarization", "medium", "high", "local_general")

    translation_public_subjects = [
        "a short greeting", "a product label", "a public announcement", "a travel phrase",
        "a menu description", "a short instruction", "a store sign", "a casual text",
    ]
    add_group([
        "Translate {subject} to Spanish.",
        "Translate {subject} to French.",
        "Translate {subject} to German.",
        "Translate {subject} into natural English.",
    ], translation_public_subjects, 28, "translation", "easy", "low", "local_tiny")

    translation_private_subjects = [
        "a legal email with my address", "a medical note", "an HR message",
        "a customer complaint with phone numbers", "a contract clause", "a payroll question",
    ]
    add_group([
        "Translate {subject} carefully and keep it private.",
        "Translate {subject} while preserving formal tone.",
        "Translate {subject} without sending it to cloud.",
        "Translate {subject} and keep names intact.",
    ], translation_private_subjects, 28, "translation", "medium", "high", "local_general")

    creative_easy_subjects = [
        "app names", "birthday messages", "thank you notes", "short taglines",
        "subject lines", "meeting icebreakers", "team slogans", "two sentence blurbs",
    ]
    add_group([
        "Brainstorm ten {subject}.",
        "Write five friendly {subject}.",
        "Draft a short set of {subject}.",
        "Create simple {subject} with a warm tone.",
    ], creative_easy_subjects, 28, "creative", "easy", "low", "local_tiny")

    creative_medium_subjects = [
        "a launch announcement", "a landing page section", "a brand story",
        "a marketing email", "a blog introduction", "a short scene with dialogue",
    ]
    add_group([
        "Write a polished {subject}.",
        "Create three versions of {subject} with different tones.",
        "Improve {subject} for clarity and persuasion.",
        "Draft {subject} for a public audience.",
    ], creative_medium_subjects, 28, "creative", "medium", "low", "cloud_fast")

    data_medium_subjects = [
        "support tickets", "survey responses", "small sales numbers", "customer comments",
        "feature requests", "bug reports", "feedback snippets", "weekly metrics",
    ]
    add_group([
        "Classify these {subject} by theme.",
        "Find trends and anomalies in these {subject}.",
        "Group these {subject} by priority.",
        "Extract useful categories from these {subject}.",
    ], data_medium_subjects, 36, "data_analysis", "medium", "medium", "local_general")

    data_hard_subjects = [
        "A/B test results", "cohort retention data", "experiment metrics",
        "analytics exports", "research survey tables", "financial model outputs",
    ]
    add_group([
        "Analyze {subject} and explain statistical limitations.",
        "Find correlations and likely drivers in {subject}.",
        "Recommend statistical tests for {subject}.",
        "Produce detailed findings from {subject}.",
    ], data_hard_subjects, 32, "data_analysis", "hard", "low", "cloud_strong")

    coding_medium_subjects = [
        "React state bug", "TypeScript helper", "Node API handler", "SQL query",
        "Python traceback", "unit test suite", "frontend component", "Express middleware",
        "database migration", "async JavaScript function",
    ]
    add_group([
        "Debug this {subject} and suggest a fix.",
        "Refactor this {subject} for readability.",
        "Generate tests for this {subject}.",
        "Explain what is wrong with this {subject}.",
    ], coding_medium_subjects, 44, "coding", "medium", "low", "local_coder")

    coding_private_subjects = [
        "internal billing endpoint", "private payroll script", "proprietary TypeScript module",
        "customer-data migration", "local-only auth middleware", "confidential reporting job",
    ]
    add_group([
        "Debug this {subject} locally.",
        "Review this {subject} without using cloud.",
        "Generate tests for this {subject} with sensitive data.",
        "Find the bug in this {subject} and keep it private.",
    ], coding_private_subjects, 32, "coding", "medium", "high", "local_coder")

    coding_hard_subjects = [
        "distributed job pipeline", "large React application", "multi-service incident",
        "database schema migration", "public open source architecture", "performance bottleneck",
        "GraphQL migration", "authentication system",
    ]
    add_group([
        "Find the root cause across this {subject}.",
        "Design an end to end refactor for this {subject}.",
        "Compare architecture options for this {subject}.",
        "Optimize this {subject} and explain tradeoffs.",
    ], coding_hard_subjects, 36, "coding", "hard", "low", "cloud_strong")

    math_medium_subjects = [
        "algebra solution", "probability calculation", "logic puzzle", "linear equation",
        "statistics homework", "geometry proof sketch", "constraint problem",
    ]
    add_group([
        "Solve this {subject} step by step.",
        "Check this {subject} for mistakes.",
        "Explain the reasoning in this {subject}.",
        "Walk through this {subject} carefully.",
    ], math_medium_subjects, 30, "math", "medium", "low", "local_reasoning")

    math_hard_subjects = [
        "advanced proof", "optimization problem", "Bayesian statistics problem",
        "complex probability derivation", "formal logic argument", "linear algebra proof",
    ]
    add_group([
        "Solve this {subject} and identify hidden assumptions.",
        "Compare approaches for this {subject}.",
        "Work through this hard {subject} with caveats.",
        "Validate each step of this {subject}.",
    ], math_hard_subjects, 30, "math", "hard", "low", "cloud_strong")

    reasoning_hard_public_subjects = [
        "business strategy", "technical tradeoff", "product bet", "operations failure",
        "market entry plan", "security proposal", "vendor decision", "pricing strategy",
    ]
    add_group([
        "Analyze this {subject} and explain risks.",
        "Compare options for this {subject}.",
        "Evaluate whether this {subject} is likely to work.",
        "Reason through the tradeoffs in this {subject}.",
    ], reasoning_hard_public_subjects, 34, "reasoning", "hard", "low", "cloud_strong")

    reasoning_private_subjects = [
        "family budget", "employment negotiation", "medical decision note",
        "legal argument", "private investment scenario", "confidential HR issue",
    ]
    add_group([
        "Reason through this private {subject} locally.",
        "Analyze this confidential {subject} without cloud.",
        "Evaluate risks in this sensitive {subject}.",
        "Compare options for this private {subject}.",
    ], reasoning_private_subjects, 30, "reasoning", "hard", "high", "local_reasoning")

    planning_medium_subjects = [
        "weekly meal plan", "study routine", "moving checklist", "workout schedule",
        "small project plan", "trip itinerary", "content calendar", "daily routine",
    ]
    add_group([
        "Create a practical {subject}.",
        "Organize this into a simple {subject}.",
        "Make a step by step {subject}.",
        "Draft a useful {subject} with priorities.",
    ], planning_medium_subjects, 30, "planning", "medium", "low", "cloud_fast")

    planning_hard_subjects = [
        "multi-quarter launch roadmap", "SaaS migration plan", "architecture rollout",
        "cross-team project strategy", "enterprise security program", "platform rebuild",
        "product expansion roadmap", "incident response program",
    ]
    add_group([
        "Design a detailed {subject} with milestones and risks.",
        "Create a dependency-aware {subject}.",
        "Build a strategic {subject} with tradeoffs.",
        "Plan an end to end {subject}.",
    ], planning_hard_subjects, 36, "planning", "hard", "low", "cloud_strong")

    long_context_subjects = [
        "technical specification", "research paper", "meeting transcript",
        "product requirements document", "policy document", "implementation plan",
        "public design review", "vendor proposal",
    ]
    add_group([
        "Analyze this very long {subject} and summarize decisions.",
        "Extract requirements from this long {subject}.",
        "Compare all sections of this long {subject}.",
        "Turn this long {subject} into a structured roadmap.",
    ], long_context_subjects, 34, "summarization", "hard", "low", "cloud_long_context")

    private_general_subjects = [
        "bank statement", "contract with payment details", "diagnosis report",
        "employee review", "customer export", "tax letter", "insurance claim",
        "document containing an API key",
    ]
    add_group([
        "Explain this private {subject} in plain language.",
        "Summarize this confidential {subject} locally.",
        "Redact sensitive details from this {subject}.",
        "Help me understand this personal {subject}.",
    ], private_general_subjects, 34, "reasoning", "medium", "high", "local_general")

    return generated


def dedupe_examples(examples):
    seen = set()
    unique_examples = []

    for example in examples:
        key = example["text"].lower()

        if key in seen:
            continue

        seen.add(key)
        unique_examples.append(example)

    return unique_examples


def tokenize(text):
    normalized = "".join(
        character.lower() if character.isalnum() or character in {"#", ".", "+"} else " "
        for character in text
    ).strip()

    if not normalized:
        return []

    words = [
        word
        for word in normalized.split()
        if len(word) > 1 and word not in STOP_WORDS
    ]
    tokens = list(words)

    for index in range(len(words) - 1):
        tokens.append(f"{words[index]}_{words[index + 1]}")

    compact_text = " ".join(words)
    min_char_ngram, max_char_ngram = CHAR_NGRAM_RANGE

    for size in range(min_char_ngram, max_char_ngram + 1):
        for index in range(max(0, len(compact_text) - size + 1)):
            ngram = compact_text[index:index + size]

            if " " not in ngram:
                tokens.append(f"char:{ngram}")

    if contains_any(normalized, PRIVACY_TERMS):
        tokens.extend(["feature:privacy", "feature:privacy"])

    if contains_any(normalized, CODING_TERMS):
        tokens.extend(["feature:coding", "feature:coding"])

    if contains_any(normalized, COMPLEX_TERMS) or len(words) > 80:
        tokens.extend(["feature:complex", "feature:complex"])

    task_feature_terms = {
        "summary": SUMMARY_TERMS,
        "translation": TRANSLATION_TERMS,
        "creative": CREATIVE_TERMS,
        "data": DATA_TERMS,
        "planning": PLANNING_TERMS,
        "math": MATH_TERMS,
        "reasoning": REASONING_TERMS,
    }

    for feature_name, terms in task_feature_terms.items():
        if contains_any(normalized, terms):
            tokens.extend([f"feature:{feature_name}", f"feature:{feature_name}"])

    if len(words) < 12:
        tokens.append("feature:short_prompt")
    elif len(words) > 80:
        tokens.append("feature:long_prompt")

    return tokens


def contains_any(text, terms):
    return any(term in text for term in terms)


def split_examples(examples, stratify_key):
    rng = random.Random(TRAIN_TEST_SEED)
    grouped = defaultdict(list)

    for example in examples:
        grouped[example[stratify_key]].append(example)

    train_examples = []
    test_examples = []

    for group in grouped.values():
        shuffled_group = list(group)
        rng.shuffle(shuffled_group)
        test_size = max(1, round(len(shuffled_group) * TEST_RATIO)) if len(shuffled_group) >= 3 else 0
        test_examples.extend(shuffled_group[:test_size])
        train_examples.extend(shuffled_group[test_size:])

    rng.shuffle(train_examples)
    rng.shuffle(test_examples)
    return train_examples, test_examples


def build_vocabulary(examples):
    counts = Counter()

    for example in examples:
        counts.update(example["tokens"])

    return sorted(
        token
        for token, count in counts.items()
        if count >= MIN_TOKEN_COUNT
    )


def train_naive_bayes(examples, vocabulary, target, labels):
    vocabulary_set = set(vocabulary)
    stats = {
        label: {
            "example_count": 0,
            "token_total": 0,
            "token_counts": Counter(),
        }
        for label in labels
    }

    for example in examples:
        label_stats = stats[example[target]]
        label_stats["example_count"] += 1

        for token in example["tokens"]:
            if token not in vocabulary_set:
                continue

            label_stats["token_total"] += 1
            label_stats["token_counts"][token] += 1

    classes = {}
    vocabulary_size = len(vocabulary)

    for label in labels:
        label_stats = stats[label]
        denominator = label_stats["token_total"] + ALPHA * vocabulary_size
        token_log_likelihoods = {}

        for token in vocabulary:
            numerator = label_stats["token_counts"][token] + ALPHA
            token_log_likelihoods[token] = math.log(numerator / denominator)

        classes[label] = {
            "log_prior": math.log((label_stats["example_count"] + ALPHA) / (len(examples) + ALPHA * len(labels))),
            "unknown_log_likelihood": math.log(ALPHA / denominator),
            "token_log_likelihoods": token_log_likelihoods,
        }

    return {
        "labels": labels,
        "classes": classes,
    }


def train_centroid(examples, vocabulary, target, labels):
    idf = build_idf(examples, vocabulary)
    sums = {label: defaultdict(float) for label in labels}
    counts = {label: 0 for label in labels}

    for example in examples:
        label = example[target]
        vector = vectorize_tokens(example["tokens"], idf)
        counts[label] += 1

        for token, value in vector.items():
            sums[label][token] += value

    classes = {}

    for label in labels:
        if counts[label] == 0:
            classes[label] = {"centroid": {}}
            continue

        centroid = {
            token: value / counts[label]
            for token, value in sums[label].items()
        }
        centroid = normalize_vector(centroid)
        classes[label] = {
            "centroid": {
                token: round(value, 6)
                for token, value in centroid.items()
                if abs(value) >= 0.00001
            }
        }

    return {
        "labels": labels,
        "idf": {token: round(value, 6) for token, value in idf.items()},
        "classes": classes,
    }


def build_idf(examples, vocabulary):
    document_frequency = Counter()
    vocabulary_set = set(vocabulary)

    for example in examples:
        document_frequency.update(set(token for token in example["tokens"] if token in vocabulary_set))

    document_count = len(examples)
    return {
        token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1
        for token in vocabulary
    }


def vectorize_tokens(tokens, idf):
    counts = Counter(token for token in tokens if token in idf)
    vector = {
        token: (1 + math.log(count)) * idf[token]
        for token, count in counts.items()
    }
    return normalize_vector(vector)


def normalize_vector(vector):
    norm = math.sqrt(sum(value * value for value in vector.values()))

    if norm == 0:
        return {}

    return {
        token: value / norm
        for token, value in vector.items()
    }


def evaluate(examples, classifiers, centroid_classifiers):
    metrics = {}

    for target in TARGETS:
        metrics[target] = {
            "naive_bayes": evaluate_target_naive_bayes(examples, classifiers[target], target),
            "centroid": evaluate_target_centroid(examples, centroid_classifiers[target], target),
            "hybrid": evaluate_target_hybrid(examples, classifiers[target], centroid_classifiers[target], target),
        }

    return metrics


def evaluate_target_naive_bayes(examples, classifier, target):
    labels = classifier["labels"]
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    correct = 0

    for example in examples:
        prediction = predict_naive_bayes(example["text"], classifier)
        actual = example[target]
        matrix[actual][prediction["label"]] += 1

        if prediction["label"] == actual:
            correct += 1

    return {
        "correct": correct,
        "total": len(examples),
        "accuracy": round(correct / len(examples), 3) if examples else 0,
        "confusion_matrix": matrix,
    }


def evaluate_target_centroid(examples, classifier, target):
    labels = classifier["labels"]
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    correct = 0

    for example in examples:
        prediction = predict_centroid(example["text"], classifier)
        actual = example[target]
        matrix[actual][prediction["label"]] += 1

        if prediction["label"] == actual:
            correct += 1

    return {
        "correct": correct,
        "total": len(examples),
        "accuracy": round(correct / len(examples), 3) if examples else 0,
        "confusion_matrix": matrix,
    }


def evaluate_target_hybrid(examples, naive_bayes_classifier, centroid_classifier, target):
    labels = naive_bayes_classifier["labels"]
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    correct = 0

    for example in examples:
        prediction = predict_hybrid(example["text"], naive_bayes_classifier, centroid_classifier)
        actual = example[target]
        matrix[actual][prediction["label"]] += 1

        if prediction["label"] == actual:
            correct += 1

    return {
        "correct": correct,
        "total": len(examples),
        "accuracy": round(correct / len(examples), 3) if examples else 0,
        "confusion_matrix": matrix,
    }


def predict_naive_bayes(text, classifier):
    tokens = tokenize(text)
    scores = []

    for label in classifier["labels"]:
        class_model = classifier["classes"][label]
        score = class_model["log_prior"]

        for token in tokens:
            score += class_model["token_log_likelihoods"].get(
                token,
                class_model["unknown_log_likelihood"],
            )

        scores.append({"label": label, "score": score})

    scores.sort(key=lambda item: item["score"], reverse=True)
    return with_probabilities(scores)[0]


def predict_centroid(text, classifier):
    tokens = tokenize(text)
    vector = vectorize_tokens(tokens, classifier["idf"])
    scores = []

    for label in classifier["labels"]:
        centroid = classifier["classes"][label]["centroid"]
        score = sum(value * centroid.get(token, 0) for token, value in vector.items())
        scores.append({"label": label, "score": score})

    scores.sort(key=lambda item: item["score"], reverse=True)
    return with_probabilities(scores, temperature=0.18)[0]


def predict_hybrid(text, naive_bayes_classifier, centroid_classifier):
    naive_bayes_scores = with_probabilities(score_naive_bayes(text, naive_bayes_classifier))
    centroid_scores = with_probabilities(score_centroid(text, centroid_classifier), temperature=0.18)
    by_label = {}

    for item in naive_bayes_scores:
        by_label.setdefault(item["label"], 0)
        by_label[item["label"]] += item["probability"] * NAIVE_BAYES_WEIGHT

    for item in centroid_scores:
        by_label.setdefault(item["label"], 0)
        by_label[item["label"]] += item["probability"] * CENTROID_WEIGHT

    scores = [
        {"label": label, "score": probability}
        for label, probability in by_label.items()
    ]
    scores.sort(key=lambda item: item["score"], reverse=True)
    return {"label": scores[0]["label"], "probability": scores[0]["score"]}


def score_naive_bayes(text, classifier):
    tokens = tokenize(text)
    scores = []

    for label in classifier["labels"]:
        class_model = classifier["classes"][label]
        score = class_model["log_prior"]

        for token in tokens:
            score += class_model["token_log_likelihoods"].get(
                token,
                class_model["unknown_log_likelihood"],
            )

        scores.append({"label": label, "score": score})

    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores


def score_centroid(text, classifier):
    tokens = tokenize(text)
    vector = vectorize_tokens(tokens, classifier["idf"])
    scores = []

    for label in classifier["labels"]:
        centroid = classifier["classes"][label]["centroid"]
        score = sum(value * centroid.get(token, 0) for token, value in vector.items())
        scores.append({"label": label, "score": score})

    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores


def with_probabilities(scores, temperature=1):
    max_score = max(item["score"] for item in scores)
    denominator = sum(math.exp((item["score"] - max_score) / temperature) for item in scores)

    return [
        {
            "label": item["label"],
            "score": item["score"],
            "probability": math.exp((item["score"] - max_score) / temperature) / denominator,
        }
        for item in scores
    ]


def top_tokens(classifier, limit):
    result = {}

    for label in classifier["labels"]:
        token_scores = classifier["classes"][label]["token_log_likelihoods"]
        result[label] = [
            token
            for token, _score in sorted(
                token_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:limit]
        ]

    return result


def print_summary(model):
    print(f"Trained {model['model_type']} router model.")
    print(
        f"Examples: {model['training_examples']} "
        f"({model['base_training_examples']} base + {model['augmented_training_examples']} augmented)"
    )
    print(f"Separate eval examples: {model['eval_examples']}")
    print(f"Vocabulary: {len(model['vocabulary'])}")
    print("Internal holdout: stratified separately for each target")

    for target in TARGETS:
        train_metric = model["metrics"]["train"][target]["hybrid"]
        holdout_metric = model["metrics"]["holdout"][target]["hybrid"]
        eval_metric = model["metrics"]["eval"][target]["hybrid"]
        print(
            f"{target}: "
            f"hybrid train={train_metric['correct']}/{train_metric['total']} accuracy={train_metric['accuracy']} "
            f"hybrid holdout={holdout_metric['correct']}/{holdout_metric['total']} accuracy={holdout_metric['accuracy']} "
            f"eval={eval_metric['correct']}/{eval_metric['total']} accuracy={eval_metric['accuracy']}"
        )

    print(f"Wrote {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
