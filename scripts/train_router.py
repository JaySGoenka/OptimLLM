#!/usr/bin/env python3
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = PROJECT_ROOT / "data" / "router-training.json"
MODEL_PATH = PROJECT_ROOT / "data" / "router-model.json"
TARGETS = ["task_type", "difficulty", "privacy", "route_class"]
TRAIN_TEST_SEED = 42
TEST_RATIO = 0.2
MIN_TOKEN_COUNT = 1
ALPHA = 0.7
CHAR_NGRAM_RANGE = (4, 5)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "me", "my", "of", "on", "or", "that", "the",
    "this", "to", "with"
}
PRIVACY_TERMS = [
    "private", "confidential", "secret", "password", "api key", "token", "ssn",
    "social security", "medical", "diagnosis", "bank", "credit card", "personal",
    "address", "phone number", "email", "legal", "contract", "customer", "journal",
]
CODING_TERMS = [
    "code", "function", "debug", "bug", "javascript", "typescript", "python",
    "react", "node", "api", "sql", "stack trace", "refactor", "component",
]
COMPLEX_TERMS = [
    "architecture", "multi-step", "deep", "detailed", "optimize", "tradeoff",
    "compare", "reason", "reasoning", "proof", "math", "logic", "design",
    "strategy", "analyze", "implement end to end", "scalable", "distributed",
]


def main():
    training_data = json.loads(TRAINING_PATH.read_text())
    validate_training_data(training_data)

    examples = [
        {**example, "tokens": tokenize(example["text"])}
        for example in training_data["examples"]
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
        holdout_metrics[target] = evaluate_target(target_test_examples, evaluation_classifier, target)
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

    model = {
        "schema_version": 2,
        "model_type": "python_multinomial_naive_bayes",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_examples": len(examples),
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
        "split": {
            "seed": TRAIN_TEST_SEED,
            "test_ratio": TEST_RATIO,
            "by_target": split_summary,
        },
        "vocabulary": full_vocabulary,
        "targets": TARGETS,
        "classifiers": final_classifiers,
        "metrics": {
            "train": evaluate(examples, final_classifiers),
            "holdout": holdout_metrics,
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

    for index, example in enumerate(examples):
        if not isinstance(example.get("text"), str) or not example["text"].strip():
            raise ValueError(f"Example {index} is missing text.")

        for target in TARGETS:
            if example.get(target) not in labels[target]:
                raise ValueError(f"Example {index} has invalid {target}: {example.get(target)}")


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


def evaluate(examples, classifiers):
    metrics = {}

    for target in TARGETS:
        metrics[target] = evaluate_target(examples, classifiers[target], target)

    return metrics


def evaluate_target(examples, classifier, target):
    labels = classifier["labels"]
    matrix = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    correct = 0

    for example in examples:
        prediction = predict(example["text"], classifier)
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


def predict(text, classifier):
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
    return scores[0]


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
    print(f"Examples: {model['training_examples']}")
    print(f"Vocabulary: {len(model['vocabulary'])}")
    print("Train/test split: stratified separately for each target")

    for target in TARGETS:
        train_metric = model["metrics"]["train"][target]
        holdout_metric = model["metrics"]["holdout"][target]
        print(
            f"{target}: "
            f"train={train_metric['correct']}/{train_metric['total']} accuracy={train_metric['accuracy']} "
            f"holdout={holdout_metric['correct']}/{holdout_metric['total']} accuracy={holdout_metric['accuracy']}"
        )

    print(f"Wrote {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
