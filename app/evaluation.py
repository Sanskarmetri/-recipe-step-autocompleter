"""Evaluation utilities for Recipe Step Autocompleter."""

import math
import re
from collections import Counter
from pathlib import Path

from app.core import milestone_2, predict_next_word


def load_recipe_groups(data_path):
    """Load recipes while preserving recipe and instruction boundaries."""

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    text = data_path.read_text(encoding="utf-8")

    groups = []
    current_group = []

    for line in text.splitlines():

        line = line.strip().lower()

        if not line:
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", line)

        if tokens:
            current_group.append(tokens)

    if current_group:
        groups.append(current_group)

    return groups


def get_probability(model, context, target):
    """Calculate smoothed probability using trigram/bigram/unigram backoff."""

    vocabulary = model["vocabulary"]
    vocab_size = len(vocabulary)
    alpha = model.get("alpha", 1.0)

    if target not in vocabulary or vocab_size == 0:
        return 0.0

    if len(context) >= 2:

        first, second = context[-2:]

        total = sum(
            count
            for (w1, w2, _), count
            in model["trigram_counts"].items()
            if w1 == first and w2 == second
        )

        if total > 0:
            count = model["trigram_counts"].get(
                (first, second, target), 0
            )

            return (
                (count + alpha) /
                (total + alpha * vocab_size)
            )

    if context:

        previous = context[-1]

        total = sum(
            count
            for (w1, _), count
            in model["bigram_counts"].items()
            if w1 == previous
        )

        if total > 0:
            count = model["bigram_counts"].get(
                (previous, target), 0
            )

            return (
                (count + alpha) /
                (total + alpha * vocab_size)
            )

    total = sum(model["unigram_counts"].values())

    count = model["unigram_counts"].get(target, 0)

    return (
        (count + alpha) /
        (total + alpha * vocab_size)
    )


def evaluate_model(data_path):
    """Evaluate using leave-one-recipe-out cross-validation."""

    groups = load_recipe_groups(data_path)

    if len(groups) < 2:
        raise ValueError("At least two recipes are required.")

    evaluated_tokens = 0
    skipped_oov_tokens = 0
    total_target_tokens = 0

    correct_top1 = 0
    correct_top3 = 0
    baseline_correct = 0
    bigram_baseline_correct = 0

    negative_log_likelihood = 0.0

    for test_index in range(len(groups)):

        train_groups = [
            group
            for index, group in enumerate(groups)
            if index != test_index
        ]

        test_group = groups[test_index]

        train_corpus = [
            instruction
            for group in train_groups
            for instruction in group
        ]

        model = milestone_2(train_corpus)

        vocabulary = model["vocabulary"]

        baseline = sorted(
            vocabulary,
            key=lambda word: (
                -model["unigram_counts"][word],
                word
            )
        )

        baseline_top3 = set(baseline[:3])

        for instruction in test_group:

            for index in range(1, len(instruction)):

                target = instruction[index]
                context = instruction[max(0, index - 2):index]

                total_target_tokens += 1

                if target not in vocabulary:
                    skipped_oov_tokens += 1
                    continue

                predictions = predict_next_word(
                    model,
                    context,
                    top_k=3
                )

                predicted_words = [
                    word for word, _ in predictions
                ]

                if predicted_words and predicted_words[0] == target:
                    correct_top1 += 1

                if target in predicted_words:
                    correct_top3 += 1

                if baseline and baseline[0] == target:
                    baseline_correct += 1

                previous_word = context[-1]
                candidates = [
                    (word, count)
                    for (previous, word), count
                    in model["bigram_counts"].items()
                    if previous == previous_word
                ]

                candidates.sort(key=lambda item: (-item[1], item[0]))

                bigram_prediction = (
                    candidates[0][0] if candidates
                    else (baseline[0] if baseline else None)
                )

                if bigram_prediction == target:
                    bigram_baseline_correct += 1

                probability = get_probability(
                    model,
                    context,
                    target
                )

                negative_log_likelihood -= math.log(probability)

                evaluated_tokens += 1

    if total_target_tokens == 0:
      raise ValueError(
        "No prediction targets available for evaluation."
    )

    if evaluated_tokens == 0:
      return {
        "evaluated_tokens": 0,
        "skipped_oov_tokens": skipped_oov_tokens,
        "vocabulary_coverage": 0.0,
        "top1_accuracy": 0.0,
        "top3_accuracy": 0.0,
        "baseline_top1_accuracy": 0.0,
        "bigram_baseline_top1_accuracy": 0.0,
        "perplexity": float("inf")
    }

    return {
        "evaluated_tokens": evaluated_tokens,
        "skipped_oov_tokens": skipped_oov_tokens,

        "vocabulary_coverage": (
            100 * evaluated_tokens / total_target_tokens
        ),

        "top1_accuracy": (
            100 * correct_top1 / evaluated_tokens
        ),

        "top3_accuracy": (
            100 * correct_top3 / evaluated_tokens
        ),

        "baseline_top1_accuracy": (
            100 * baseline_correct / evaluated_tokens
        ),

        "bigram_baseline_top1_accuracy": (
            100 * bigram_baseline_correct / evaluated_tokens
        ),

        "perplexity": math.exp(
            negative_log_likelihood / evaluated_tokens
        )
    }


def main():
    data_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "recipes.txt"
    )

    results = evaluate_model(data_path)

    print("\nRECIPE STEP AUTOCOMPLETER EVALUATION")
    print("=" * 42)

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

    print("=" * 42)


if __name__ == "__main__":
    main()