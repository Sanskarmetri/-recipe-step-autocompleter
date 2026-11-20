"""Core functionality for Recipe Step Autocompleter."""
from collections import Counter
import re
from pathlib import Path


# --- Milestone 1: Compile a corpus of cooking recipe instructions.

def milestone_1(data_path=None):
    """
    Load recipe instructions and preprocess them.

    Returns:
        A list of tokenized recipe instructions.
    """

    if data_path is None:
        data_path = (
            Path(__file__).resolve().parent.parent
            / "data"
            / "recipes.txt"
        )

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Recipe dataset not found: {data_path}"
        )

    instructions = data_path.read_text(
        encoding="utf-8"
    ).splitlines()

    corpus = []

    for instruction in instructions:

        instruction = instruction.strip().lower()

        if not instruction:
            continue

        tokens = re.findall(
            r"[a-z]+(?:'[a-z]+)?|\d+",
            instruction
        )

        if tokens:
            corpus.append(tokens)

    return corpus


# --- Milestone 2
def milestone_2(corpus=None, alpha=1.0):
    """
    Build unigram, bigram and trigram models.

    Uses Maximum Likelihood Estimation and Laplace smoothing.
    """

    if corpus is None:
        corpus = milestone_1()

    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()

    for sentence in corpus:

        unigram_counts.update(sentence)

        bigram_counts.update(
            zip(sentence, sentence[1:])
        )

        trigram_counts.update(
            zip(sentence, sentence[1:], sentence[2:])
        )

    vocabulary = set(unigram_counts)
    vocab_size = len(vocabulary)

    # Calculate bigram probabilities.
    bigram_probs = {}

    for (word1, word2), count in bigram_counts.items():

        probability = count / unigram_counts[word1]

        smoothed_probability = (
            (count + alpha) /
            (unigram_counts[word1] + alpha * vocab_size)
        )

        bigram_probs[(word1, word2)] = {
            "mle": probability,
            "smoothed": smoothed_probability
        }

    # Calculate trigram probabilities.
    trigram_probs = {}

    for (word1, word2, word3), count in trigram_counts.items():

        context_count = bigram_counts[(word1, word2)]

        probability = count / context_count

        smoothed_probability = (
            (count + alpha) /
            (context_count + alpha * vocab_size)
        )

        trigram_probs[(word1, word2, word3)] = {
            "mle": probability,
            "smoothed": smoothed_probability
        }

    return {
    "unigram_counts": unigram_counts,
    "bigram_counts": bigram_counts,
    "trigram_counts": trigram_counts,
    "vocabulary": vocabulary,
    "bigram_probs": bigram_probs,
    "trigram_probs": trigram_probs,
    "alpha": alpha
}
def calculate_probability(model, context, next_word):

    if len(context) == 2:
        key = tuple(context + [next_word])
        denominator = model["bigram_counts"].get(tuple(context), 0)
        numerator = model["trigram_counts"].get(key, 0)

    elif len(context) == 1:
        key = tuple(context + [next_word])
        denominator = model["unigram_counts"].get(context[0], 0)
        numerator = model["bigram_counts"].get(key, 0)

    else:
        return 0

    if denominator == 0:
        return 0

    return numerator / denominator

# --- Milestone 3
def milestone_3(model, context, next_word):

    """

    Calculate the probability of the next word

    given the previous context.

    """

    return calculate_probability(

        model,

        context,

        next_word

    )


# --- Milestone 4
# --- Milestone 4: Next-word prediction

def predict_next_word(model, context, top_k=3):
    """
    Predict the next word using trigram, bigram
    and unigram backoff.

    Uses Laplace smoothing.
    """

    if isinstance(context, str):
        context = context.lower().split()

    if not context:
        context = []

    context = context[-2:]

    vocabulary = model["vocabulary"]
    vocab_size = len(vocabulary)

    if vocab_size == 0:
        return []

    alpha = model.get("alpha", 1.0)

    predictions = []

    # 1. Trigram model
    if len(context) == 2:

        trigram_total = sum(
            count
            for (first, second, third), count
            in model["trigram_counts"].items()
            if first == context[0] and second == context[1]
        )

        if trigram_total > 0:

            denominator = trigram_total + alpha * vocab_size

            for word in vocabulary:

                count = model["trigram_counts"].get(
                    tuple(context + [word]), 0
                )

                probability = (
                    count + alpha
                ) / denominator

                predictions.append((word, probability))

    # 2. Bigram model (backoff)
    if not predictions and context:

        previous_word = context[-1]

        bigram_total = sum(
            count
            for (first, second), count
            in model["bigram_counts"].items()
            if first == previous_word
        )

        if bigram_total > 0:

            denominator = bigram_total + alpha * vocab_size

            for word in vocabulary:

                count = model["bigram_counts"].get(
                    (previous_word, word), 0
                )

                probability = (
                    count + alpha
                ) / denominator

                predictions.append((word, probability))

    # 3. Unigram model (backoff)
    if not predictions:

        total_words = sum(model["unigram_counts"].values())

        denominator = total_words + alpha * vocab_size

        for word in vocabulary:

            count = model["unigram_counts"].get(word, 0)

            probability = (
                count + alpha
            ) / denominator

            predictions.append((word, probability))

    predictions.sort(
        key=lambda item: (-item[1], item[0])
    )

    return predictions[:top_k]