import pytest
from app.core import predict_next_word
from app.core import predict_next_word
from app.core import milestone_1
from app.core import milestone_2
from app.core import milestone_3
from app.core import calculate_probability

def test_milestone_1():
    corpus = milestone_1()
    assert len(corpus) > 0
    assert all(len(instruction) > 0 for instruction in corpus)
    assert all(
        token == token.lower()
        for instruction in corpus
        for token in instruction
    )
    assert ["heat", "oil", "in", "a", "pan"] in corpus
    assert "pan." not in [
        token
        for instruction in corpus
        for token in instruction
    ]

def test_milestone_2():
    corpus = [
        ["heat", "oil", "in", "pan"],
        ["heat", "oil", "in", "pot"],
        ["add", "salt", "to", "pan"]
    ]
    model = milestone_2(corpus)
    assert model["unigram_counts"]["heat"] == 2
    assert model["bigram_counts"][("heat", "oil")] == 2
    assert model["trigram_counts"][("heat", "oil", "in")] == 2
    assert model["bigram_probs"][("heat", "oil")]["mle"] == 1.0
    assert model["bigram_probs"][("heat", "oil")]["smoothed"] < 1.0

def test_milestone_3():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["heat", "oil", "in", "pot"],
        ["add", "salt", "to", "pan"]
    ]

    model = milestone_2(corpus)

    probability = milestone_3(
        model,
        ["heat", "oil"],
        "in"
    )

    assert probability == 1.0
def test_milestone_4():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["heat", "oil", "in", "pot"],
        ["add", "salt", "to", "pan"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["heat", "oil"]
    )

    assert len(predictions) > 0

    assert predictions[0][0] == "in"

    assert abs(predictions[0][1] - 0.3) < 1e-9   
def test_milestone_6():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["heat", "oil", "in", "pot"],
        ["add", "salt", "to", "pan"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["heat", "oil"],
        top_k=3
    )

    assert len(predictions) == 3

    assert all(
        probability > 0
        for word, probability in predictions
    )

    assert predictions[0][0] == "in"    
def test_unigram_backoff():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["add", "salt", "to", "pan"],
        ["mix", "flour", "with", "water"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["unknown", "unknown"]
    )

    assert len(predictions) == 3

    assert all(
        probability > 0
        for word, probability in predictions
    )


def test_bigram_backoff():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["add", "oil", "to", "pan"],
        ["mix", "oil", "with", "water"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["unknown", "oil"]
    )

    assert len(predictions) == 3

    assert predictions[0][0] in [
        "in", "to", "with"
    ]
def test_bigram_probability_normalization():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["add", "oil", "to", "pan"],
        ["mix", "oil", "with", "water"],
        ["oil"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["unknown", "oil"],
        top_k=len(model["vocabulary"])
    )

    total_probability = sum(
        probability
        for word, probability in predictions
    )

    assert abs(total_probability - 1.0) < 1e-9        
def test_trigram_probability_normalization():

    corpus = [
        ["heat", "oil", "in", "pan"],
        ["heat", "oil"]
    ]

    model = milestone_2(corpus)

    predictions = predict_next_word(
        model,
        ["heat", "oil"],
        top_k=len(model["vocabulary"])
    )

    total_probability = sum(
        probability
        for word, probability in predictions
    )

    assert abs(total_probability - 1.0) < 1e-9    