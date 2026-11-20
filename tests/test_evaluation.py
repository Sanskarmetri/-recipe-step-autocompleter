import math

from app.evaluation import load_recipe_groups, evaluate_model


def test_load_recipe_groups_preserves_boundaries(tmp_path):

    data = tmp_path / "recipes.txt"

    data.write_text(
        "heat oil\n"
        "add salt\n"
        "\n"
        "boil water\n"
        "add rice\n",
        encoding="utf-8"
    )

    groups = load_recipe_groups(data)

    assert len(groups) == 2
    assert len(groups[0]) == 2
    assert len(groups[1]) == 2


def test_evaluation_returns_valid_metrics(tmp_path):

    data = tmp_path / "recipes.txt"

    data.write_text(
    "heat oil\n"
    "add salt\n"
    "\n"
    "boil oil\n"
    "add salt\n",
    encoding="utf-8"
)

    results = evaluate_model(data)

    assert results["evaluated_tokens"] > 0

    assert results["skipped_oov_tokens"] >= 0

    assert 0 <= results["vocabulary_coverage"] <= 100

    assert 0 <= results["top1_accuracy"] <= 100

    assert 0 <= results["top3_accuracy"] <= 100

    assert results["top1_accuracy"] <= results["top3_accuracy"]

    assert math.isfinite(results["perplexity"])

    assert results["perplexity"] >= 1
def test_evaluation_handles_all_oov_tokens(tmp_path):

    data = tmp_path / "recipes.txt"

    data.write_text(
        "heat oil\n"
        "add salt\n"
        "\n"
        "boil water\n"
        "add rice\n",
        encoding="utf-8"
    )

    results = evaluate_model(data)

    assert results["evaluated_tokens"] == 0

    assert results["skipped_oov_tokens"] > 0

    assert results["vocabulary_coverage"] == 0

    assert math.isinf(results["perplexity"])