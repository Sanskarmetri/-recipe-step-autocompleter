from app.cli import autocomplete


def test_cli_suggestion_selection(monkeypatch, capsys):

    inputs = iter([
        "heat oil",
        "1",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs)
    )

    autocomplete()

    output = capsys.readouterr().out

    assert "Suggested next words:" in output
    assert "Updated instruction:" in output
    assert "heat oil" in output
    assert "Goodbye!" in output


def test_cli_clear_command(monkeypatch, capsys):

    inputs = iter([
        "heat oil",
        "clear",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs)
    )

    autocomplete()

    output = capsys.readouterr().out

    assert "Instruction cleared!" in output
    assert "Goodbye!" in output


def test_cli_invalid_selection(monkeypatch, capsys):

    inputs = iter([
        "heat oil",
        "99",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda _: next(inputs)
    )

    autocomplete()

    output = capsys.readouterr().out

    assert "Invalid selection" in output
