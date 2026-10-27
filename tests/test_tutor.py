import json

import pytest

from prada import cli as prada_cli
from tutor import fake_client
from tutor.languages import LANGUAGES, get_language
from tutor.orchestrator import run_turn
from tutor.prompts import build_system_prompt
from tutor.schema import ErrorType, turn_schema


def test_every_language_error_type_exists_in_enum():
    values = {e.value for e in ErrorType}
    for lang in LANGUAGES.values():
        assert set(lang.error_types) <= values


@pytest.mark.parametrize("code", LANGUAGES)
def test_schema_limits_error_types(code):
    lang = get_language(code)
    assert turn_schema(lang)["$defs"]["ErrorType"]["enum"] == list(lang.error_types)


def test_unknown_language():
    with pytest.raises(ValueError, match="unknown language"):
        get_language("xx")


@pytest.mark.parametrize("code", LANGUAGES)
def test_prompt_is_language_specific(code):
    lang = get_language(code)
    prompt = build_system_prompt(lang)
    assert lang.name in prompt and lang.script in prompt
    assert "{" not in prompt
    others = [x for x in LANGUAGES.values() if x is not lang]
    assert not any(o.name in prompt for o in others)


@pytest.mark.parametrize("code", LANGUAGES)
def test_run_turn_with_fake_client(code):
    turn = run_turn("hello", code, chat=fake_client.chat)
    assert turn.is_correct
    assert turn.spoken.endswith(get_language(code).sample_question)


def test_run_turn_sends_history_and_schema():
    seen = {}

    def chat(messages, schema):
        seen["messages"], seen["schema"] = messages, schema
        return fake_client.chat(messages, schema)

    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    run_turn("c", "fr", history, chat=chat)
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert "French" in seen["messages"][0]["content"]
    assert "preposition" in seen["schema"]["$defs"]["ErrorType"]["enum"]


def test_invalid_model_reply_is_rejected():
    bad = json.dumps({"reasoning": "x"})
    with pytest.raises(ValueError):
        run_turn("hi", "ru", chat=lambda messages, schema: bad)


def test_cli_tutor_smoke(monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    inputs = iter(["Я читаю книгу.", ""])

    def fake_input(_):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    assert prada_cli.main(["tutor", "--lang", "ja"]) == 0
    assert "何を読むのが好きですか？" in capsys.readouterr().out


def test_cli_wake_reports_missing_mac(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("PRADA_WAKE_MAC", raising=False)
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "none.toml"))
    assert prada_cli.main(["wake"]) == 1
    assert "no MAC" in capsys.readouterr().err
