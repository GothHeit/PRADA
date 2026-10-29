import json
import sys
import types

import pytest

from prada import config, llm
from tutor import eval as tutor_eval
from tutor.languages import LANGUAGES


def write_config(tmp_path, text):
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")


def test_default_brain_is_qwen25():
    assert config.llm_settings() == config.LlmSettings("qwen2.5:14b", None)


def test_llm_table_and_env_override(monkeypatch, tmp_path):
    write_config(tmp_path, '[llm]\nmodel = "qwen3.5:9b"\nthink = false\n')
    assert config.llm_settings() == config.LlmSettings("qwen3.5:9b", False)
    monkeypatch.setenv("PRADA_LLM_MODEL", "gemma4:12b")
    monkeypatch.setenv("PRADA_LLM_THINK", "on")
    assert config.llm_settings() == config.LlmSettings("gemma4:12b", True)


@pytest.mark.parametrize(
    "text, error",
    [
        ('[llm]\nmodle = "x"\n', "unknown"),
        ('[llm]\nthink = "no"\n', "true or false"),
        ('[llm]\nmodel = ""\n', "model name"),
        ('llm = "x"\n', "must be a table"),
    ],
)
def test_bad_llm_table(tmp_path, text, error):
    write_config(tmp_path, text)
    with pytest.raises(ValueError, match=error):
        config.llm_settings()


def test_bad_think_env(monkeypatch):
    monkeypatch.setenv("PRADA_LLM_THINK", "maybe")
    with pytest.raises(ValueError, match="PRADA_LLM_THINK"):
        config.llm_settings()


def recording_chat(monkeypatch):
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    calls = []

    def chat(messages, schema, **options):
        calls.append(options)
        return "{}"

    monkeypatch.setattr(llm, "chat", chat)
    return calls


def test_client_uses_the_configured_model(monkeypatch, tmp_path):
    calls = recording_chat(monkeypatch)
    write_config(tmp_path, '[llm]\nmodel = "qwen3.5:9b"\nthink = false\n')
    llm.client(None)([], {})
    llm.client(None, model="gemma4:12b", think=True)([], {})
    assert [(c["model"], c["think"]) for c in calls] == [
        ("qwen3.5:9b", False),
        ("gemma4:12b", True),
    ]


@pytest.fixture
def fake_ollama(monkeypatch):
    sent = []

    def chat(**args):
        sent.append(args)
        return {"message": {"content": "{}"}}

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=chat))
    return sent


def test_think_is_only_sent_when_set(fake_ollama):
    llm.chat([], {}, model="qwen2.5:14b")
    llm.chat([], {}, model="qwen3.5:9b", think=False)
    assert "think" not in fake_ollama[0]
    assert fake_ollama[1]["think"] is False and fake_ollama[1]["model"] == "qwen3.5:9b"


def test_eval_model_flag_reaches_every_call(monkeypatch, capsys):
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    calls = []
    question = LANGUAGES["ru"].sample_question

    def chat(messages, schema, **options):
        calls.append(options)
        return json.dumps(
            {
                "reasoning": "",
                "is_correct": True,
                "error_type": None,
                "original": messages[-1]["content"],
                "correction": None,
                "explanation_en": None,
                "reply_target": question,
                "spoken": question,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm, "chat", chat)
    assert tutor_eval.main(["--lang", "ru", "--model", "qwen3.5:9b", "--think", "off"]) == 0
    assert {(c["model"], c["think"]) for c in calls} == {("qwen3.5:9b", False)}
    out = capsys.readouterr().out
    assert out.startswith("model: qwen3.5:9b, think off") and "seconds per sentence" in out


def test_blank_model_env_is_blamed(monkeypatch):
    monkeypatch.setenv("PRADA_LLM_MODEL", "  ")
    with pytest.raises(ValueError, match="PRADA_LLM_MODEL"):
        config.llm_settings()
