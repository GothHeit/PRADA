import json

import pytest

from prada import cli as prada_cli
from tutor import eval as tutor_eval
from tutor.languages import LANGUAGES


def oracle(lang):
    """A chat stub that answers every eval sentence exactly as labeled."""
    labels = {c.text: c for c in tutor_eval.load_cases(lang)}

    def chat(messages, schema, model=None):
        case = labels[messages[-1]["content"]]
        question = LANGUAGES[lang].sample_question
        return json.dumps(
            {
                "reasoning": "oracle",
                "is_correct": case.is_correct,
                "error_type": case.error_type,
                "original": case.text,
                "correction": case.correction,
                "explanation_en": None if case.is_correct else "because",
                "reply_target": question,
                "spoken": f"Ok. {question}",
            },
            ensure_ascii=False,
        )

    return chat


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_evalset_is_well_formed(lang):
    cases = tutor_eval.load_cases(lang)
    assert len(cases) >= 10
    assert any(c.is_correct for c in cases) and any(not c.is_correct for c in cases)
    allowed = LANGUAGES[lang].error_types
    assert all(c.error_type in allowed for c in cases if not c.is_correct)


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_oracle_scores_perfectly(lang):
    s = tutor_eval.score_language(lang, chat=oracle(lang))
    assert s.failed == 0
    assert s.verdict_ok == s.cases and s.format_ok == s.cases
    assert s.category_ok == s.correction_ok == s.wrong_cases > 0


def test_fake_client_scores_below_perfect():
    from tutor import fake_client

    s = tutor_eval.score_language("fr", chat=fake_client.chat)
    assert 0 < s.verdict_ok < s.cases and s.category_ok == 0


def test_unparseable_reply_counts_as_failed():
    s = tutor_eval.score_language("ru", chat=lambda messages, schema: "not json")
    assert s.failed == s.cases and s.verdict_ok == 0


def test_normalize_ignores_punctuation_and_case():
    assert tutor_eval.normalize("Я читаю  книгу!") == tutor_eval.normalize("я читаю книгу.")


def test_cli_eval_smoke(monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    assert prada_cli.main(["eval", "--lang", "ja"]) == 0
    out = capsys.readouterr().out
    assert "[ja]" in out and "error category" in out
