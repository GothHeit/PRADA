import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str
    # Grammar points the model should check, listed in the prompt.
    checklist: str
    # Allowed error_type values (ErrorType members' values) for this language.
    error_types: tuple[str, ...]
    # Fixed follow-up question used by the fake LLM client.
    sample_question: str


LANGUAGES: dict[str, Language] = {
    "ru": Language(
        code="ru",
        name="Russian",
        script="Cyrillic",
        checklist="case, verb aspect, verbs of motion, gender/number agreement, word order, spelling, conjugation",
        error_types=(
            "case",
            "aspect",
            "motion_verb",
            "agreement",
            "word_order",
            "lexical",
            "spelling",
            "conjugation",
            "other",
        ),
        sample_question="Что ты любишь читать?",
    ),
    "fr": Language(
        code="fr",
        name="French",
        script="Latin",
        checklist="noun gender and adjective agreement, articles, verb conjugation, tense and mood, prepositions, pronouns, word order, spelling and accents",
        error_types=(
            "agreement",
            "article",
            "conjugation",
            "tense_mood",
            "preposition",
            "pronoun",
            "word_order",
            "lexical",
            "spelling",
            "other",
        ),
        sample_question="Qu'est-ce que tu aimes lire ?",
    ),
    "ja": Language(
        code="ja",
        name="Japanese",
        script="Japanese script (kana and kanji)",
        checklist="particles, verb and adjective conjugation, politeness level, counters, word order, kana/kanji usage",
        error_types=(
            "particle",
            "conjugation",
            "politeness",
            "counter",
            "word_order",
            "lexical",
            "script",
            "other",
        ),
        sample_question="何を読むのが好きですか？",
    ),
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code.lower()]
    except KeyError:
        raise ValueError(f"unknown language {code!r}; choose from {', '.join(LANGUAGES)}") from None


_SCRIPTS = (("ja", re.compile(r"[぀-ヿ一-鿿]")), ("ru", re.compile(r"[Ѐ-ӿ]")))


def script_language(text: str) -> str | None:
    """The language whose script `text` uses, when the script alone tells (ja, ru)."""
    return next((code for code, pattern in _SCRIPTS if pattern.search(text)), None)
