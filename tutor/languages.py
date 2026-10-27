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
    # How to pick error_type when categories overlap, listed in the prompt ("" = none).
    categories: str = ""


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
        categories="""- agreement: gender/number mismatch between a noun and its adjective, participle or subject (elle est content).
- article: wrong or missing article, including the wrong gender of the article itself (un orange -> une orange).
- conjugation: a verb form that does not exist or has the wrong person/ending, with the right tense (nous mangeons -> mangent for ils).
- tense_mood: the right verb but the wrong tense or mood for the context: past-time words with a present verb (hier ... mangeons), si + conditional/future, missing subjunctive, wrong auxiliary avoir/être.
- preposition: wrong or missing preposition, or a missing contraction (à le -> au, de les -> des).
- pronoun: wrong pronoun choice (direct vs indirect: lui/le/leur/les, y, en) or wrong placement of a pronoun.
- word_order: words in the wrong position (adjective after/before the noun, adverb, negation).
- spelling: only a missing or wrong accent or misspelled word whose grammar is right (a -> à, ete -> été).
- lexical: a real word with the wrong meaning.
- other: only when nothing above fits.""",
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
        categories="""- particle: a wrong, missing or extra particle (を, が, に, で, は, へ, と).
- conjugation: the verb, adjective or copula is in the wrong form or tense: plain form where です/ます is needed or the other way round, past tense missing, い-adjective + でした (高いでした -> 高かったです).
- politeness: the grammar is fine but the register is wrong for the listener, e.g. an ordinary verb about a superior's action where honorific/humble language is needed (食べますか -> 召し上がりますか).
- counter: wrong counter word for the thing being counted (人 for animals -> 匹).
- word_order: only when the order really breaks Japanese (the verb is not last, a modifier is far from its noun). Japanese word order is otherwise free, so a reordered but natural sentence is correct.
- lexical: a real word with the wrong meaning or collocation (水を食べる -> 水を飲む).
- script: the wrong kana/kanji or a spelling mistake in the writing. Writing a word entirely in hiragana is not a mistake.
- other: only when nothing above fits.""",
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


_ACCENTS = re.compile(r"[àâçéèêëîïôùûüœ]")
_WORD = re.compile(r"[a-zà-ÿœ]+")
# Function words that settle French vs English; words shared by both (a, on, as, me) are left out.
_FRENCH = frozenset(
    "je tu il elle nous vous ils elles le la les un une des du au aux et est suis es sommes "
    "sont ai avons ont pas ne que qui dans pour avec sur mon ma mes ton ta ses ce cette ces "
    "mais où très bien oui non merci bonjour salut j l qu c n été être avoir fait vais va "
    "allé allée aime mange lis suis ça quel quelle comment pourquoi".split()
)
_ENGLISH = frozenset(
    "the is are am was were be what how why where when who i you he she it we they my your "
    "his her our their of to in and not do does did can will would this that with for from "
    "have has had weather please hello hi yes no thanks".split()
)


def latin_language(text: str) -> str | None:
    """ "fr" when Latin-script `text` reads as French; None for English or anything unclear."""
    lowered = text.lower()
    words = _WORD.findall(lowered)
    french = sum(w in _FRENCH for w in words) + bool(_ACCENTS.search(lowered))
    english = sum(w in _ENGLISH for w in words)
    return "fr" if french > english else None


def detect_language(text: str) -> str | None:
    """The tutor language `text` is written in, when it can tell: script first, then French."""
    return script_language(text) or latin_language(text)
