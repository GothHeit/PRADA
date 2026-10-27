from enum import StrEnum

from pydantic import BaseModel, Field

from tutor.languages import Language


class ErrorType(StrEnum):
    # Shared
    AGREEMENT = "agreement"
    WORD_ORDER = "word_order"
    LEXICAL = "lexical"
    SPELLING = "spelling"
    CONJUGATION = "conjugation"
    OTHER = "other"
    # Russian
    CASE = "case"
    ASPECT = "aspect"
    MOTION_VERB = "motion_verb"
    # French
    ARTICLE = "article"
    TENSE_MOOD = "tense_mood"
    PREPOSITION = "preposition"
    PRONOUN = "pronoun"
    # Japanese
    PARTICLE = "particle"
    POLITENESS = "politeness"
    COUNTER = "counter"
    SCRIPT = "script"


class TutorTurn(BaseModel):
    reasoning: str = Field(
        ...,
        description="Step by step, analyze the sentence's grammar using the checklist from the system prompt. Decide the error category from this analysis. This is your scratchpad — think before committing.",
    )
    is_correct: bool = Field(..., description="Whether the phrase is correct")
    error_type: ErrorType | None = Field(
        ..., description="The error category | Null if the sentence is correct"
    )
    original: str = Field(..., description="Original phrase")
    correction: str | None = Field(
        ..., description="Original phrase corrected | Null if the sentence is correct"
    )
    explanation_en: str | None = Field(
        ..., description="Grammar explanation in English only | Null if the sentence is correct"
    )
    reply_target: str = Field(
        ...,
        description="A short, simple, beginner-level follow-up question in the target language, in its own script, that invites the student to keep the conversation going. Always present, always in the target language, never English, never empty.",
    )
    spoken: str = Field(
        ...,
        description="The single natural message the student actually hears — the ONLY field shown or spoken; every other field is silent and internal. Weave the fields above into ONE smooth, natural tutor turn. If is_correct is false: give the correction and its explanation in English (reuse correction and explanation_en, do not re-derive). If is_correct is true: briefly affirm in the target language. In BOTH cases, the LAST sentence of spoken MUST be the reply_target text copied VERBATIM — the exact same characters, do NOT translate it or rephrase it into English. Never mention field names.",
    )


def turn_schema(language: Language) -> dict:
    """JSON schema for TutorTurn with error_type limited to the language's categories."""
    schema = TutorTurn.model_json_schema()
    schema["$defs"]["ErrorType"]["enum"] = list(language.error_types)
    return schema
