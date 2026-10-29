from prada import llm
from tutor.languages import Language, get_language
from tutor.prompts import build_system_prompt
from tutor.schema import TutorTurn, turn_schema


def default_chat(model: str | None = None, think: bool | None = None):
    """The model client: the tutor's fake one when PRADA_FAKE_LLM is set, Ollama otherwise
    (`model` and `think` override the configured ones)."""
    from tutor import fake_client

    return llm.client(fake_client.chat, model=model, think=think)


def run_turn(
    user_text: str,
    language: Language | str,
    history: list[dict] | None = None,
    chat=None,
) -> TutorTurn:
    if isinstance(language, str):
        language = get_language(language)
    chat = chat or default_chat()
    messages = (
        [{"role": "system", "content": build_system_prompt(language)}]
        + (history or [])
        + [{"role": "user", "content": user_text}]
    )
    raw = chat(messages=messages, schema=turn_schema(language))
    return TutorTurn.model_validate_json(raw)
