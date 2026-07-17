from tutor.schema import TutorTurn
from tutor.prompts import build_system_prompt
import tutor.ollama_client

def run_turn(user_text: str, language: str, history: list[dict] | None = None) -> TutorTurn:
    system = build_system_prompt(language=language)
    if history is None:
        history = []
    messages = [
    {"role": "system", "content": system}] + history + [
    {"role": "user", "content": user_text},
    ]
    raw = tutor.ollama_client.chat(messages=messages, schema=TutorTurn.model_json_schema())

    return TutorTurn.model_validate_json(raw)