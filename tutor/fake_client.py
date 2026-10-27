"""Offline stand-in for the Ollama client: lets tests and smoke runs work without a model."""

import json
import re

from tutor.languages import LANGUAGES


def chat(messages, schema, model=None) -> str:
    system = messages[0]["content"]
    name = re.search(r"You are a (\w+) language tutor", system).group(1)
    language = next(lang for lang in LANGUAGES.values() if lang.name == name)
    student = messages[-1]["content"]
    question = language.sample_question
    return json.dumps(
        {
            "reasoning": "fake client: no analysis performed",
            "is_correct": True,
            "error_type": None,
            "original": student,
            "correction": None,
            "explanation_en": None,
            "reply_target": question,
            "spoken": f"Good. {question}",
        },
        ensure_ascii=False,
    )
