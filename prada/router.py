"""The router: one structured-output call picks which module handles an utterance.

"/name text" skips the model and goes straight to module `name`.
"""

import json

from pydantic import BaseModel, Field, ValidationError

from prada.module import Chat, Module

_PROMPT = """You are the router of PRADA, a personal assistant made of modules. Read the user's message and pick the ONE module that should handle it. Fill the JSON object matching the provided schema: first reasoning (one short sentence), then module.

Modules:
{modules}

If no module clearly fits, pick {default}.{current}
"""


class Route(BaseModel):
    reasoning: str = Field(..., description="One short sentence: what the user wants")
    module: str = Field(..., description="Name of the module that handles the message")


def route_schema(names: list[str]) -> dict:
    schema = Route.model_json_schema()
    schema["properties"]["module"]["enum"] = names
    return schema


def build_prompt(modules: list[Module], current: str | None = None) -> str:
    listing = "\n".join(f"- {m.name}: {m.description}" for m in modules)
    note = f"\nThe conversation so far is with {current}." if current else ""
    return _PROMPT.format(modules=listing, default=modules[0].name, current=note)


def explicit(text: str, names) -> tuple[str, str] | None:
    """(module, rest) for "/module rest" when `module` exists."""
    if text.startswith("/"):
        name, _, rest = text[1:].partition(" ")
        if name in names:
            return name, rest.strip()
    return None


def route(text: str, modules: list[Module], chat: Chat, current: str | None = None) -> str:
    """Name of the module for `text`; the first module when the model's answer is unusable."""
    names = [m.name for m in modules]
    messages = [
        {"role": "system", "content": build_prompt(modules, current)},
        {"role": "user", "content": text},
    ]
    try:
        name = Route.model_validate_json(chat(messages=messages, schema=route_schema(names))).module
    except (ValidationError, ValueError):
        return names[0]
    return name if name in names else names[0]


def fake_chat(messages, schema, model=None) -> str:
    """Offline router: the first module named in the message, else the first module."""
    names = schema["properties"]["module"]["enum"]
    text = messages[-1]["content"].lower()
    name = next((n for n in names if n in text), names[0])
    return json.dumps({"reasoning": "fake router: keyword match", "module": name})
