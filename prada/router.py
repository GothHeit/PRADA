"""The router: one structured-output call picks which module handles an utterance.

"/name text" skips the model and goes straight to module `name`.
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from prada.module import Chat, Module

_PROMPT = """You are the router of PRADA, a personal assistant made of modules. Read the user's message and pick the ONE module that should handle it. Fill the JSON object matching the provided schema: first reasoning (one short sentence), then module.

Modules:
{modules}

If no module clearly fits, pick {default}. Pick a language-practice module only for a sentence the user wrote to practise a language, never for a question or request in English. A question about the user's own life or past (what they said, planned, wanted or told you before, e.g. "which camera was I thinking of buying?") goes to {default}, which remembers earlier conversations: never to a notes or web module. Pick a notes module only when the user explicitly asks to write something down or to read their notes.{current}
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
    note = (
        f"\nThe conversation so far is with {current}; stay with it only if the message plausibly continues it."
        if current
        else ""
    )
    return _PROMPT.format(modules=listing, default=modules[0].name, current=note)


_QUESTION = re.compile(
    r"^\W*(?:who|whom|whose|what|which|when|where|why|how|is|are|was|were|do|does|did|have|has"
    r"|had|can|could|will|would|should|shall|may|might|am)\b",
    re.IGNORECASE,
)
_OWN_PAST = re.compile(
    r"\b(?:(?:was|were|did)\s+i|i\s+(?:was|said|told|mentioned|wanted|thought"
    r"|planned|asked|bought|did)|(?:do|did)\s+you\s+(?:remember|recall)"
    r"|(?:do|did)\s+you\s+know\s+(?:who|what|which|where|when)\s+(?:my|i)"
    r"|(?:tell|told|said\s+to)\s+you|my\s+\w+(?:'s)?\s+name|my\s+favou?rite)\b",
    re.IGNORECASE,
)


def is_question(text: str) -> bool:
    """True for "what's on my list?" or "do you know...": a "?" or a question word first."""
    return text.rstrip().endswith("?") or bool(_QUESTION.match(text.replace("\u2019", "'")))


_TELL_ME = re.compile(r"\btell\s+me\s+(?:who|what|which|where|when)\b", re.IGNORECASE)


def about_own_past(text: str) -> bool:
    """True for a question about the user's own past ("which camera was I thinking of
    buying?", "do you remember my sister's name?", "tell me who my favorite singer is"):
    only her memory can answer it."""
    text = text.replace("\u2019", "'")
    asks = is_question(text) or bool(_TELL_ME.search(text))
    return asks and bool(_OWN_PAST.search(text))


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
