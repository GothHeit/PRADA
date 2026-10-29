"""PRADA's persona: one voice shared by every module that talks to the user.

Who she is (name, character, tone) comes from a local TOML file, $PRADA_PERSONA or
persona.toml next to the config file, so a personal persona never has to live in the
repo; `prada persona init` writes one to edit. Without it she uses a short neutral
default. What she may claim (`GUARDRAIL`) stays here, so no persona file can drop it.

    name = "PRADA"
    character = '''You are {name}, ...'''
    tone = "warm and encouraging"
"""

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from prada import config

DEFAULT_NAME = "PRADA"
DEFAULT_CHARACTER = (
    "You are {name}, a private assistant who runs on the user's own computer, in their home. "
    "You are friendly, clear and helpful, and you remember what they told you earlier "
    "in the conversation."
)
DEFAULT_TONE = "friendly and encouraging"

# Always after the character: the one rule a persona file can't change.
GUARDRAIL = (
    "You can only do what you are actually given tools for: "
    "never claim to have done something you did not do."
)


@dataclass(frozen=True)
class Persona:
    name: str = DEFAULT_NAME
    character: str = DEFAULT_CHARACTER  # may use {name}
    tone: str = DEFAULT_TONE


def persona_path() -> Path:
    return Path(os.environ.get("PRADA_PERSONA") or config.config_path().parent / "persona.toml")


def load(path: Path | None = None) -> Persona:
    """The persona file at `path` (default `persona_path()`), or the default if there is none.

    Raises ValueError on a file that can't be read as a persona."""
    path = path or persona_path()
    if not path.is_file():
        return Persona()
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"can't read {path}: {e}") from e
    unknown = set(data) - {"name", "character", "tone"}
    if unknown:
        raise ValueError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    for key, value in data.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} in {path} must be a non-empty string")
    return Persona(**{k: v.strip() for k, v in data.items()})


def render(persona: Persona) -> str:
    """The persona prompt: her character, then the guardrail."""
    return f"{persona.character.replace('{name}', persona.name)} {GUARDRAIL}"


def template(persona: Persona) -> str:
    """A persona file holding `persona`, for `prada persona init`."""
    character = (
        f"'''\n{persona.character}'''"  # literal string: no escapes needed
        if "'''" not in persona.character
        else _quote(persona.character)
    )
    return (
        "# PRADA's persona. Restart the node after editing.\n"
        "# {name} in character is replaced by name. The rule that she never claims\n"
        "# actions she didn't do is added by PRADA itself, so it isn't needed here.\n\n"
        f"name = {_quote(persona.name)}\n\n"
        f"character = {character}\n\n"
        f"tone = {_quote(persona.tone)}\n"
    )


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _load_or_default() -> Persona:
    try:
        return load()
    except ValueError as e:
        print(f"prada: {e}; using the default persona", file=sys.stderr)
        return Persona()


# Read once per process: restart the node to pick up an edited persona file.
CURRENT = _load_or_default()
NAME = CURRENT.name
# Who PRADA is. Modules put this at the top of their system prompt.
CHARACTER = render(CURRENT)
# A one-line version of the voice, for prompts whose main job is something else.
TONE = CURRENT.tone


def system_prompt(task: str) -> str:
    """The persona followed by a module's own instructions."""
    return f"{CHARACTER}\n\n{task}"


def waking() -> str:
    return "Okay, I sent the wake signal to your PC. Give it a moment to stretch."


def declined() -> str:
    return "Okay, I won't. Nothing happened."


def not_a_yes() -> str:
    return "I didn't take that as a yes, so I didn't do it. Ask me again if you want it."


def timed_out() -> str:
    return "That question timed out, so I didn't do it. Ask me again if you want it."


def failed(what: str, reason: str) -> str:
    """What PRADA says when an action did not happen: never a claim that it did."""
    return f"I tried to {what}, but it didn't work: {reason.rstrip('. ')}."


def cant_wake() -> str:
    """Asked to wake the PC before its MAC address is in the config: nothing was sent."""
    return (
        "I can't wake your PC yet, I don't know its network address. "
        "Add its MAC under [wake] in my config and I'll be able to."
    )


def noted(text: str) -> str:
    return f"Got it, I wrote that down: “{text}”."


def not_noted() -> str:
    """When a question reaches the notes: she never writes one down."""
    return 'That sounds like a question, so I didn\'t write it down. Say "note that..." if you want a note.'


def offer_search() -> str:
    """When a question would need the internet but she wasn't asked to search."""
    return "I'd have to look that up online. Want me to search for it?"


def want_search() -> str:
    """Added when she said she isn't sure of something but forgot to ask."""
    return "Want me to look it up?"


def search_what() -> str:
    return "Sure, what should I look up?"


def looking_up() -> str:
    """Said out loud as a search starts, so the wait isn't silent; never that it's done."""
    return "Let me look that up."


def no_search() -> str:
    return "Okay, I won't look it up."


def cant_do() -> str:
    """When her answer kept saying she did something she has no way to do: nothing happened."""
    return (
        "I can't do that from here yet, so nothing happened. "
        'If it helps, I can keep a note of it: just say "note that ...".'
    )


def lost_words() -> str:
    """When the model's answer came out broken twice: nothing happened, she asks again."""
    return "Sorry, I lost my words for a second. Could you say that again?"
