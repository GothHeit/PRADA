"""The situation: what is going on around a turn, the way PRADA's own code sees it.

One `Situation` holds the clock, when they last talked, how the words reached her (voice or
typed, which ear, how sure the hearing was), the music players and what they play, a
question or offer she is waiting on, what she did lately (the action log), what isn't
working, and what she can do. The node builds one per turn from the modules' state providers
(`Module.observe`); `render` turns it into the lines both minds read: the status block of the
assistant and the situation of the understand step (prada.mind). The trace keeps it.

Written in the third person ("PRADA asked", "the user"), so the same text fits a mind that
is her and one that only decides for her.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from prada.status import ago, now_line


class Source(BaseModel):
    """How a request reached PRADA; filled by the client that sent it."""

    via: Literal["voice", "typed"] = "typed"
    ear: str = Field("", description="Which microphone or device heard it")
    start: Literal["", "wake word", "follow-up", "interruption"] = Field(
        "", description="Voice: what opened the listening"
    )
    language: str = Field("", description="The language it was heard in, e.g. en")
    sure: float | None = Field(
        None, description="Voice: how sure the ears were of the language, 0 to 1"
    )


SHORT = 80  # longest error text kept in a `broken` line


def short(error) -> str:
    """An error's text, cut to `SHORT` characters, for a `broken` line."""
    text = " ".join(str(error).split()) or type(error).__name__
    return text if len(text) <= SHORT else text[: SHORT - 3].rstrip() + "..."


UNSURE = 0.5  # ears less sure of the language than this: the words may be misheard


@dataclass
class Player:
    app: str  # "Spotify", "Corpus"
    state: str  # "playing", "paused", "stopped"
    title: str = ""
    artist: str = ""


@dataclass
class Option:
    label: str  # "the album: Wave by Antonio Carlos Jobim"
    tracks: int = 0
    app: str = ""


@dataclass
class Ability:
    name: str  # the module
    about: str  # what she can do with it, one line
    examples: list[str] = field(default_factory=list)  # how to ask plainly


@dataclass
class Situation:
    now: float
    # None: no music side to ask (or it wasn't asked); []: none open
    players: list[Player] | None = field(default_factory=list)
    question: str = ""  # what PRADA asked the user to choose, waiting for the answer
    options: list[Option] = field(default_factory=list)  # numbered from 1
    offer: str = ""  # what PRADA offered to look up online, waiting for a yes
    remembered: list[str] = field(default_factory=list)  # what the user told her before
    last_talk: float | None = None  # when they last talked; None = not given
    first_talk: bool = False  # True: they have never talked before
    source: Source | None = None
    confirm: list[str] = field(default_factory=list)  # her questions about held actions
    done: list[str] = field(default_factory=list)  # the action log, lately, her lines
    since: int = 0  # minutes the action log lines go back
    broken: list[str] = field(default_factory=list)  # what isn't working right now
    abilities: list[Ability] = field(default_factory=list)


def lines(s: Situation) -> list[str]:
    out = [now_line(s.now)]
    if s.first_talk:
        out.append("The user and PRADA have never talked before.")
    elif s.last_talk is not None:
        out.append(f"The user and PRADA last talked {ago(s.now - s.last_talk)}.")
    if s.source is not None:
        out.append(_heard(s.source))
    if s.players is not None:
        if s.players:
            out.append("Music players:")
            for p in s.players:
                what = f' "{p.title}"' + (f" by {p.artist}" if p.artist else "") if p.title else ""
                out.append(f"- {p.app}: {p.state}{what}")
        else:
            out.append("Music players: none open.")
    if s.options:
        out.append(f'Waiting for the user to choose: PRADA asked "{s.question}"')
        for n, o in enumerate(s.options, 1):
            extra = ", ".join(x for x in (_songs(o.tracks), o.app) if x)
            out.append(f"  {n}. {o.label}" + (f" ({extra})" if extra else ""))
    if s.offer:
        out.append(f'Waiting for a yes or no: PRADA offered to search online for "{s.offer}"')
    for asked in s.confirm:
        out.append(f'Waiting for a yes or no: PRADA asked "{asked}"')
    if s.done:
        out.append(f"What PRADA did in the last {s.since} minutes (her action log):")
        out += [f"- {d}" for d in s.done]
    if s.broken:
        out.append("Not working right now:")
        out += [f"- {b}" for b in s.broken]
    if s.abilities:
        out.append(
            "What PRADA can do when the user asks plainly (her system does it and tells her "
            "how it went):"
        )
        for a in s.abilities:
            asks = ", ".join(f'"{e}"' for e in a.examples)
            asks = f" (e.g. {asks})" if asks else ""
            out.append(f"- {a.about}{asks}")
    if s.remembered:
        out.append("What the user told PRADA in earlier conversations:")
        out += [f"- {r}" for r in s.remembered]
    return out


def render(s: Situation) -> str:
    return "\n".join(lines(s))


def _heard(src: Source) -> str:
    if src.via == "typed":
        return "The user typed this message."
    how = {"wake word": "after the wake word", "follow-up": "as a follow-up, without the wake word"}
    how["interruption"] = "interrupting PRADA"
    parts = [how[src.start]] if src.start else []
    if src.ear:
        parts.append(f"heard by {src.ear}")
    if src.language:
        parts.append(f"in {src.language}")
    line = "The user said this by voice" + (f" ({', '.join(parts)})" if parts else "") + "."
    if src.sure is not None and src.sure < UNSURE:
        line += " The ears weren't sure which language it was: some words may be misheard."
    return line


def _songs(n: int) -> str:
    return "" if not n else f"{n} song{'s' if n > 1 else ''}"
