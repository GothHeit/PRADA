"""Understand: what does kafka want done, given the situation she is in?

One constrained model call reads the situation (`render`), the last few turns and kafka's
newest words. It returns a one-sentence `note` first, then the tool `calls` kafka asked for,
each with a `quote` of kafka's own words. Code then checks every call: the quote must be in
kafka's words of this turn or the one before (ignoring case, spaces and punctuation), and
the arguments must fit the tool and the situation (a player that is open, an option that
was offered). A call that fails is dropped with the reason, never repaired by guessing; a
reply that isn't valid JSON means no calls.

Pure: nothing but `chat` is called and nothing runs. Nothing at runtime uses it yet:
`prada eval --suite understand --mind understand` measures it against today's rules.
"""

import json
import re
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ValidationError

from prada.module import Chat
from prada.music.play import fold
from prada.situation import Option, Player, Situation, render  # noqa: F401 (Option: re-export)

APPS = ("Spotify", "Corpus")
KINDS = ("any", "song", "album", "artist", "playlist", "mood")
MODES = ("play", "queue")


@dataclass(frozen=True)
class Tool:
    name: str
    about: str  # one line, for the prompt
    args: tuple[str, ...] = ()
    needs: tuple[str, ...] = ()  # arguments that must not be empty


_PLAYER = "app: the player, Spotify or Corpus, as the situation shows it (one call per player)"
TOOLS = (
    Tool(
        "music.play",
        "play music on the PC. what: the song, album, artist, genre or mood as the user "
        "said it (empty = anything); kind: song, album, artist, playlist, mood or any; app: "
        "Spotify or Corpus, only when the user named one; mode: play, or queue to add it "
        "after what is playing",
        ("what", "kind", "app", "mode"),
    ),
    Tool("music.pause", f"pause a player that is playing. {_PLAYER}", ("app",)),
    Tool("music.resume", f"resume a paused player. {_PLAYER}", ("app",)),
    Tool("music.next", f"skip to the next song on a player that is playing. {_PLAYER}", ("app",)),
    Tool("music.previous", f"go back to the previous song. {_PLAYER}", ("app",)),
    Tool(
        "music.choose",
        "the user picks one of the options PRADA just asked them to choose from. option: "
        "its number",
        ("option",),
        ("option",),
    ),
    Tool("notes.save", "write a note. text: what to write, short", ("text",), ("text",)),
    Tool("wake.send", "turn on the user's PC (it can only turn it on)"),
    Tool(
        "web.search",
        "search the internet now. query: what to search for",
        ("query",),
        ("query",),
    ),
    Tool(
        "web.offer",
        "ask the user whether to search the internet, for a question that needs live or "
        "recent facts. query: what to search for",
        ("query",),
    ),
    Tool(
        "memory.fix",
        "PRADA heard or remembered a word wrong in something the user told her. wrong: her "
        "word; right: the user's word",
        ("wrong", "right"),
        ("wrong", "right"),
    ),
)
BY_NAME = {t.name: t for t in TOOLS}

_PROMPT = """You decide what PRADA, a home assistant, should do for the user right now. You do not talk to the user and you are not a character: you only write down the actions the user asked for. Most messages ask for none.

Fill the JSON object: first note (one short sentence: what the user wants right now), then calls (an empty list when nothing should be done). Each call has quote, tool and that tool's arguments.

Tools (only these, with only the arguments listed):
{tools}

Rules:
1. Make a call only for what the user's newest message asks for, or for their answer to a question or offer PRADA made in the last turn (a yes, or which option). Talking about music, games, work, plans or the past is not a request.
2. quote: the user's own words that ask for the call, copied exactly from the newest message, or from the user's previous message when the newest one only says yes or adjusts that request.
3. If the user changes their mind inside the message, only the final wish counts. If they take the request back, make no call.
4. If the user says PRADA got their last request wrong, make the request they meant, repeating what still holds from before. memory.fix is only for a word PRADA heard or remembered wrong in something the user told her about themselves, with both words in the newest message. New facts about the user's life that replace old ones are not a fix.
5. Never call for what the situation shows is already so: no pause for a paused player, no skip, pause or previous when nothing is playing, no resume for what is playing.
6. A question PRADA can answer from what she knows or sees needs no call. A question that needs live or recent facts from the internet (weather, news, prices, results) gets web.offer. web.search only when the user asks to search or look something up, or says yes to PRADA's offer.
7. If no tool does what the user wants, make no call. Never use a tool for something it doesn't do.
8. A polite or indirect request (a wish, a question, a complaint about what is playing) is still a request: make the call it asks for. The user may speak Portuguese or mix languages; write arguments in the user's words, without filler."""


def prompt() -> str:
    """The system prompt: fixed text, so Ollama can reuse its cache from turn to turn."""
    return _PROMPT.format(tools="\n".join(f"- {t.name}: {t.about}" for t in TOOLS))


def schema() -> dict:
    """Constrained output: `note` before `calls`, `tool` one of `TOOLS`, typed arguments."""
    text = {"type": "string"}
    call = {
        "type": "object",
        "properties": {
            "quote": text,
            "tool": {"type": "string", "enum": [t.name for t in TOOLS]},
            "what": text,
            "kind": {"type": "string", "enum": list(KINDS)},
            "app": {"type": "string", "enum": list(APPS)},
            "mode": {"type": "string", "enum": list(MODES)},
            "option": {"type": "integer"},
            "text": text,
            "query": text,
            "wrong": text,
            "right": text,
        },
        "required": ["quote", "tool"],
    }
    return {
        "type": "object",
        "properties": {"note": text, "calls": {"type": "array", "items": call}},
        "required": ["note", "calls"],
    }


# --- the situation ----------------------------------------------------------------------


def messages(situation: Situation, turns: list[tuple[str, str]], text: str) -> list[dict]:
    """One user message: the situation, the last turns (user, PRADA), the newest words.
    The situation's abilities are left out: the tools in the prompt say what can be done."""
    parts = ["Situation:", render(replace(situation, abilities=[])), ""]
    if turns:
        parts.append("Last turns:")
        for said, replied in turns:
            parts += [f"User: {said}", f"PRADA: {replied}"]
        parts.append("")
    parts.append(f"The user's newest message: {text}")
    return [{"role": "system", "content": prompt()}, {"role": "user", "content": "\n".join(parts)}]


# --- the call and its checks ------------------------------------------------------------


class _Call(BaseModel):
    quote: str = ""
    tool: str
    what: str = ""
    kind: str = "any"
    app: str = ""
    mode: str = "play"
    option: int = 0
    text: str = ""
    query: str = ""
    wrong: str = ""
    right: str = ""


class _Reply(BaseModel):
    note: str
    calls: list


@dataclass
class Call:
    tool: str
    args: dict
    quote: str


@dataclass
class Understood:
    note: str = ""
    calls: list[Call] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # each: the call and why it was dropped
    valid: bool = True  # False: the reply was not the JSON asked for (so no calls)


def _bare(text: str) -> str:
    return fold(text).replace(" ", "")


def quote_ok(quote: str, said: list[str]) -> bool:
    """The quote is the user's own words: whole words in a row from one of `said`, ignoring
    case, accents, spaces and punctuation ("play" is not quoted from "I played tennis")."""
    q = _bare(quote)
    if len(q) < 2:
        return False
    for s in said:
        words = fold(s).split()
        for i in range(len(words)):
            run = ""
            for w in words[i:]:
                run += w
                if run == q:
                    return True
                if not q.startswith(run):
                    break
    return False


def _player_for(tool: str, app: str, players: list[Player] | None) -> tuple[str, str]:
    """(app, "") when the player fits the tool, else ("", why). With no app named, the one
    player that fits is taken."""
    want = "paused" if tool == "music.resume" else "playing"
    players = players or []
    fits = [p for p in players if p.state == want]
    if not app:
        if len(fits) == 1:
            return fits[0].app, ""
        return "", f"no {want} player" if not fits else "which player?"
    player = next((p for p in players if p.app.casefold() == app.casefold()), None)
    if player is None:
        return "", f"{app} is not open"
    if player.state != want:
        return "", f"{player.app} is {player.state}"
    return player.app, ""


def _check(raw: dict, situation: Situation, said: list[str], text: str) -> Call | str:
    """The checked call, or why it is dropped."""
    try:
        c = _Call.model_validate(raw)
    except ValidationError:
        return "malformed"
    tool = BY_NAME.get(c.tool)
    if tool is None:
        return "unknown tool"
    if not quote_ok(c.quote, said):
        return f"quote {c.quote!r} is not the user's words"
    args = {name: getattr(c, name) for name in tool.args}
    for name in tool.needs:
        if not args[name]:
            return f"no {name}"
    if c.tool == "music.play":
        if args["kind"] not in KINDS or args["mode"] not in MODES:
            return "bad kind or mode"
        if args["app"] and args["app"] not in APPS:
            return f"unknown app {args['app']!r}"
    elif c.tool.startswith("music.") and c.tool != "music.choose":
        app, why = _player_for(c.tool, c.app, situation.players)
        if why:
            return why
        args["app"] = app
    elif c.tool == "music.choose" and not 1 <= c.option <= len(situation.options):
        return "no such option"
    elif c.tool == "memory.fix":
        if _bare(c.wrong) == _bare(c.right):
            return "same word"
        if not set(fold(c.right).split()) <= set(fold(text).split()):
            return "the right word is not in the newest message"
    return Call(c.tool, args, c.quote)


def understand(chat: Chat, situation: Situation, turns: list[tuple[str, str]], text: str):
    """What the user wants done: a note and the checked calls (see the module docstring).
    `turns` are the last exchanges, oldest first, as (user, PRADA)."""
    reply = chat(messages=messages(situation, turns, text), schema=schema())
    try:
        data = _Reply.model_validate_json(reply)
    except (ValidationError, ValueError):
        return Understood(valid=False)
    said = [text] + ([turns[-1][0]] if turns else [])
    out = Understood(note=data.note.strip())
    for raw in data.calls:
        checked = _check(raw, situation, said, text) if isinstance(raw, dict) else "malformed"
        if isinstance(checked, str):
            out.dropped.append(f"{json.dumps(raw, ensure_ascii=False)}: {checked}")
        elif all((checked.tool, checked.args) != (k.tool, k.args) for k in out.calls):
            out.calls.append(checked)
    return out


# --- offline ----------------------------------------------------------------------------

_FAKE = re.compile(r"^(?:play|toca)\s+(.+)$|^(pause|skip|next)\b", re.IGNORECASE)


def fake_chat(messages, schema, model=None) -> str:
    """Offline understand: "play X" plays X, "pause" and "skip" act on the player that
    fits; anything else is chat."""
    text = messages[-1]["content"].rpartition("The user's newest message: ")[2].strip()
    m = _FAKE.match(text)
    calls = []
    if m and m.group(1):
        calls.append({"quote": text, "tool": "music.play", "what": m.group(1)})
    elif m:
        verb = "music.pause" if m.group(2).lower() == "pause" else "music.next"
        calls.append({"quote": text, "tool": verb})
    return json.dumps({"note": "fake understand: keyword match", "calls": calls})
