"""The flight recorder: one row per turn, so a wrong answer can be looked at afterwards.

Each turn the node handles is recorded with what was heard, the path it took (which code
shortcut or the router, and which module answered), every model call (which prompt, tokens
in and out when the model reports them, milliseconds), the actions and how each ended, the
reply and the total time. `prada trace` reads it in plain words; `prada trace --case N`
turns turn N into an eval case line to paste into an evalset.

Only the newest `KEEP_DAYS` days (and at most `MAX_TURNS` rows) are kept. Writing here never
changes a reply: a store error is printed and the turn is simply not recorded.
"""

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

MIGRATIONS = [
    """
    CREATE TABLE trace_turn (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,         -- UTC, ISO 8601, when the turn started
        session TEXT,
        heard TEXT NOT NULL,      -- the request's text, as it reached the node
        module TEXT,              -- who answered
        reply TEXT NOT NULL,
        error TEXT,
        total_ms INTEGER NOT NULL,
        detail TEXT NOT NULL      -- JSON: path, model calls, actions, options
    );
    CREATE INDEX trace_ts ON trace_turn(ts);
    """
]

KEEP_DAYS = 14
MAX_TURNS = 5000
SHOWN = 10  # turns `prada trace` shows by default
BEFORE = 3  # earlier turns of the same session put in a case's "before"
BEFORE_GAP = 30 * 60  # seconds: an earlier turn older than this is another conversation
CHARS = 2000  # longest text kept per field


@dataclass
class Call:
    """One model call: `prompt` = who asked and with which schema ("router/Route")."""

    prompt: str
    ms: int
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str | None = None


@dataclass
class Act:
    """One action and how its story ended (the action log's status, or "asked")."""

    kind: str
    summary: str
    status: str
    ms: int = 0
    error: str | None = None


@dataclass
class Turn:
    """What the node collects while it handles one request."""

    heard: str
    session: str | None = None
    options: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)  # wall clock, for the timestamp
    t0: float = field(default_factory=time.perf_counter)  # for the total time
    path: list[str] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    acts: list[Act] = field(default_factory=list)
    situation: str = ""  # as the model was shown it (prada.situation.render)

    def step(self, text: str) -> None:
        self.path.append(text)


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC).timestamp()


def prompt_name(owner: str, schema) -> str:
    """ "router/Route": who made the call and the title of the schema it asked for."""
    title = schema.get("title") if isinstance(schema, dict) else None
    return f"{owner}/{title}" if title else owner


class Trace:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float] = time.time):
        self._db = db
        self._clock = clock

    def add(self, turn: Turn, module: str | None, reply: str, error: str | None) -> int:
        """Store a finished turn and drop what is too old; returns its id."""
        now = self._clock()
        detail = {
            "path": turn.path,
            "calls": [asdict(c) for c in turn.calls],
            "actions": [asdict(a) for a in turn.acts],
            "options": turn.options,
            "situation": turn.situation,
        }
        with self._db:
            cur = self._db.execute(
                "INSERT INTO trace_turn (ts, session, heard, module, reply, error, total_ms, detail)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _utc(turn.started),
                    turn.session,
                    turn.heard[:CHARS],
                    module,
                    reply[:CHARS],
                    error,
                    round((time.perf_counter() - turn.t0) * 1000),
                    json.dumps(detail, ensure_ascii=False),
                ),
            )
            self._db.execute(
                "DELETE FROM trace_turn WHERE ts < ? OR id <= ?",
                (_utc(now - KEEP_DAYS * 86400), cur.lastrowid - MAX_TURNS),
            )
        return cur.lastrowid

    def last(self, n: int = SHOWN) -> list[dict]:
        """The newest `n` turns, oldest first."""
        rows = self._db.execute(
            "SELECT * FROM trace_turn ORDER BY id DESC LIMIT ?", (max(0, n),)
        ).fetchall()
        return [_row(r) for r in reversed(rows)]

    def get(self, turn_id: int) -> dict | None:
        r = self._db.execute("SELECT * FROM trace_turn WHERE id = ?", (turn_id,)).fetchone()
        return _row(r) if r else None

    def before(self, turn: dict, n: int = BEFORE) -> list[dict]:
        """Up to `n` turns of the same session just before `turn`, in the same conversation
        (no gap over `BEFORE_GAP`), oldest first."""
        if turn["session"] is None:
            return []
        rows = self._db.execute(
            "SELECT * FROM trace_turn WHERE session = ? AND id < ? ORDER BY id DESC LIMIT ?",
            (turn["session"], turn["id"], n),
        ).fetchall()
        out, at = [], _ts(turn["ts"])
        for r in rows:
            if at - _ts(r["ts"]) > BEFORE_GAP:
                break
            out.append(_row(r))
            at = _ts(r["ts"])
        return list(reversed(out))


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d.update(json.loads(d.pop("detail") or "{}"))
    return d


def describe(turn: dict) -> str:
    """A turn in plain words, a few lines."""
    at = datetime.fromtimestamp(_ts(turn["ts"])).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    head = f"#{turn['id']}  {at}  {_secs(turn['total_ms'])}  -> {turn['module'] or '?'}"
    lines = [head, f"  heard: {turn['heard']!r}"]
    if turn.get("path"):
        lines.append("  path:  " + "; ".join(turn["path"]))
    for c in turn.get("calls", []):
        tokens = ""
        if c.get("tokens_in") is not None or c.get("tokens_out") is not None:
            n_in, n_out = (c.get(k) for k in ("tokens_in", "tokens_out"))
            tokens = f", {'?' if n_in is None else n_in} tokens in, {'?' if n_out is None else n_out} out"
        failed = f", failed: {c['error']}" if c.get("error") else ""
        lines.append(f"  model: {c['prompt']} {_secs(c['ms'])}{tokens}{failed}")
    for a in turn.get("actions", []):
        why = f" ({a['error']})" if a.get("error") else ""
        took = f" {_secs(a['ms'])}" if a.get("ms") else ""
        lines.append(f"  did:   {a['summary']} [{a['kind']}]: {a['status']}{took}{why}")
    if turn.get("situation"):
        lines.append("  saw:   " + turn["situation"].replace("\n", "\n         "))
    if turn["error"]:
        lines.append(f"  error: {turn['error']}")
    lines.append(f"  said:  {turn['reply']!r}")
    return "\n".join(lines)


def case_line(turn: dict, before: list[dict]) -> str:
    """The turn as a line for the assistant evalset (prada/evalset/assistant.jsonl): what she
    heard and the module that answered. The trace only knows what she did, not what was
    right, so check `module` (or swap it for `says`) before pasting it."""
    case = {"category": "routing", "text": turn["heard"]}
    if before:
        case["before"] = [b["heard"] for b in before]
    if turn["module"]:
        case["module"] = turn["module"]
    return json.dumps(case, ensure_ascii=False)


def _secs(ms: int) -> str:
    return f"{ms} ms" if ms < 1000 else f"{ms / 1000:.1f} s"
