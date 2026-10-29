"""The action log: an append-only record of everything PRADA was asked to do in the world.

One row per action, written when its story ends: it ran (`done`), ran but could not be
verified (`failed`), was refused by the node (`refused`), was asked about and the person said
no (`declined`), never answered (`expired`) or said something that wasn't a yes (`superseded`). Rows are only ever inserted, never changed.
`today` lets her answer "what did you do today?" from it, in her own words about what the
log says and nothing else.
"""

import json
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

MIGRATIONS = [
    """
    CREATE TABLE actionlog_entry (
        id INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,         -- UTC, ISO 8601
        module TEXT NOT NULL,
        kind TEXT NOT NULL,       -- "<module>.<verb>"
        summary TEXT NOT NULL,    -- Action.describe(): "wake your PC up"
        risk TEXT NOT NULL,       -- safe | reversible | irreversible
        confirmed INTEGER NOT NULL, -- 1 = the person said yes to this one
        status TEXT NOT NULL,     -- done | failed | refused | declined | expired | superseded
        detail TEXT NOT NULL,     -- the action's fields (JSON) and, on failure, why
        error TEXT
    );
    CREATE INDEX actionlog_ts ON actionlog_entry(ts);
    """
]

DETAIL_CHARS = 300
SHOWN = 12  # most entries read out loud
RECENT = 3  # most entries in the status block (prada.status)

_ASKS = re.compile(
    r"\b(?:what|which)\s+(?:(?:did|have)\s+you\s+(?:do|done|been\s+up\s+to)|things\s+did\s+you\s+do)"
    r"\s+(?P<when>today|yesterday)\b",
    re.IGNORECASE,
)


def asks_for_log(text: str) -> str | None:
    """ "today" or "yesterday" if `text` asks what she did, else None."""
    m = _ASKS.search(text)
    return m["when"].lower() if m else None


_ONE_DAY = timedelta(days=1)


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ActionLog:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float] = time.time):
        self._db = db
        self._clock = clock

    def add(
        self,
        module: str,
        action: dict,
        summary: str,
        risk: str,
        confirmed: bool,
        status: str,
        error: str | None = None,
    ) -> None:
        detail = json.dumps(
            {k: str(v)[:DETAIL_CHARS] for k, v in action.items() if k != "kind"},
            ensure_ascii=False,
        )
        with self._db:
            self._db.execute(
                "INSERT INTO actionlog_entry"
                " (ts, module, kind, summary, risk, confirmed, status, detail, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _utc(self._clock()),
                    module,
                    action.get("kind", "?"),
                    summary,
                    risk,
                    int(confirmed),
                    status,
                    detail,
                    error,
                ),
            )

    def recent(self, since: float, limit: int = RECENT) -> list[str]:
        """Her lines for the last `limit` entries since `since`, oldest first."""
        rows = self._db.execute(
            "SELECT ts, summary, confirmed, status, error FROM actionlog_entry"
            " WHERE ts >= ? ORDER BY id DESC LIMIT ?",
            (_utc(since), limit),
        ).fetchall()
        return [_line(r) for r in reversed(rows)]

    def day(self, when: str = "today") -> str:
        """What she did `when` (local time), from the log alone."""
        now = datetime.fromtimestamp(self._clock()).astimezone()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if when == "yesterday":
            start, end = start - _ONE_DAY, start
        else:
            end = start + _ONE_DAY
        rows = self._db.execute(
            "SELECT ts, summary, confirmed, status, error FROM actionlog_entry"
            " WHERE ts >= ? AND ts < ? ORDER BY id DESC LIMIT ?",
            (_utc(start.timestamp()), _utc(end.timestamp()), SHOWN),
        ).fetchall()
        if not rows:
            return f"I haven't done anything {when}, as far as my log shows."
        lines = [_line(r) for r in reversed(rows)]
        return f"Here's what my log shows for {when}: " + "; ".join(lines) + "."


def _line(r: sqlite3.Row) -> str:
    at = (
        datetime.strptime(r["ts"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=UTC)
        .astimezone()
        .strftime("%H:%M")
    )
    what, status = r["summary"], r["status"]
    if status == "done":
        if r["confirmed"]:
            return f"at {at} I asked whether to {what}, you said yes, and I did"
        return f"at {at} you asked me to {what}, and I did"
    if status == "failed":
        why = f" ({r['error'].rstrip('. ')})" if r["error"] else ""
        return f"at {at} I tried to {what}, but it didn't work{why}"
    if status == "declined":
        return f"at {at} I asked whether to {what}, and you said no, so I didn't"
    if status == "expired":
        return f"at {at} I asked whether to {what}, but you didn't answer in time, so I didn't"
    if status == "superseded":
        return f"at {at} I asked whether to {what}, but you didn't say yes, so I left it"
    return f"at {at} I meant to {what}, but I'm not allowed to, so I didn't"
