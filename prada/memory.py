"""PRADA's long-term memory of conversations: what was said, and what she keeps of it.

Owned by the node, not a module, so every module shares one PRADA. Every finished turn is
stored. Turns close together form an *episode* (a conversation); after `EPISODE_GAP` of
silence the next turn starts a new one. The current episode is what `recent` returns, so
she continues where you left off even after the node restarts, and a fresh start after a
long silence. Once an episode is over the model writes a short first-person summary of it
(through the persona, like everything she says); `remembered` loads those summaries,
newest first, within a character budget. `recall` searches everything stored, turns and
summaries, for what the user just said (keyword search, see prada.search; embeddings can plug
in behind the same index later) and returns the few best matches she hasn't been given yet.
"""

import json
import sqlite3
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from prada import persona
from prada.search import Hit, KeywordIndex, terms

MIGRATIONS = [
    """
    CREATE TABLE memory_episode (
        id INTEGER PRIMARY KEY,
        started REAL NOT NULL,
        ended REAL NOT NULL,
        summary TEXT,          -- NULL = not summarized yet, '' = too short to keep
        model TEXT             -- who wrote the summary ('fake' under PRADA_FAKE_LLM)
    );
    CREATE TABLE memory_turn (
        id INTEGER PRIMARY KEY,
        episode INTEGER NOT NULL REFERENCES memory_episode(id),
        ts REAL NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    );
    CREATE INDEX memory_turn_episode ON memory_turn(episode, id);
    """
]

EPISODE_GAP = 30 * 60  # seconds of silence that end a conversation
MIN_MESSAGES = 4  # an episode shorter than this (two exchanges) is not worth a summary
MEMORY_CHARS = 1500  # budget for the summaries loaded into a prompt (about 400 tokens)
SUMMARY_CHARS = 600  # a stored summary is cut to this
TRANSCRIPT_CHARS = 6000  # most of an episode shown to the summarizer (the end is kept)
MESSAGE_CHARS = 500  # each message is cut to this in that transcript
RECENT_CHARS = 1500  # and to this when the current episode goes back into a prompt
SUMMARIES_LOADED = 40  # most summaries read from the database for `remembered`
TAIL_MESSAGES = 8  # messages of the newest unsummarized conversation shown as-is
TAIL_CHARS = 1500  # budget for that transcript
RECALL_HITS = 4  # most matches `recall` returns
RECALL_CHARS = 900  # budget for them
RECALL_LINE = 300  # each match is cut to this
RECALL_FETCH = 24  # candidates read from the index before the ones she already has are dropped

_TASK = (
    "You are writing down what you want to remember of a conversation you just had with the "
    "user, for your own memory, in at most four short sentences. Write it as yourself, "
    "speaking to them: 'I' for you and 'you' for them, for example 'You told me you are "
    "learning Russian. I promised to quiz you on cases.' Never write 'he', 'she', 'they' or "
    "'the user' for them, and never guess their gender. Keep what will matter later: facts "
    "about them (names, plans, preferences, what they are working on or learning), what they "
    "asked you to remember, and anything you promised. Practice sentences they wrote for a "
    "language drill are exercises, not facts about them: note the language and the kinds of "
    "mistakes, not the sentences. Leave out small talk and anything you are not sure of. Only "
    "say you did something if the conversation shows it happened. Fill the JSON object "
    "matching the provided schema."
)
_ASK = "\n\nNow write what you remember, in the first person, speaking to the user ('I ...', 'You ...')."


class Summary(BaseModel):
    summary: str = Field(..., description="What you remember of the conversation")


def fake_chat(messages, schema, model=None) -> str:
    return json.dumps({"summary": "(fake summary) " + messages[-1]["content"][:80]})


class Memory:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock
        # An older database gets its index built here, once; with no FTS5 she just can't search.
        self.index = KeywordIndex(db)
        self.index.backfill()

    def _live(self) -> sqlite3.Row | None:
        """The conversation still going, if any."""
        row = self.db.execute("SELECT id, ended FROM memory_episode ORDER BY id DESC").fetchone()
        return None if row is None or self.clock() - row["ended"] > EPISODE_GAP else row

    def recent(self, limit: int) -> list[dict]:
        """The last `limit` messages of the conversation still going, oldest first."""
        row = self._live()
        if row is None:
            return []
        rows = self.db.execute(
            "SELECT role, content FROM memory_turn WHERE episode = ? ORDER BY id DESC LIMIT ?",
            (row["id"], limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"][:RECENT_CHARS]} for r in reversed(rows)]

    def add(self, said: str, replied: str) -> None:
        """Store a finished turn, starting a new episode after a long silence."""
        now = self.clock()
        with self.db:
            row = self.db.execute(
                "SELECT id, ended FROM memory_episode ORDER BY id DESC"
            ).fetchone()
            if row is None or now - row["ended"] > EPISODE_GAP:
                episode = self.db.execute(
                    "INSERT INTO memory_episode (started, ended) VALUES (?, ?)", (now, now)
                ).lastrowid
            else:
                episode = row["id"]
                self.db.execute("UPDATE memory_episode SET ended = ? WHERE id = ?", (now, episode))
            rows = []
            for role, content in (("user", said), ("assistant", replied)):
                turn = self.db.execute(
                    "INSERT INTO memory_turn (episode, ts, role, content) VALUES (?, ?, ?, ?)",
                    (episode, now, role, content),
                ).lastrowid
                rows.append((turn, content))
            self.index.add_turns(rows)

    def _over(self) -> list[sqlite3.Row]:
        """Episodes that ended and have no summary yet."""
        return self.db.execute(
            "SELECT id, started FROM memory_episode WHERE summary IS NULL AND ended < ? "
            "ORDER BY id",
            (self.clock() - EPISODE_GAP,),
        ).fetchall()

    def has_pending(self) -> bool:
        return bool(self._over())

    def jobs(self) -> list[tuple[int, str | None]]:
        """(episode id, transcript) for each finished episode without a summary; the
        transcript is None when it is too short to keep. Only reads: the model call that
        turns a transcript into a summary (`summarize`) happens outside the database."""
        jobs = []
        for episode in self._over():
            turns = self.db.execute(
                "SELECT role, content FROM memory_turn WHERE episode = ? ORDER BY id",
                (episode["id"],),
            ).fetchall()
            if len(turns) < MIN_MESSAGES:
                jobs.append((episode["id"], None))
                continue
            lines = [_line(t["role"], t["content"][:MESSAGE_CHARS]) for t in turns]
            jobs.append((episode["id"], "\n".join(lines)[-TRANSCRIPT_CHARS:]))
        return jobs

    def store(self, episode: int, summary: str, model: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE memory_episode SET summary = ?, model = ? WHERE id = ?",
                (summary, model, episode),
            )
            if model != "fake":  # a fake summary is never loaded, so never found either
                self.index.add_summary(episode, summary)

    def _summaries(self, budget: int) -> list[tuple[int, str]]:
        """(episode id, line) of the newest summaries that fit `budget`, newest first."""
        rows = self.db.execute(
            "SELECT id, started, summary FROM memory_episode "
            "WHERE summary IS NOT NULL AND summary != '' AND model IS NOT 'fake' "
            "ORDER BY id DESC LIMIT ?",
            (SUMMARIES_LOADED,),
        ).fetchall()
        lines, used = [], 0
        for r in rows:
            line = f"- {_day(r['started'])}: {r['summary']}"
            if used + len(line) > budget:
                break
            lines.append((r["id"], line))
            used += len(line) + 1
        return lines

    def remembered(self, budget: int = MEMORY_CHARS) -> str:
        """The newest episode summaries, one line each with its date, up to `budget` chars."""
        text = "\n".join(line for _, line in reversed(self._summaries(budget)))  # oldest first
        if tail := self._tail():
            text += ("\n\n" if text else "") + tail
        return text

    def _tail_turns(self) -> tuple[sqlite3.Row | None, list[sqlite3.Row]]:
        """The newest finished conversation that has no summary yet and the last
        `TAIL_MESSAGES` of it (oldest first)."""
        row = self.db.execute(
            "SELECT id, started FROM memory_episode WHERE summary IS NULL AND ended < ? "
            "ORDER BY id DESC",
            (self.clock() - EPISODE_GAP,),
        ).fetchone()
        if row is None:
            return None, []
        turns = self.db.execute(
            "SELECT id, role, content FROM memory_turn WHERE episode = ? ORDER BY id DESC LIMIT ?",
            (row["id"], TAIL_MESSAGES),
        ).fetchall()
        return row, turns[::-1]

    def _tail(self) -> str:
        """The end of the newest finished conversation that has no summary yet, as a
        transcript. The summary is written after the next reply, so without this she would
        know nothing of the conversation she just came back from."""
        row, turns = self._tail_turns()
        if row is None:
            return ""
        lines, used = [], 0
        for line in (_line(t["role"], t["content"][:MESSAGE_CHARS]) for t in reversed(turns)):
            if used + len(line) > TAIL_CHARS and lines:
                break  # whole lines only, newest kept
            lines.append(line)
            used += len(line) + 1
        return (
            f"Your latest conversation ({_day(row['started'])}), not summarized yet; the end "
            "of it:\n" + "\n".join(reversed(lines))
        )

    def recall(self, query: str, recent: int = 0) -> str:
        """What she finds when she searches her memory for `query` (what the user just
        said): up to `RECALL_HITS` real past messages or summaries, oldest first, one line
        each, within `RECALL_CHARS`. Never the last `recent` messages of the current
        conversation, the unsummarized tail or the summaries `remembered` already gives her.
        "" when nothing matches."""
        if not self.index.available:
            return ""
        skip_turns = {t["id"] for t in self._tail_turns()[1]}
        if live := self._live():
            skip_turns |= {
                r["id"]
                for r in self.db.execute(
                    "SELECT id FROM memory_turn WHERE episode = ? ORDER BY id DESC LIMIT ?",
                    (live["id"], recent),
                )
            }
        skip_summaries = {i for i, _ in self._summaries(MEMORY_CHARS)}
        lines, used, seen = [], 0, set()
        need = min(2, len(terms(query)))  # one shared word is a coincidence, not a memory
        for hit in self.index.search(query, RECALL_FETCH):
            if hit.matched < need:
                continue
            if hit.ref in (skip_turns if hit.kind == "turn" else skip_summaries):
                continue
            if hit.text in seen:
                continue
            seen.add(hit.text)
            line = _hit_line(hit)
            if lines and used + len(line) > RECALL_CHARS:
                break
            lines.append(((hit.ts, hit.kind == "summary", hit.ref), line))
            used += len(line) + 1
            if len(lines) == RECALL_HITS:
                break
        return "\n".join(line for _, line in sorted(lines, key=lambda x: x[0]))


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _line(role: str, content: str) -> str:
    return f"{'User' if role == 'user' else persona.NAME}: {content}"


def _hit_line(hit: Hit) -> str:
    text = " ".join(hit.text.split())
    if len(text) > RECALL_LINE:
        text = text[:RECALL_LINE].rstrip() + "..."
    if hit.kind == "summary":
        return f"- {_day(hit.ts)}, what I wrote down about that conversation: {text}"
    who = "you said" if hit.role == "user" else "I said"
    return f'- {_day(hit.ts)}, {who}: "{text}"'


def summarize(chat: Callable[..., str], transcript: str) -> str:
    """A first-person summary of a transcript, written by the model (raises if it can't)."""
    messages = [
        {"role": "system", "content": persona.system_prompt(_TASK)},
        {"role": "user", "content": transcript + _ASK},
    ]
    raw = chat(messages=messages, schema=Summary.model_json_schema())
    return Summary.model_validate_json(raw).summary.strip()[:SUMMARY_CHARS]
