"""PRADA's long-term memory of conversations: what was said, and what she keeps of it.

Owned by the node, not a module, so every module shares one PRADA. Every finished turn is
stored. Turns close together form an *episode* (a conversation); after `EPISODE_GAP` of
silence the next turn starts a new one. The current episode is what `recent` returns, so
she continues where you left off even after the node restarts, and a fresh start after a
long silence. Once an episode is over the model writes a short first-person summary of it
(through the persona, like everything she says); `remembered` loads those summaries,
newest first, within a character budget. No embeddings: just the latest summaries.
"""

import json
import sqlite3
import time
from collections.abc import Callable

from pydantic import BaseModel, Field

from prada import persona

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

_TASK = (
    "You are writing down what you want to remember of the conversation below, for your "
    "own memory, in the first person and in at most four short sentences. Keep what will "
    "matter later: facts about the user (names, plans, preferences, what they are working "
    "on or learning), what they asked you to remember, and anything you promised. Leave "
    "out small talk and anything you are not sure of. Only say you did something if the "
    "conversation shows it happened. Fill the JSON object matching the provided schema."
)


class Summary(BaseModel):
    summary: str = Field(..., description="What you remember of the conversation")


def fake_chat(messages, schema, model=None) -> str:
    return json.dumps({"summary": "(fake summary) " + messages[-1]["content"][:80]})


class Memory:
    def __init__(self, db: sqlite3.Connection, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock

    def recent(self, limit: int) -> list[dict]:
        """The last `limit` messages of the conversation still going, oldest first."""
        row = self.db.execute("SELECT id, ended FROM memory_episode ORDER BY id DESC").fetchone()
        if row is None or self.clock() - row["ended"] > EPISODE_GAP:
            return []
        rows = self.db.execute(
            "SELECT role, content FROM memory_turn WHERE episode = ? ORDER BY id DESC LIMIT ?",
            (row["id"], limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

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
            self.db.executemany(
                "INSERT INTO memory_turn (episode, ts, role, content) VALUES (?, ?, ?, ?)",
                [(episode, now, "user", said), (episode, now, "assistant", replied)],
            )

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
            lines = [
                f"{'User' if t['role'] == 'user' else persona.NAME}: {t['content'][:MESSAGE_CHARS]}"
                for t in turns
            ]
            jobs.append((episode["id"], "\n".join(lines)[-TRANSCRIPT_CHARS:]))
        return jobs

    def store(self, episode: int, summary: str, model: str) -> None:
        with self.db:
            self.db.execute(
                "UPDATE memory_episode SET summary = ?, model = ? WHERE id = ?",
                (summary, model, episode),
            )

    def remembered(self, budget: int = MEMORY_CHARS) -> str:
        """The newest episode summaries, one line each with its date, up to `budget` chars."""
        rows = self.db.execute(
            "SELECT started, summary FROM memory_episode "
            "WHERE summary IS NOT NULL AND summary != '' AND model IS NOT 'fake' "
            "ORDER BY id DESC"
        ).fetchall()
        lines, used = [], 0
        for r in rows:
            line = f"- {time.strftime('%Y-%m-%d', time.localtime(r['started']))}: {r['summary']}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(reversed(lines))  # oldest first, so the story reads forward


def summarize(chat: Callable[..., str], transcript: str) -> str:
    """A first-person summary of a transcript, written by the model (raises if it can't)."""
    messages = [
        {"role": "system", "content": persona.system_prompt(_TASK)},
        {"role": "user", "content": transcript},
    ]
    raw = chat(messages=messages, schema=Summary.model_json_schema())
    return Summary.model_validate_json(raw).summary.strip()[:SUMMARY_CHARS]
