"""PRADA's long-term memory of conversations: what was said, and what she keeps of it.

Owned by the node, not a module, so every module shares one PRADA. Every finished turn is
stored. Turns close together form an *episode* (a conversation); after `EPISODE_GAP` of
silence the next turn starts a new one. The current episode is what `recent` returns, so
she continues where you left off even after the node restarts, and a fresh start after a
long silence. Once an episode is over the model writes a short first-person summary of it
(through the persona, like everything she says); `remembered` loads those summaries,
newest first, within a character budget.

The same model call also writes down *facts* about the person, each with their own words
and when they said them (evidence, not her paraphrase): the quote is checked against what
they really said, or the fact is dropped. A fact that corrects an older one supersedes it;
the old one is kept, marked, so she can still say what they used to think.

`recall` searches everything stored, turns, summaries and facts, for what the user just said,
by keyword (prada.search) and by meaning (prada.vectors), merges both rankings (reciprocal
rank fusion) and returns the few best she hasn't been given yet, said the way a person
remembers: "About three months ago, one morning, you told me ...".
"""

import json
import re
import sqlite3
import time
from collections.abc import Callable
from contextlib import contextmanager

from pydantic import BaseModel, Field

from prada import persona, status
from prada.search import Hit, KeywordIndex, plain, terms
from prada.vectors import Embedder, VectorIndex

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
    """,
    """
    CREATE TABLE memory_fact (
        id INTEGER PRIMARY KEY,
        episode INTEGER NOT NULL REFERENCES memory_episode(id),
        turn INTEGER NOT NULL REFERENCES memory_turn(id),  -- the message the quote is from
        ts REAL NOT NULL,          -- when they said it
        fact TEXT NOT NULL,        -- "You play the cello."
        quote TEXT NOT NULL,       -- their own words
        superseded_by INTEGER REFERENCES memory_fact(id),  -- NULL = still true
        model TEXT NOT NULL
    );
    CREATE TABLE memory_vector (
        kind TEXT NOT NULL,        -- 'turn', 'summary' or 'fact'
        ref INTEGER NOT NULL,
        model TEXT NOT NULL,       -- the embedding model
        vec BLOB NOT NULL,         -- float32, unit length
        PRIMARY KEY (kind, ref, model)
    );
    """,
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
RECALL_HITS = 3  # most matches `recall` returns
RECALL_CHARS = 900  # budget for them
RECALL_LINE = 300  # each match is cut to this
CUT_NOTE = "[you interrupted me here]"  # ends a reply she was stopped in the middle of
CUT_SILENT = "[you interrupted me before I said anything]"
RECALL_FETCH = 24  # candidates read from the index before the ones she already has are dropped
RRF_K = 60  # reciprocal rank fusion: a hit's score is the sum of 1 / (RRF_K + rank)
KNOWN_FACTS = 20  # facts the summarizer sees, so it can correct one instead of repeating it
MIN_QUOTE = 3  # characters a quote must have to count as evidence

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
    "say you did something if the conversation shows it happened. "
    "Also list the facts about them worth keeping for a long time: what they told you about "
    "themselves, their life, the people in it, their plans and preferences, or asked you to "
    "remember. Each fact is one short sentence speaking to them ('You play the cello.') with "
    "the quote: their own words from the conversation that say it, copied exactly from a "
    "'User:' line, never your words. No facts from drill sentences, nothing you guessed. If "
    "a fact changes or corrects one you already know (listed with numbers below the "
    "conversation), set replaces to that number; leave out facts you already know "
    "unchanged. Fill the JSON object matching the provided schema."
)
_ASK = "\n\nNow write what you remember, in the first person, speaking to the user ('I ...', 'You ...')."


class Fact(BaseModel):
    fact: str = Field(..., description="One short sentence to them, e.g. 'You play the cello.'")
    quote: str = Field(..., description="Their exact words from a User line that say it")
    replaces: int | None = Field(None, description="Number of a known fact this corrects")


class Summary(BaseModel):
    summary: str = Field(..., description="What you remember of the conversation")
    facts: list[Fact] = Field(default_factory=list, description="Facts about them to keep")


def fake_chat(messages, schema, model=None) -> str:
    return json.dumps({"summary": "(fake summary) " + messages[-1]["content"][:80]})


class Memory:
    def __init__(
        self,
        db: sqlite3.Connection,
        clock: Callable[[], float] = time.time,
        embedder: Embedder | None = None,
    ):
        """`embedder`: what searches by meaning (prada.vectors); None = keywords only."""
        self.db = db
        self.clock = clock
        self.vectors = VectorIndex(db, embedder)
        # An older database gets its index built here, once; with no FTS5 she just can't search.
        self.index = KeywordIndex(db)
        try:
            self.index.backfill()
        except sqlite3.Error:  # a locked or damaged index must not stop the conversation
            self.index.available = False

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

    def last_talk(self) -> float | None:
        """When the last turn was stored, None if never."""
        return self.db.execute("SELECT MAX(ts) FROM memory_turn").fetchone()[0]

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

    def cut(self, text: str, heard: str) -> bool:
        """She was interrupted while saying `text`: if that is the newest reply stored, keep
        only `heard` of it, marked as cut off. False if the newest reply is another one."""
        text = text.strip()
        row = self.db.execute(
            "SELECT id, content FROM memory_turn WHERE role = 'assistant' ORDER BY id DESC"
        ).fetchone()
        if row is None or not text or not row["content"].endswith(text):
            return False
        note = row["content"][: -len(text)]  # e.g. the web module's "[from a web search] "
        heard = heard.strip()
        content = f"{note}{heard} {CUT_NOTE}" if heard else f"{note}{CUT_SILENT}"
        with self.db:
            self.db.execute("UPDATE memory_turn SET content = ? WHERE id = ?", (content, row["id"]))
            self.index.replace_turn(row["id"], content)
            self.vectors.forget("turn", row["id"])
        return True

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
        transcript is None when there is nothing the user said. Only reads: the model call
        that turns a transcript into a summary (`summarize`) happens outside the database.
        A conversation of one exchange still gets the call, for its facts, but one that is
        `short` keeps no summary."""
        jobs = []
        for episode in self._over():
            turns = self.db.execute(
                "SELECT role, content FROM memory_turn WHERE episode = ? ORDER BY id",
                (episode["id"],),
            ).fetchall()
            if not any(t["role"] == "user" for t in turns):
                jobs.append((episode["id"], None))
                continue
            lines = [_line(t["role"], t["content"][:MESSAGE_CHARS]) for t in turns]
            jobs.append((episode["id"], "\n".join(lines)[-TRANSCRIPT_CHARS:]))
        return jobs

    def short(self, episode: int) -> bool:
        """Too short (under `MIN_MESSAGES`) for a summary worth keeping."""
        count = self.db.execute(
            "SELECT COUNT(*) FROM memory_turn WHERE episode = ?", (episode,)
        ).fetchone()[0]
        return count < MIN_MESSAGES

    def known_facts(self, transcript: str = "") -> list[sqlite3.Row]:
        """Facts still true that the summarizer should see next to `transcript`, so a
        correction replaces one instead of repeating it: all of them when there are few,
        else the ones `transcript` shares words with, then the newest."""
        sql = (
            "SELECT id, fact, quote FROM memory_fact WHERE superseded_by IS NULL "
            "AND model IS NOT 'fake'"
        )
        rows = self.db.execute(sql + " ORDER BY id DESC LIMIT ?", (KNOWN_FACTS + 1,)).fetchall()
        if len(rows) <= KNOWN_FACTS:
            return rows[::-1]
        found = [h.ref for h in self.index.search(transcript, KNOWN_FACTS * 2) if h.kind == "fact"]
        ids = list(dict.fromkeys(found + [r["id"] for r in rows]))
        by_id = {
            r["id"]: r
            for r in self.db.execute(
                sql + f" AND id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
        }
        return [by_id[i] for i in ids if i in by_id][:KNOWN_FACTS]

    def store(
        self,
        episode: int,
        summary: str,
        model: str,
        facts: list[Fact] = (),
        known: list[int] = (),
    ) -> None:
        """Keep an episode's summary and the facts the model found in it. A fact whose quote
        is not something the user said in this episode is dropped; one that `replaces` a
        fact from `known` (the ids shown to the model) supersedes it, which stays."""
        said = self.db.execute(
            "SELECT id, ts, content FROM memory_turn WHERE episode = ? AND role = 'user' "
            "ORDER BY id DESC",
            (episode,),
        ).fetchall()
        with self.db:
            self.db.execute(
                "UPDATE memory_episode SET summary = ?, model = ? WHERE id = ?",
                (summary, model, episode),
            )
            if model != "fake":  # a fake summary is never loaded, so never found either
                self.index.add_summary(episode, summary)
            for fact in facts:
                text, quote = " ".join(fact.fact.split()), " ".join(fact.quote.split())
                turn = _quoted(quote, said)
                if turn is None or not text:
                    continue  # no evidence: she would be remembering her own words
                new = self.db.execute(
                    "INSERT INTO memory_fact (episode, turn, ts, fact, quote, model) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (episode, turn["id"], turn["ts"], text, quote, model),
                ).lastrowid
                if fact.replaces in known and fact.replaces != new:
                    self.db.execute(
                        "UPDATE memory_fact SET superseded_by = ? "
                        "WHERE id = ? AND superseded_by IS NULL",
                        (new, fact.replaces),
                    )
                if model != "fake":
                    self.index.add_fact(new, text, quote)

    def facts(self) -> list[sqlite3.Row]:
        """Every fact kept, oldest first (superseded ones too)."""
        return self.db.execute("SELECT * FROM memory_fact ORDER BY id").fetchall()

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

    @contextmanager
    def _snapshot(self):
        """One read transaction, so a summary written meanwhile can't make two reads disagree."""
        if self.db.in_transaction:
            yield
            return
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            self.db.commit()

    def remembered(self, budget: int = MEMORY_CHARS) -> str:
        """The newest episode summaries, one line each with its date, up to `budget` chars."""
        with self._snapshot():
            summaries, tail = self._summaries(budget), self._tail()
        text = "\n".join(line for _, line in reversed(summaries))  # oldest first
        if tail:
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

    def _tail_lines(self) -> tuple[sqlite3.Row | None, list[tuple[int, str]]]:
        """The tail conversation and the (turn id, line) pairs of it that fit `TAIL_CHARS`,
        oldest first: whole lines only, newest kept."""
        row, turns = self._tail_turns()
        lines, used = [], 0
        for t in reversed(turns):
            line = _line(t["role"], t["content"][:MESSAGE_CHARS])
            if used + len(line) > TAIL_CHARS and lines:
                break
            lines.append((t["id"], line))
            used += len(line) + 1
        return row, lines[::-1]

    def _tail(self) -> str:
        """The end of the newest finished conversation that has no summary yet, as a
        transcript. The summary is written after the next reply, so without this she would
        know nothing of the conversation she just came back from."""
        row, lines = self._tail_lines()
        if row is None:
            return ""
        return (
            f"Your latest conversation ({_day(row['started'])}), not summarized yet; the end "
            "of it:\n" + "\n".join(line for _, line in lines)
        )

    def recall(self, query: str, recent: int = 0) -> str:
        """What she finds when she searches her memory for `query` (what the user just
        said): up to `RECALL_HITS` facts, past messages or summaries, oldest first, one
        sentence each with roughly when, within `RECALL_CHARS`. Keyword and meaning hits
        are merged by reciprocal rank fusion. Never the last `recent` messages of the
        current conversation, the unsummarized tail or the summaries `remembered` already
        gives her; a superseded fact is shown as its newest version, and a message a shown
        fact quotes is left out. A turn found only by its topic (its episode's summary) is a
        vague line like "ok, let's do that": at most one per conversation, and none when
        `remembered` already gives her that conversation's summary. "" when nothing matches."""
        if not self.index.available and not self.vectors.available:
            return ""
        hits: dict[tuple[str, int], Hit] = {}
        score: dict[tuple[str, int], float] = {}
        need = min(2, len(terms(query)))  # one shared word is a coincidence, not a memory
        if self.index.available:
            ranked = [h for h in self.index.search(query, RECALL_FETCH) if h.matched >= need]
            for rank, hit in enumerate(ranked):
                key = (hit.kind, hit.ref)
                hits.setdefault(key, hit)
                score[key] = score.get(key, 0) + 1 / (RRF_K + rank)
        meant = set()
        for rank, (kind, ref, _) in enumerate(self.vectors.search(query, RECALL_FETCH)):
            key = (kind, ref)
            if key not in hits and (hit := self._hit(kind, ref)) is not None:
                hits[key] = hit
            if key in hits:
                meant.add(key)
                score[key] = score.get(key, 0) + 1 / (RRF_K + rank)
        order = sorted(score, key=lambda k: -score[k])

        with self._snapshot():
            skip_turns = {i for i, _ in self._tail_lines()[1]}
            if live := self._live():
                skip_turns |= {
                    r["id"]
                    for r in self.db.execute(
                        "SELECT id FROM memory_turn WHERE episode = ? ORDER BY id DESC LIMIT ?",
                        (live["id"], recent),
                    )
                }
            skip_summaries = {i for i, _ in self._summaries(MEMORY_CHARS)}
            facts = {k: self._fact(k[1]) for k in order if k[0] == "fact"}
        quoted = {  # the fact says it better
            row["turn"] for fact, before in facts.values() for row in (fact, before) if row
        }
        covered = set(skip_summaries)  # no vague line from these episodes (or a second one)
        now = self.clock()
        lines, used, seen = [], 0, set()
        for key in order:
            hit = hits[key]
            if hit.kind == "fact":
                fact, before = facts[key]
                if ("fact", fact["id"]) in seen:
                    continue
                seen.add(("fact", fact["id"]))
                line, ts = _fact_line(fact, before, now), fact["ts"]
            else:
                if hit.ref in (skip_turns | quoted if hit.kind == "turn" else skip_summaries):
                    continue
                if hit.text in seen:
                    continue
                if hit.kind == "turn" and not hit.own and key not in meant:
                    if hit.episode in covered:
                        continue
                    covered.add(hit.episode)
                seen.add(hit.text)
                line, ts = _hit_line(hit, now), hit.ts
            if lines and used + len(line) > RECALL_CHARS:
                break
            lines.append(((ts, hit.kind, hit.ref), line))
            used += len(line) + 1
            if len(lines) == RECALL_HITS:
                break
        return "\n".join(line for _, line in sorted(lines, key=lambda x: x[0]))

    def _hit(self, kind: str, ref: int) -> Hit | None:
        """A hit found by meaning only, read from the store (None if it is gone or fake)."""
        if kind == "turn":
            sql = "SELECT id, episode, ts, role, content FROM memory_turn WHERE id = ?"
        elif kind == "summary":
            sql = (
                "SELECT id, id, started, '', summary FROM memory_episode WHERE id = ? "
                "AND summary != '' AND model IS NOT 'fake'"
            )
        else:
            sql = "SELECT id, episode, ts, 'user', fact FROM memory_fact WHERE id = ? AND model IS NOT 'fake'"
        row = self.db.execute(sql, (ref,)).fetchone()
        return None if row is None else Hit(kind, row[0], row[1], row[2], row[3], row[4], 0, 1, 0.0)

    def _fact(self, ref: int) -> tuple[sqlite3.Row, sqlite3.Row | None]:
        """The newest version of a fact, and the one it replaced (if any)."""
        row = self.db.execute("SELECT * FROM memory_fact WHERE id = ?", (ref,)).fetchone()
        for _ in range(100):  # a chain of corrections; bounded in case of a cycle
            if row["superseded_by"] is None:
                break
            row = self.db.execute(
                "SELECT * FROM memory_fact WHERE id = ?", (row["superseded_by"],)
            ).fetchone()
        before = self.db.execute(
            "SELECT * FROM memory_fact WHERE superseded_by = ? ORDER BY id DESC", (row["id"],)
        ).fetchone()
        return row, before


def _day(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


def _line(role: str, content: str) -> str:
    return f"{'User' if role == 'user' else persona.NAME}: {content}"


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text[:RECALL_LINE].rstrip() + "..." if len(text) > RECALL_LINE else text


def _when(ts: float, now: float) -> str:
    when = status.perceived(ts, now)
    return when[0].upper() + when[1:]


def _hit_line(hit: Hit, now: float) -> str:
    """A memory as she would say it to herself: "About a month ago, one evening, you said: ..."."""
    text = _short(hit.text)
    if hit.kind == "summary":
        return f"{_when(hit.ts, now)}, we talked; what I kept of it: {text}"
    who = "you said" if hit.role == "user" else "I said"
    return f'{_when(hit.ts, now)}, {who}: "{text}"'


def _fact_line(fact: sqlite3.Row, before: sqlite3.Row | None, now: float) -> str:
    text = fact["fact"].rstrip(".")
    told = (
        f"you told me {text[0].lower()}{text[1:]}"
        if re.match(r"(?i)your?\b", text)
        else "you told me"
    )
    line = f'{_when(fact["ts"], now)}, {told}, in your words: "{_short(fact["quote"])}"'
    if before is not None:
        when = status.perceived(before["ts"], now)
        line += f'. That changed what you had said {when}: "{_short(before["quote"])}"'
    return line


_CJK = re.compile("[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")


def _words(text: str) -> str:
    return " ".join(re.findall(r"\w+", plain(text.casefold())))


def _quoted(quote: str, said: list[sqlite3.Row]) -> sqlite3.Row | None:
    """The newest of the user's messages that contains `quote`, ignoring case, accents,
    punctuation and spacing; None if none does (the quote is not their words)."""
    words = _words(quote)
    if len(words) < MIN_QUOTE:
        return None
    # Whole words ("the" is no evidence inside "there"); CJK has no spaces to match on.
    pad = "" if _CJK.search(words) else " "
    words = f"{pad}{words}{pad}"
    return next((r for r in said if words in f"{pad}{_words(r['content'])}{pad}"), None)


def summarize(chat: Callable[..., str], transcript: str, known=()) -> Summary:
    """A first-person summary of a transcript and the facts in it, written by the model
    (raises if it can't). `known`: facts still true (rows with id, fact, quote), shown with
    their ids so a correction can name the one it replaces."""
    text = transcript
    if known:
        text += "\n\nFacts you already know about them (number: fact):\n" + "\n".join(
            f"{k['id']}: {k['fact']}" for k in known
        )
    messages = [
        {"role": "system", "content": persona.system_prompt(_TASK)},
        {"role": "user", "content": text + _ASK},
    ]
    raw = chat(messages=messages, schema=Summary.model_json_schema())
    result = Summary.model_validate_json(raw)
    result.summary = result.summary.strip()[:SUMMARY_CHARS]
    return result
