"""Keyword search over what PRADA has said and kept: past turns and episode summaries.

`KeywordIndex` is SQLite FTS5 (stdlib, no dependency) with accent- and case-insensitive
matching ("acao" finds "ação"). Each turn is indexed with its episode's summary beside it
(the `topic` column), so a vague line like "ok, let's do that" is found by what the
conversation was about; the stored turn itself never changes. Its three methods, `add_turns`, `add_summary` and
`search(query, limit) -> list[Hit]`, are the whole interface `prada.memory` uses, so an
embedding index can replace or sit beside it later without touching the callers.

Limits: matching is by whole word, so there is no stemming ("running" does not find "run")
and no synonyms; that is what a vector index is for. Chinese and Japanese have no spaces, so
every kana/kanji character is indexed on its own and a query run is matched as a phrase of
consecutive characters: "日本語" finds "日本語を勉強" but a word written in other kana or kanji
is a different phrase, and particles are not removed. Hangul and Thai are not special-cased.
"""

import re
import sqlite3
import unicodedata
from dataclasses import dataclass

# Kana (incl. the long-vowel mark), CJK ideographs and their extensions, halfwidth katakana.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
_CJK_CHAR = re.compile(f"([{_CJK}])")
_TOKEN = re.compile(f"[{_CJK}]+|(?:(?![{_CJK}])[^\\W_])+")
_HIRAGANA_PAIR = re.compile("[\u3040-\u309f]{2}")
_HAN = re.compile("[㐀-䶿一-鿿豈-﫿]")

MIN_WORD = 3  # shorter non-CJK words are too common to say anything
# Words too common to point at a memory (English, Portuguese, French).
STOPWORDS = frozenset(
    """
    the and for are but not you your yours with this that these those have has had was were
    what when where who whom how why can could would should will just about from they them
    their then than into out all any some there here his her its our been being does did
    tell remember know said say told please
    que para com uma uns por mais como mas dos das nos nas não nao sim você voce vocês meu
    minha meus isso esse essa este esta está estou tem foi são ser sobre quando onde quem
    les des est une pas pour dans qui sur avec vous nous mais sont être plus tout cette
    """.split()
)

# English contractions: a negation ("don't", "won't") says nothing about a memory and
# "'s"/"'re" endings are cut, so "kafka's" is "kafka" (the index splits it the same way).
_NEGATION = re.compile(r"\w+n['’]t\b")
_ENDING = re.compile(r"['’](?:s|m|ll|re|ve|d)\b")
TOPIC_WEIGHT = 0.5  # bm25 weight of a turn's topic against its own words (1.0)


@dataclass
class Hit:
    kind: str  # "turn" or "summary"
    ref: int  # memory_turn.id, or the episode id for a summary
    episode: int
    ts: float
    role: str  # "user" / "assistant"; "" for a summary
    text: str
    matched: int  # how many distinct query terms it contains (a turn: with its topic)
    own: int  # how many of those are in `text` itself (0 = found by its topic only)
    rank: float  # bm25, lower is better


def fold(text: str) -> str:
    """Text as indexed: every kana/kanji character on its own."""
    return _CJK_CHAR.sub(r" \1 ", text)


def terms(query: str) -> list[str]:
    """The searchable terms of a message: words of `MIN_WORD`+ characters that are not
    stopwords, and for each run of kana/kanji its overlapping pairs of characters (a lone
    ideograph is a term itself; a lone kana is a particle and is dropped). Pairs of two
    hiragana are mostly grammar (です, ので) and are dropped unless nothing else is left."""
    found = []
    query = _ENDING.sub("", _NEGATION.sub(" ", query.lower()))
    for tok in _TOKEN.findall(query):
        if _CJK_CHAR.match(tok):
            if len(tok) == 1:
                if _HAN.match(tok):
                    found.append(tok)
                continue
            pairs = [tok[i : i + 2] for i in range(len(tok) - 1)]
            content = [p for p in pairs if not _HIRAGANA_PAIR.fullmatch(p)]
            found += [" ".join(p) for p in content or pairs]
        elif len(tok) >= MIN_WORD and tok not in STOPWORDS and not tok.isdigit():
            found.append(tok)
    return list(dict.fromkeys(found))


def _match(term: str) -> str:
    return '"' + term + '"'  # a phrase; terms never contain a double quote


class KeywordIndex:
    def __init__(self, db: sqlite3.Connection):
        """Creates the index tables if missing. `available` is False when this SQLite has
        no FTS5, in which case adding is a no-op and search finds nothing."""
        self.db = db
        try:
            cols = [r[1] for r in db.execute("PRAGMA table_info(memory_turn_fts)")]
            if cols and "topic" not in cols:  # built before topics: `backfill` rebuilds it
                db.execute("DROP TABLE memory_turn_fts")
            for table, columns in (
                ("memory_turn_fts", "content, topic"),
                ("memory_summary_fts", "content"),
            ):
                db.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
                    f"USING fts5({columns}, tokenize='unicode61 remove_diacritics 2')"
                )
            db.commit()
            self.available = True
        except sqlite3.OperationalError:
            self.available = False

    def backfill(self) -> None:
        """Index what is stored but not indexed yet (an older database, or turns written
        while the index was unavailable). Turn ids only grow, so this is one range scan."""
        if not self.available:
            return
        last = self.db.execute("SELECT MAX(rowid) FROM memory_turn_fts").fetchone()[0] or 0
        with self.db:
            self.db.executemany(
                "INSERT INTO memory_turn_fts (rowid, content, topic) VALUES (?, ?, ?)",
                [
                    (i, fold(c), fold(topic))
                    for i, c, topic in self.db.execute(
                        "SELECT t.id, t.content, CASE WHEN e.model IS NOT 'fake' "
                        "THEN COALESCE(e.summary, '') ELSE '' END "
                        "FROM memory_turn t JOIN memory_episode e ON e.id = t.episode "
                        "WHERE t.id > ? ORDER BY t.id",
                        (last,),
                    )
                ],
            )
            for row in self.db.execute(
                "SELECT id, summary FROM memory_episode WHERE summary != '' "
                "AND model IS NOT 'fake' AND id NOT IN (SELECT rowid FROM memory_summary_fts)"
            ).fetchall():
                self.add_summary(row[0], row[1])

    def add_turns(self, rows) -> None:
        """`rows`: (memory_turn id, content) pairs of a conversation still going (no topic
        yet). Call inside the caller's transaction."""
        if self.available:
            self.db.executemany(
                "INSERT INTO memory_turn_fts (rowid, content, topic) VALUES (?, ?, '')",
                [(i, fold(c)) for i, c in rows],
            )

    def replace_turn(self, turn: int, content: str) -> None:
        """A stored turn's text changed (she was cut off). Call inside the caller's
        transaction."""
        if self.available:
            self.db.execute(
                "UPDATE memory_turn_fts SET content = ? WHERE rowid = ?", (fold(content), turn)
            )

    def add_summary(self, episode: int, summary: str) -> None:
        """Index an episode's summary, and make it the topic of every turn of the episode."""
        if self.available:
            self.db.execute(
                "UPDATE memory_turn_fts SET topic = ? "
                "WHERE rowid IN (SELECT id FROM memory_turn WHERE episode = ?)",
                (fold(summary), episode),
            )
            self.db.execute("DELETE FROM memory_summary_fts WHERE rowid = ?", (episode,))
            if summary:
                self.db.execute(
                    "INSERT INTO memory_summary_fts (rowid, content) VALUES (?, ?)",
                    (episode, fold(summary)),
                )

    def search(self, query: str, limit: int) -> list[Hit]:
        """Best `limit` matches for the terms of `query`: most distinct terms first, then
        terms in the hit's own text, then bm25 (a turn's topic weighs `TOPIC_WEIGHT`). Anything
        matching a single term is returned too, so callers filter."""
        wanted = terms(query)
        if not self.available or not wanted:
            return []
        match = " OR ".join(_match(t) for t in wanted)
        hits = [
            *self._run(
                "SELECT 'turn', t.id, t.episode, t.ts, t.role, t.content, f.content, f.topic, "
                f"bm25(memory_turn_fts, 1.0, {TOPIC_WEIGHT}) AS r "
                "FROM memory_turn_fts f JOIN memory_turn t ON t.id = f.rowid "
                # own words first, so a long chat's vague lines can't crowd out the real ones
                "WHERE memory_turn_fts MATCH ? "
                "ORDER BY bm25(memory_turn_fts, 1.0, 0.0), r LIMIT ?",
                match,
                limit,
                wanted,
            ),
            *self._run(
                "SELECT 'summary', e.id, e.id, e.started, '', e.summary, f.content, '', f.rank "
                "FROM memory_summary_fts f JOIN memory_episode e ON e.id = f.rowid "
                "WHERE memory_summary_fts MATCH ? ORDER BY f.rank LIMIT ?",
                match,
                limit,
                wanted,
            ),
        ]
        hits.sort(key=lambda h: (-h.matched, -h.own, h.rank))
        return hits[:limit]

    def _run(self, sql: str, match: str, limit: int, wanted: list[str]) -> list[Hit]:
        out = []
        for kind, ref, episode, ts, role, text, folded, topic, rank in self.db.execute(
            sql, (match, limit)
        ):
            own, near = _found(folded, wanted), _found(topic, wanted)
            out.append(Hit(kind, ref, episode, ts, role, text, len(own | near), len(own), rank))
        return out


def _found(folded: str, wanted: list[str]) -> set[str]:
    """The terms of `wanted` in an indexed text, accents ignored on both sides as the index does."""
    words = " " + " ".join(_TOKEN.findall(_plain(folded.lower()))) + " "
    return {t for t in wanted if f" {_plain(t)} " in words}


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
