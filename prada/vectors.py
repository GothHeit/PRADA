"""Search by meaning: an embedding of everything PRADA keeps, compared by cosine similarity.

Each turn, summary and fact gets one vector from an embedding model (Ollama, `[memory]
embed_model`, default nomic-embed-text), stored as float32 bytes in `memory_vector` beside
the text. Search is brute force: the configured model's vectors sit in one numpy matrix and
a query is one matrix product, milliseconds for a home's worth of conversation (tens of
thousands of lines are a few dozen MB), with no other database. This is what finds "what
instrument do I play?" in "I practiced the cello all night", which share no word.

Vectors are written in the background after a reply (`missing` -> `embed` -> `add`), never
while she answers, so a slow or missing embedding model never delays a reply: until it
answers, she searches by keyword only (prada.search). Changing the model re-embeds
everything the same way, since vectors are kept per model.
"""

import sys
import time
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from prada import config, llm
from prada.search import terms

EMBED_CHARS = 1000  # each text is cut to this before it is embedded
FAKE_DIM = 256
RETRY = 60  # seconds without searching by meaning after the embedding model failed


@dataclass
class Embedder:
    model: str  # vectors are stored per model
    embed: Callable[[list[str], bool], list[list[float]]]  # (texts, is_query) -> vectors
    min_similarity: float = 0.55  # cosine a match must reach


def ollama_embed(model: str, keep_alive=None) -> Callable[[list[str], bool], list[list[float]]]:
    def embed(texts: list[str], query: bool) -> list[list[float]]:
        import ollama

        if "nomic" in model:  # nomic-embed-text is trained with these task prefixes
            prefix = "search_query: " if query else "search_document: "
            texts = [prefix + t for t in texts]
        return ollama.embed(model=model, input=texts, keep_alive=keep_alive)["embeddings"]

    return embed


def fake_embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Offline stand-in: the words `prada.search.terms` keeps, hashed into `FAKE_DIM`
    buckets. Texts that share words are close; it knows no synonyms."""
    out = []
    for text in texts:
        vec = [0.0] * FAKE_DIM
        for term in terms(text):
            vec[zlib.crc32(term.encode()) % FAKE_DIM] += 1.0
        out.append(vec)
    return out


def embedder(keep_alive=None) -> Embedder | None:
    """The configured embedder: the fake one under PRADA_FAKE_LLM, None when `[memory]
    embed_model` is "" (keyword search only)."""
    settings = config.memory_settings()
    if llm.fake_enabled():
        return Embedder("fake", fake_embed, settings.min_similarity)
    if not settings.embed_model:
        return None
    return Embedder(
        settings.embed_model,
        ollama_embed(settings.embed_model, keep_alive),
        settings.min_similarity,
    )


def _missing_sql() -> str:
    """What has no vector for a model yet: every turn, real summaries and facts (the fake
    model's are never searched)."""

    def side(kind: str, table: str, key: str, text: str, where: str) -> str:
        return (
            f"SELECT '{kind}', x.{key}, {text} FROM {table} x LEFT JOIN memory_vector v "
            f"ON v.kind = '{kind}' AND v.ref = x.{key} AND v.model = :model "
            f"WHERE v.ref IS NULL{where}"
        )

    return (
        " UNION ALL ".join(
            [
                side("turn", "memory_turn", "id", "x.content", ""),
                side(
                    "summary",
                    "memory_episode",
                    "id",
                    "x.summary",
                    " AND x.summary != '' AND x.model IS NOT 'fake'",
                ),
                side(
                    "fact",
                    "memory_fact",
                    "id",
                    "x.fact || ' ' || x.quote",
                    " AND x.model IS NOT 'fake'",
                ),
            ]
        )
        + " LIMIT :limit"
    )


class VectorIndex:
    def __init__(self, db, embedder: Embedder | None):
        self.db = db
        self.embedder = embedder
        self._keys: list[tuple[str, int]] = []
        self._matrix: np.ndarray | None = None  # loaded on the first search
        self._retry_at = 0.0
        self._warned = False

    @property
    def available(self) -> bool:
        return self.embedder is not None

    def embed(self, texts: list[str], query: bool = False) -> np.ndarray:
        """Unit vectors for `texts`, one row each (raises if the model fails)."""
        cut = [t[:EMBED_CHARS] for t in texts]
        vecs = np.asarray(self.embedder.embed(cut, query), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    def missing(self, limit: int) -> list[tuple[str, int, str]]:
        """(kind, ref, text) of up to `limit` things with no vector yet."""
        if not self.available:
            return []
        rows = self.db.execute(_missing_sql(), {"model": self.embedder.model, "limit": limit})
        return [(kind, ref, text) for kind, ref, text in rows]

    def add(self, items: list[tuple[str, int, str]], vecs: np.ndarray) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO memory_vector (kind, ref, model, vec) VALUES (?, ?, ?, ?)",
                [
                    (kind, ref, self.embedder.model, vec.astype(np.float32).tobytes())
                    for (kind, ref, _), vec in zip(items, vecs, strict=True)
                ],
            )
        self._matrix = None  # read again on the next search

    def forget(self, kind: str, ref: int) -> None:
        """The text changed (a turn cut short): its vector is written again later. Call
        inside the caller's transaction."""
        self.db.execute("DELETE FROM memory_vector WHERE kind = ? AND ref = ?", (kind, ref))
        self._matrix = None

    def _load(self) -> None:
        rows = [
            (kind, ref, np.frombuffer(blob, dtype=np.float32))
            for kind, ref, blob in self.db.execute(
                "SELECT kind, ref, vec FROM memory_vector WHERE model = ?", (self.embedder.model,)
            )
        ]
        sizes = Counter(vec.size for *_, vec in rows)
        size = sizes.most_common(1)[0][0] if sizes else 0
        # another model under the same name wrote the odd ones out
        keys = [(kind, ref) for kind, ref, vec in rows if vec.size == size]
        vecs = [vec for *_, vec in rows if vec.size == size]
        self._keys = keys
        self._matrix = np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)

    def search(self, query: str, limit: int) -> list[tuple[str, int, float]]:
        """(kind, ref, similarity) of the `limit` closest in meaning to `query`, closest
        first, only those at `min_similarity` or above. [] if the model can't be reached."""
        if not self.available or not query.strip() or time.monotonic() < self._retry_at:
            return []
        if self._matrix is None:
            self._load()
        if not self._keys:
            return []
        try:
            q = self.embed([query], query=True)[0]
        except Exception as e:  # Ollama down or the model not pulled: keywords still work
            self._retry_at = time.monotonic() + RETRY
            if not self._warned:
                print(f"memory: searching by meaning is off for now ({e})", file=sys.stderr)
                self._warned = True
            return []
        if q.size != self._matrix.shape[1]:
            return []
        sims = self._matrix @ q
        best = np.argsort(-sims)[:limit]
        floor = self.embedder.min_similarity
        return [(*self._keys[i], float(sims[i])) for i in best if sims[i] >= floor]
