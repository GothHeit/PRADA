"""The Corpus side of "play X": keeping PRADA's copy of the catalog fresh, and turning a
name, a genre or a mood into the songs Corpus should play.

`Library` owns the catalog: it opens it, reloads it when `prada music import` wrote a newer
one, and reads Corpus' export again when it's older than `REFRESH` (in the background, so a
request never waits for it). `find(catalog, query, kind, mood)` picks the songs: a clear
song, album or artist; a genre ("jazz"); a mood ("calm", from genre words, since Corpus has
no mood tags); a question when two kinds of thing share the name; or the line she says when
it isn't there. Reading the request itself ("play X") is the parser's job, not this one's.
"""

import random
import re
import sys
import threading
import time
from dataclasses import dataclass

from prada.music.catalog import MIGRATIONS, STORE, TIE, Catalog, Hit, Track, key, norm, parse_export
from prada.music.corpus import CorpusError, FakeCorpus
from prada.store import db_path, open_store

REFRESH = 6 * 3600.0  # seconds before the catalog is read from Corpus again
RETRY = 600.0  # seconds before trying again after Corpus couldn't export

# Genre words that fit a mood; Corpus has no mood tags, so genre is the only clue.
MOODS = {
    "calm": ["ambient", "classical", "jazz", "acoustic", "piano", "lofi", "lo fi", "chill",
             "new age", "soundtrack", "folk", "bossa nova", "instrumental", "easy listening",
             "downtempo", "classica", "mpb"],
    "energetic": ["rock", "metal", "punk", "electronic", "dance", "edm", "techno", "house",
                  "drum and bass", "hardcore", "alternative"],
    "focus": ["classical", "ambient", "instrumental", "soundtrack", "post rock", "lofi", "lo fi",
              "piano", "classica"],
    "happy": ["pop", "j pop", "k pop", "funk", "disco", "dance", "ska", "reggae", "city pop"],
    "sad": ["blues", "ballad", "emo", "shoegaze", "slowcore", "acoustic", "singer songwriter"],
}  # fmt: skip
_MOOD_WORDS = {
    "calm": "calm", "calming": "calm", "relaxing": "calm", "relax": "calm", "chill": "calm",
    "soft": "calm", "quiet": "calm", "peaceful": "calm", "mellow": "calm", "calma": "calm",
    "calmo": "calm", "calmas": "calm", "calmos": "calm", "tranquila": "calm",
    "tranquilo": "calm", "relaxante": "calm", "relaxar": "calm", "suave": "calm",
    "energetic": "energetic", "upbeat": "energetic", "heavy": "energetic", "loud": "energetic",
    "agitada": "energetic", "agitado": "energetic", "animada": "energetic",
    "animado": "energetic", "pesada": "energetic", "pesado": "energetic",
    "focus": "focus", "study": "focus", "studying": "focus", "concentrate": "focus",
    "work": "focus", "working": "focus", "estudar": "focus", "concentrar": "focus",
    "trabalhar": "focus", "foco": "focus",
    "happy": "happy", "cheerful": "happy", "fun": "happy", "alegre": "happy", "feliz": "happy",
    "sad": "sad", "melancholic": "sad", "triste": "sad", "melancolica": "sad",
}  # fmt: skip
SAMPLE = 50  # songs handed to Corpus for a mood, a genre or "some music"
_FILLER = set(
    "something some anything a an bit little of to for with the music songs song tunes "
    "algo alguma coisa uma um umas uns musica musicas som sons pra para de do da bem mais "
    "meio pouco it this that isso again".split()
)


def mood_of(text: str) -> str | None:
    """The mood in "calm", "pra relaxar", "something to study to"; None if it isn't one."""
    words = [w for w in norm(text).split() if w not in _FILLER]
    if words and all(w in _MOOD_WORDS for w in words):
        return _MOOD_WORDS[words[0]]
    return None


class Library:
    """The Corpus catalog for one process: `corpus` is a `CorpusCli` or a `FakeCorpus`."""

    def __init__(self, corpus, path=None):
        """`path` = the database (default: the node's; ":memory:" for a fake Corpus)."""
        self.corpus = corpus
        self.path = path
        self._catalog: Catalog | None = None
        self._lock = threading.Lock()
        self._reading = False
        self._tried = 0.0

    @property
    def fake(self) -> bool:
        return isinstance(self.corpus, FakeCorpus)

    def open(self) -> Catalog:
        with self._lock:
            if self._catalog is None:
                path = self.path or (":memory:" if self.fake else db_path())
                self._catalog = Catalog(open_store(STORE, MIGRATIONS, path))
            return self._catalog

    def catalog(self) -> Catalog:
        """The catalog as it is now; a stale one starts a refresh (waited for only from a
        fake Corpus; never tried while Corpus isn't set up: `prada music import` says why)."""
        catalog, now = self.open(), time.time()
        catalog.sync()
        stale = catalog.imported_at is None or now - catalog.imported_at > REFRESH
        if stale and not self._reading and now - self._tried > RETRY and self.corpus.available():
            self.refresh(wait=self.fake)
        return catalog

    def refresh(self, wait: bool = True) -> int:
        """Reads Corpus' export into the catalog; returns how many songs it has (-1 when it
        runs in the background). Raises `CorpusError` only when waited for."""
        self._reading, self._tried = True, time.time()

        def read() -> int:
            try:
                tracks = parse_export(self.corpus.export())
                catalog = self.open()
                if not tracks and len(catalog):
                    raise CorpusError(
                        f"Corpus exported no songs; I kept the {len(catalog)} I read before"
                    )
                catalog.replace(tracks, time.time())
                return len(tracks)
            finally:
                self._reading = False

        if wait:
            return read()

        def background() -> None:
            try:
                read()
            except (CorpusError, OSError) as e:
                print(f"music: Corpus' catalog not read ({e})", file=sys.stderr)

        threading.Thread(target=background, daemon=True).start()
        return -1


@dataclass
class Found:
    label: str  # what she says she plays: "Space Dementia by Muse", "something calm"
    kind: str  # "song", "album", "artist", "genre" or "mood"
    tracks: list[Track]
    shuffle: bool


@dataclass
class Choice:
    """Different things answer to the same name: she asks which one."""

    name: str
    options: dict[str, Found]  # by kind

    def question(self) -> str:
        names = {"song": "the song", "album": "the album", "artist": "everything by the artist"}
        return f"{self.name}: {' or '.join(names[k] for k in self.options)}?"

    def answer(self, text: str) -> Found | None:
        """The option an answer names ("the album", "a música"), else None."""
        t = norm(text)
        return next((self.options[k] for k in self.options if _ANSWERS[k].fullmatch(t)), None)


_LEAD = r"(?:(?:hey |ok |oi )?prada |please |por favor )*"
_ANSWER = {
    "song": r"(?:the |a |just the |so a )?(?:song|track|musica|faixa)(?: so)?",
    "album": r"(?:the |o )?(?:whole )?(?:album|record|disco)(?: todo| inteiro)?",
    "artist": r"(?:the |o |a )?(?:artist|band|artista|banda)|everything|tudo",
}
_ANSWERS = {k: re.compile(rf"{_LEAD}(?:{p})(?: please| por favor)?") for k, p in _ANSWER.items()}


def _found(hit: Hit, shuffle: bool | None) -> Found:
    if hit.kind == "song":
        return Found(hit.describe(), "song", hit.tracks[:1], False)
    if hit.kind == "album":
        tracks = sorted(hit.tracks, key=lambda t: t.path)  # files are numbered in album order
        return Found(hit.describe(), "album", tracks, bool(shuffle))
    return Found(hit.name, "artist", hit.tracks, shuffle is not False)


def find(
    catalog: Catalog,
    query: str,
    kind: str | None = None,
    mood: str | None = None,
    shuffle: bool | None = None,
    rng: random.Random | None = None,
) -> Found | Choice | str:
    """What Corpus plays for `query`: songs, a question, or the line she says when it isn't
    there. `kind` = "song", "album" or "artist" when the words said so; `mood` = a key of
    `MOODS`, or "any" for "play some music"; `shuffle` None = the default for the kind (a
    song or album in order, an artist, genre or mood shuffled)."""
    rng = rng or random.Random()
    shuffled = shuffle is not False

    def sample(tracks: list[Track]) -> list[Track]:
        return rng.sample(tracks, min(SAMPLE, len(tracks)))

    if mood == "any":
        if not len(catalog):
            return "Your Corpus library is empty, or I haven't read it yet."
        return Found("some music", "mood", sample(catalog.tracks()), shuffled)
    if mood:
        tracks = catalog.by_genre(MOODS[mood])
        if not tracks:
            return f"None of your Corpus songs has a genre that sounds {mood}."
        return Found(f"something {mood}", "mood", sample(tracks), shuffled)
    hits = catalog.find(query, kind=kind)
    genre = " ".join(w for w in key(query).split() if w not in _FILLER)
    if kind is None and genre and not (hits and hits[0].score == 1):
        if tracks := catalog.by_genre([genre]):  # "some jazz", "um rock"
            return Found(f"some {genre}", "genre", sample(tracks), shuffled)
    if not hits:
        return f"I couldn't find {query} in your Corpus library."
    best, kinds = hits[0], {}
    for h in hits:
        if best.score - h.score <= TIE:
            kinds.setdefault(h.kind, h)  # the best hit of each kind that ties
    if best.kind == "album" and len(best.tracks) == 1 and "song" in kinds:
        best = kinds["song"]  # a single: the album is just the song
    rivals = {
        k: h
        for k, h in kinds.items()
        if k != best.kind and not (k == "album" and len(h.tracks) == 1)
    }
    if rivals and kind is None:
        options = {best.kind: _found(best, shuffle)}
        options.update({k: _found(h, shuffle) for k, h in rivals.items()})
        return Choice(best.name, options)
    return _found(best, shuffle)
