"""'Play X': reading the wish from her words and picking the best Spotify result, in code.

`wish(text)` turns "play Muse Space Dementia", "toca Evidências do Chitãozinho e Xororó",
"play the album Origin of Symmetry" or "toca algo calmo" into a `Wish` without the model.
Queue and shuffle words ("add X to the queue", "play X next", "on shuffle") are read too,
for Corpus. `pick(wish, results)` chooses what Spotify plays: a result that covers everything kafka said
wins (Spotify's own top result first), else the closest track; nothing close = None, so
she never plays something random.
"""

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from prada.music.spotify import Hit, Results

KINDS = ("track", "artist", "album", "playlist", "mood")


@dataclass
class Wish:
    query: str  # what to search for, folded (see `fold`)
    kind: str = "any"  # any, track, artist, album, playlist, mood
    app: str | None = None  # "spotify" or "corpus" when named
    mode: str = "play"  # "play", "queue" (end of the queue) or "next" (after this song)
    shuffle: bool | None = None  # "on shuffle" / "in order"; None = not said
    queued: bool = False  # the verb was "add ... to the queue" / "shuffle": not only music


def fold(text: str) -> str:
    """Lower case, Latin accents dropped ("Evidências" -> "evidencias"), punctuation to
    spaces. Other scripts (Japanese, Cyrillic) are kept as they are."""
    out = []
    for c in unicodedata.normalize("NFC", text.lower()):
        base = unicodedata.normalize("NFD", c)[0]
        out.append(base if base.isascii() else c)
    text = re.sub(r"['’]", "", "".join(out))
    return re.sub(r"[^\w]+", " ", text).strip()


_STOP = {"the", "a", "an", "by", "of", "and", "do", "da", "dos", "das", "de", "o", "e", "feat"}


def tokens(text: str) -> set[str]:
    return {t for t in fold(text).split() if t not in _STOP} or set(fold(text).split())


# Fillers before the request: "so Prada, can you ...", "e aí Prada, você pode ...".
FILLER = (
    r"so |and |then |ok |okay |alright |well |hey |hi |now |e |entao |ai |oi |ola |agora "
    r"|bom |tipo |hm+ |uh+ "
)
POLITE = (
    r"please |por favor |(?:can|could|would|will) you (?:(?:please|maybe|just) )?"
    r"|would you mind |do you mind |(?:voce )?(?:pode|poderia|consegue|podia) (?:(?:por favor|so) )?"
)
_LEAD = (
    rf"(?:{FILLER}|prada |{POLITE}|(?:eu )?quero |i want (?:you )?to |id like (?:you )?to "
    r"|i wanna |lets |vamos |can i |can we |could i |could we |da pra |da para )*"
)
_TRAIL = r"(?: (?:please|por favor|prada|for me|pra mim|now|agora|ai|then|entao))*"
# "play"/"toca" mean music on their own; the others only with a sign of music (`_marked`).
_VERB = (
    r"(?:(?P<plain>play|(?<=mind )playing|toca|tocar|toque)|playing|put on|putting on|coloca|colocar|coloque"
    r"|bota|botar|poe|ouvir|escutar|listen to|hear|let me hear|let us hear)"
    r"(?P<to_play> (?:pra|para) tocar)?(?: me| pra mim| para mim| pra gente| for me| for us)?"
)
_ON = r"(?: (?:on|in|from|no|na|do|pelo) (?P<app>spotify|corpus))?"
_PLAY = re.compile(rf"(?P<lead>{_LEAD}){_VERB} (?P<body>.+?){_ON}{_TRAIL}")
# Leads about kafka doing it ("i wanna play Minecraft", "can I play ..."): a game as often
# as music, so they never skip the router by themselves.
_SELF = re.compile(r"\b(?:i wanna|can i|could i|can we|could we)\b")
# "play" followed by these isn't music.
_NOT_MUSIC = re.compile(
    r"(?:(?:a |the |some )?(?:game|games|jogo|jogos|chess|xadrez|cards|cartas|dead|along|around"
    r"|pretend|cello|violoncelo|guitar|violao|piano|with me|comigo|a joke|a prank|it safe|fair"
    r"|a role|hide and seek|back (?:what|that|it|my|the recording)|(?:the |o )?(?:sino|bell"
    r"|campainha|alarm|alarme|buzina|horn)|(?:dumb|nice|it again|that again|again|isso de novo"
    r"|de novo|it back|that back)$)\b)"
)
_MOOD = re.compile(
    r"(?:something|anything|some music|some songs|music|songs|algo|alguma coisa|uma musica"
    r"|alguma musica|musica|musicas|um som) (?:(?:that s|thats|that is|que seja|que e|meio|bem"
    r"|mais|more|really|bem) )?(?P<mood>.+)"
)
_KINDED = [
    ("album", re.compile(r"(?:the |o |a )?(?:album|disco) (?:(?:called|chamado) )?(?P<q>.+)")),
    ("album", re.compile(r"(?:the |o )?(?P<q>.+) (?:album|disco)")),
    ("playlist", re.compile(r"(?:the |a |my |minha )?playlist (?:(?:called|chamada) )?(?P<q>.+)")),
    ("playlist", re.compile(r"(?:the |a |my |minha )?(?P<q>.+) playlist")),
    (
        "artist",
        re.compile(
            r"(?:something by|songs by|music by|anything by"
            r"|(?:uma |alguma )?musicas? (?:do|da|de|dos|das)"
            r"|algo (?:do|da|de|dos|das)|alguma coisa (?:do|da|de|dos|das)) (?P<q>.+)"
        ),
    ),
    ("any", re.compile(r"(?:some|um pouco (?:do|da|de|dos|das)) (?P<q>.+)")),
    (
        "track",
        re.compile(
            r"(?:(?:the|a) (?:song|track|musica|faixa)(?: (?:called|chamada))?"
            r"|(?:song|track|musica|faixa) (?:called|chamada)) (?P<q>.+)"
        ),
    ),
]
_BY = re.compile(r"(?P<title>.+) (?:by|do|da|dos|das) (?P<by>.+)")
# Words that only mean "music", never a mood or a name.
_EMPTY = {
    *("it", "music", "some music", "songs", "some songs", "a song", "something", "anything"),
    *("me", "us", "you", "mim", "you guys", "isso", "this", "that"),
    *("musica", "uma musica", "alguma musica", "musicas", "som", "um som", "algo", "alguma coisa"),
}


_POLITE = re.compile(r" (?:please|por favor|prada|for me|pra mim|now|agora)$")
_SHUFFLE = re.compile(
    r" (?P<s>on shuffle|shuffled|in shuffle|at random|in random order|no aleatorio|aleatorio"
    r"|em modo aleatorio|embaralhado|embaralhada|in order|em ordem|na ordem)$"
)
_IN_ORDER = ("in order", "em ordem", "na ordem")
_NEXT = re.compile(
    r" (?:next|after this(?: one| song)?|depois dessa|depois desta|em seguida|na sequencia)$"
)
_QUEUE = re.compile(
    r"(?P<lead>.*?)(?:add|adiciona|adicionar|coloca|bota|poe) (?P<body>.+?)"
    r" (?:(?:to|on|in) (?:the|my) queue|(?:na|a|pra|para a) fila)"
)
_QUEUE_VERB = re.compile(r"(?P<lead>.*?)queue(?: up)? (?P<body>.+)")
_SHUFFLE_VERB = re.compile(r"(?P<lead>.*?)(?:shuffle|embaralha) (?P<body>.+)")


def _extras(t: str) -> tuple[str, str, bool | None, bool]:
    """Peels the queue and shuffle words off a folded request: ("play x", mode, shuffle,
    whether the verb was rewritten to "play")."""
    mode, shuffle, rewritten = "play", None, False
    while True:
        if found := _POLITE.search(t):
            pass
        elif found := _SHUFFLE.search(t):
            shuffle = found["s"] not in _IN_ORDER
        elif mode == "play" and (found := _NEXT.search(t)):
            mode = "next"
        else:
            break
        t = t[: found.start()]
    if mode == "play" and (m := _QUEUE.fullmatch(t) or _QUEUE_VERB.fullmatch(t)):
        t, mode, rewritten = f"{m['lead']}play {m['body']}", "queue", True
    elif m := _SHUFFLE_VERB.fullmatch(t):
        t, shuffle, rewritten = f"{m['lead']}play {m['body']}", True, True
    return t, mode, shuffle, rewritten


def wish(text: str, strict: bool = False, peel: bool = True) -> Wish | None:
    """What kafka asked to play, or None if `text` isn't a plain "play <something>".
    `strict` (before the router): a verb that isn't only about music ("coloca", "put on",
    "ouvir") counts only with a sign of music: a kind ("o disco", "playlist", "algo calmo"),
    "pra tocar" or "on Spotify" ("X do Y" alone is too common: "a foto do gato"). "Add X
    to the queue" and "shuffle X" never count by themselves ("shuffle the deck", "coloca o
    carro na fila"): the module claims those only when Corpus has X. `peel=False` keeps
    trailing words such as "next" in the name ("play Who's Next")."""
    if peel:
        t, mode, shuffle, rewritten = _extras(fold(text))
    else:
        t, mode, shuffle, rewritten = fold(text), "play", None, False
    m = _PLAY.fullmatch(t)
    if not m:
        return None
    body, app = m["body"].strip(), m["app"]
    if _NOT_MUSIC.match(body) or body in _EMPTY or body in KINDS:
        return None
    found = _read(body, app)
    found.mode, found.shuffle, found.queued = mode, shuffle, rewritten
    plain = m["plain"] and not rewritten and not _SELF.search(m["lead"])
    marked = plain or m["to_play"] or app or found.kind != "any"
    return found if marked or not strict else None


def _read(body: str, app: str | None) -> Wish:
    for kind, pattern in _KINDED:
        if k := pattern.fullmatch(body):
            return Wish(k["q"], kind, app)
    if mood := _MOOD.fullmatch(body):
        return Wish(mood["mood"], "mood", app)
    if by := _BY.fullmatch(body):
        return Wish(f"{by['title']} {by['by']}", "any", app)
    return Wish(body, "any", app)


def _match(a: str, b: str) -> bool:
    return a == b or (len(a) > 3 and SequenceMatcher(None, a, b).ratio() >= 0.8)


def _share(have: set[str], within: set[str]) -> float:
    """The share of `have` found in `within` (close spellings count: whisper slips)."""
    if not have:
        return 0.0
    return sum(any(_match(t, w) for w in within) for t in have) / len(have)


ORDER = {"track": 0, "artist": 1, "album": 2, "playlist": 3}
TITLE = 0.7  # how close the rest of the words must be to a title by an artist named exactly


def _exact(have: set[str], within: set[str]) -> float:
    return len(have & within) / len(have) if have else 0.0


def named_artists(wish: Wish, results: Results) -> list[str]:
    """Artists in the results whose whole name kafka said word for word ("Muse" in "muse
    space dimension"), longest first, when other words were said too (a title)."""
    said = tokens(wish.query)
    names = {
        h.name if h.kind == "artist" else a.strip()
        for h in results.hits
        for a in ([h.name] if h.kind == "artist" else h.by.split(","))
        if h.kind in ("artist", "track", "album")
    }
    found = [n for n in names if (t := tokens(n)) and t <= said and t != said]
    return sorted(found, key=lambda n: -len(tokens(n)))


def _by_artist(wish: Wish, results: Results) -> Hit | None:
    """A track (or album) by an artist kafka named exactly whose title is close to the rest
    of the words: "muse space dimension" -> Space Dementia by Muse (a whisper slip)."""
    words = fold(wish.query).split()
    best, best_score = None, TITLE
    for artist in named_artists(wish, results):
        own = tokens(artist)
        rest = " ".join(w for w in words if w not in own)
        if not rest:
            continue
        for h in results.hits:
            if h.kind not in ("track", "album") or not any(
                tokens(a) == own for a in h.by.split(",")
            ):
                continue
            if wish.kind != "any" and h.kind != wish.kind:
                continue
            score = SequenceMatcher(None, rest, fold(h.name)).ratio()
            if score > best_score:
                best, best_score = h, score
    return best


def pick(wish: Wish, results: Results) -> Hit | None:
    """The result to play, or None when nothing is close enough to what kafka said.
    A result that covers everything said wins. Otherwise an artist named word for word
    comes first, with the closest title of theirs; a near match that needs a slip in
    most of the words ("Dimensions of Time" by "Space Museum" for "muse space dimension")
    is never played."""
    hits = results.hits
    if wish.kind == "mood":
        return next((h for h in hits if h.kind == "playlist"), None)
    if wish.kind != "any":
        hits = [h for h in hits if h.kind == wish.kind]
    said = tokens(wish.query)

    def words(h: Hit) -> set[str]:
        return tokens(f"{h.name} {h.by}") if h.kind != "artist" else tokens(h.name)

    def said_of(h: Hit) -> float:  # how much of what kafka said this result explains
        return _share(said, words(h))

    def named(h: Hit) -> float:  # how much of the result's name kafka said
        return _share(tokens(h.name), said)

    full = [h for h in hits if said_of(h) == 1 and named(h) == 1]
    if full:
        top = next((h for h in full if h.uri == results.top), None)
        if top:
            return top
        exact = [h for h in full if tokens(h.name) == said]  # just a name: artist first
        if exact:
            return min(exact, key=lambda h: (h.kind != "artist", ORDER[h.kind]))
        return min(full, key=lambda h: ORDER[h.kind])
    if found := _by_artist(wish, results):
        return found
    close = [
        h for h in hits if said_of(h) >= 0.5 and named(h) >= 0.5 and _exact(said, words(h)) >= 0.5
    ]
    if not close:
        return None
    return max(close, key=lambda h: (said_of(h) + named(h), h.kind == "track"))
