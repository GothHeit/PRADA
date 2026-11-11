"""The music module: "play X" on Spotify, and pause, resume, skip and "what's playing?" for
Spotify and Corpus.

Plain commands ("pause", "next song", "próxima", "que música é essa?") are matched in
code, so they skip the router and never wait for the model. Each button press is a typed
SAFE action: the node presses it through Windows' media controls and checks the player
really changed (paused, playing, a new track) before she says so. "What's playing?" is
answered from the players only, never guessed.

"Play Muse Space Dementia" / "toca Evidências" is read in code too (`prada.music.play`) and
becomes one SAFE `music.play` action: the node searches Spotify through its Web API
(`prada.music.spotify`), picks the closest result, starts it on one of kafka's devices and
checks Spotify really plays it. No model call, so it starts in about a second.

Corpus comes first when the music is clearly in kafka's own files: the request is looked up
in PRADA's copy of the Corpus catalog (`prada.music.library`) and becomes a SAFE
`music.play_corpus` action, which hands the files to Corpus' command line and counts once
Windows shows Corpus playing one of them. Corpus also takes "add X to the queue", "play X
next", genres ("toca um rock") and moods ("something calm", by genre). Naming an app
("on Spotify", "no Corpus") decides it; a playlist is always Spotify.

While the node runs (`start`), a watcher logs every track played to `music_play`.
"""

import re
import sys
import time
import unicodedata
from typing import ClassVar, Literal

from prada import llm, vocabulary
from prada import situation as seen
from prada.actions import Action, ActionFailed, Executor
from prada.module import Context, Module, Reply, Request
from prada.music import library
from prada.music.bridge import BridgeError, FakeBridge, Player, Status, WindowsBridge
from prada.music.catalog import key
from prada.music.corpus import CorpusCli, CorpusError, FakeCorpus
from prada.music.play import FILLER, POLITE, Wish, fold, named_artists, pick, tokens, wish
from prada.music.playlog import MIGRATIONS, POLL, PlayLog, Watcher, favorites
from prada.music.spotify import (
    FakeSpotify,
    Hit,
    Now,
    Results,
    SpotifyApi,
    SpotifyError,
    choose,
    open_app,
)
from prada.store import open_store

VERIFY = 3.0  # seconds to wait for the player to show the change
VERIFY_STEP = 0.15
LAUNCH = 20.0  # seconds Spotify may take to open and show up as a device
SPOTIFY_STEP = 0.5  # seconds between looks at Spotify (each is a Web API call)
ARTIST_LIMIT = 10  # an artist's songs searched when a title by them was misheard
SPOTIFY_APP = "Spotify.exe"  # Spotify's app id in Windows' media controls
FAVORITES = 25  # most played tracks whose names go to her ears
FAVORITE_DAYS = 90  # ... over this many days
EAR_WORDS = 80  # music names offered to her ears at most
FRESH = 3 * POLL  # seconds the watcher's last look at the players stays good for her
PLAY_VERIFY = 12.0  # seconds for Corpus to show a song it was asked to play (it may start)

# Matched on lower-case text without accents or punctuation (see `_norm`).
_OBJ = r"(?:(?:the|this|that|a|essa|esta|esse|o) )?(?:music|song|track|tune|musica|faixa|som)"
_APP = r"(?P<app>spotify|corpus)"
_ON = r"(?: (?:on|in|no|na|do|da) (?P<on>spotify|corpus))?"
_WHAT = rf"(?: {_OBJ}| {_APP}| it| ai| isso)?{_ON}"
_COMMANDS = {
    "pause": [
        rf"(?:pause|pausa|pausar|pausa ai){_WHAT}",
        rf"(?:stop|para|parar|stop playing) (?:{_OBJ}|{_APP}){_ON}",
        r"stop playing",
    ],
    "resume": [
        rf"(?:resume|unpause|play|despausa|despausar|retoma|retomar){_WHAT}",
        rf"(?:continue|continua|continuar|volta|voltar|toca|tocar|solta) (?:{_OBJ}|{_APP}){_ON}",
        rf"(?:keep playing|continue playing|volta a tocar|continua tocando|pode tocar){_ON}",
    ],
    "next": [
        rf"(?:next|skip|pula|pular|proxima|avanca|passa){_WHAT}",
        rf"(?:next|proxima|another|outra) (?:{_OBJ}|one){_ON}",
        rf"(?:skip|pula|passa) (?:this|that|essa|esta) (?:one|ai){_ON}",
        rf"(?:play the |toca a )?(?:next|proxima)(?: {_OBJ})?{_ON}",
    ],
    "previous": [
        rf"(?:previous|anterior){_WHAT}",
        rf"(?:play the |toca a |go back to the |volta (?:pra|para) a )?(?:previous|last) (?:{_OBJ}|one){_ON}",
        rf"{_OBJ} anterior{_ON}",
        rf"go back (?:a|one) (?:song|track){_ON}",
        rf"(?:volta|voltar) (?:a|uma) {_OBJ}{_ON}",
        rf"(?:volta|voltar) (?:pra|para) (?:a )?anterior{_ON}",
    ],
    "now": [
        rf"(?:whats|what is|what s) (?:playing|this {_OBJ}|the {_OBJ}|on){_ON}(?: now| right now)?",
        rf"(?:whats|what is) on {_APP}(?: now| right now)?",
        rf"what {_OBJ} is (?:this|that|playing)(?: now| right now)?",
        r"what am i listening to(?: now| right now)?",
        rf"(?:whats|what is) the name of (?:this|the) {_OBJ}",
        r"who (?:sings|is singing|is this)(?: this| that)?(?: song)?",
        rf"(?:que|qual) (?:e )?(?:a )?{_OBJ} (?:que )?(?:e |esta |ta )?(?:essa|esta|tocando)(?: agora)?",
        r"o que (?:esta|ta) tocando(?: agora)?" + _ON,
        rf"qual (?:e )?o nome (?:dessa|desta|da) {_OBJ}",
        r"quem (?:canta|ta cantando|esta cantando)(?: essa| isso| esta)?(?: musica)?",
    ],
}
_LEAD = rf"(?:{FILLER}|prada |{POLITE})*"
_TRAIL = r"(?: (?:please|por favor|prada|for me|pra mim|now|agora|ai))*"
_MATCHERS = [
    (verb, re.compile(rf"{_LEAD}(?:{pattern}){_TRAIL}"))
    for verb, patterns in _COMMANDS.items()
    for pattern in patterns
]


def _bare_title(title: str) -> str:
    """A title without its version notes: "Space Dementia - Remastered (2011)" -> "Space
    Dementia"."""
    bare = re.sub(r"\s*[(\[][^)\]]*[)\]]|\s+-\s.*$", "", title).strip()
    return bare or title


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"['’]", "", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def command(text: str) -> tuple[str, str | None] | None:
    """("pause"|"resume"|"next"|"previous"|"now", "spotify"|"corpus"|None) for a plain
    music command in English or Portuguese, else None. "play Muse" is not one (yet)."""
    t = _norm(text)
    for verb, matcher in _MATCHERS:
        if m := matcher.fullmatch(t):
            names = m.groupdict()
            return verb, names.get("app") or names.get("on")
    return None


class _Press(Action):
    app: str  # Windows' app id of the player
    player: str  # its name, "Spotify"
    doing: ClassVar[str] = "use"

    def describe(self) -> str:
        return f"{self.doing} {self.player}"


class Pause(_Press):
    kind: Literal["music.pause"] = "music.pause"
    doing: ClassVar[str] = "pause"


class Resume(_Press):
    kind: Literal["music.resume"] = "music.resume"
    doing: ClassVar[str] = "resume"


class Next(_Press):
    kind: Literal["music.next"] = "music.next"
    doing: ClassVar[str] = "skip to the next track on"


class Previous(_Press):
    kind: Literal["music.previous"] = "music.previous"
    doing: ClassVar[str] = "go back a track on"


def _settled(player: Player) -> bool:
    """Mid-skip, players briefly show no title or the state "changing"; that's no answer."""
    return bool(player.title) and player.state != "changing"


class _PressExecutor(Executor):
    """Presses one button, then watches the player until it shows the change."""

    verb = ""

    def __init__(self, module: "MusicModule"):
        self._module = module

    def changed(self, before: Player, after: Player) -> bool:
        raise NotImplementedError

    def run(self, action: _Press, ctx: Context) -> dict:
        bridge = self._module.bridge()
        try:
            before = bridge.status().get(action.app)
            if before is None:
                raise ActionFailed(f"{action.player} isn't open")
            if not bridge.control(action.app, self.verb):
                raise ActionFailed(f"{action.player} didn't take the command")
            self._module.pressed(action.app)
            after = before
            deadline = time.monotonic() + self._module.verify
            while True:
                after = bridge.status().get(action.app) or after
                if self.changed(before, after) or time.monotonic() >= deadline:
                    break
                time.sleep(VERIFY_STEP)
        except BridgeError as e:
            raise ActionFailed(str(e)) from e
        return {"player": action.player, "before": vars(before), "after": vars(after)}

    def check(self, action: _Press, result: dict, ctx: Context) -> None:
        before, after = Player(**result["before"]), Player(**result["after"])
        if not self.changed(before, after):
            raise ActionFailed(f"{action.player} didn't change")


class PauseExecutor(_PressExecutor):
    action = Pause
    verb = "pause"

    def changed(self, before: Player, after: Player) -> bool:
        return _settled(after) and not after.playing

    def done(self, action: Pause, result: dict) -> str:
        return f"Paused {action.player}."


class ResumeExecutor(_PressExecutor):
    action = Resume
    verb = "play"

    def changed(self, before: Player, after: Player) -> bool:
        return after.playing

    def done(self, action: Resume, result: dict) -> str:
        return f"Playing again: {Player(**result['after']).describe()}."


class NextExecutor(_PressExecutor):
    action = Next
    verb = "next"

    def changed(self, before: Player, after: Player) -> bool:
        return _settled(after) and after.track != before.track

    def done(self, action: Next, result: dict) -> str:
        return f"Skipped. Now it's {Player(**result['after']).describe()}."


class PreviousExecutor(_PressExecutor):
    action = Previous
    verb = "previous"

    def changed(self, before: Player, after: Player) -> bool:
        # Most players restart the track when it's a few seconds in, instead of going back.
        restarted = before.position > 3 and after.position < before.position - 2
        return _settled(after) and (after.track != before.track or restarted)

    def done(self, action: Previous, result: dict) -> str:
        before, after = Player(**result["before"]), Player(**result["after"])
        if after.track == before.track:
            return f"Back to the start of {after.describe()}."
        return f"Went back to {after.describe()}."


class Play(Action):
    kind: Literal["music.play"] = "music.play"
    query: str  # what kafka asked for, folded
    want: str = "any"  # any, track, artist, album, playlist, mood

    def describe(self) -> str:
        return f"play {self.query} on Spotify"


def _plays(hit: Hit, now: Now) -> bool:
    """Spotify plays `hit`: the same uri or context, or another release of the same track
    (same title and a shared artist)."""
    same = now.uri == hit.uri or now.context == hit.uri
    relinked = (
        hit.kind == "track"
        and fold(now.title) == fold(hit.name)
        and bool(tokens(now.artists) & tokens(hit.by))
    )
    return now.playing and (same or relinked)


def find(api, asked: Wish) -> Hit:
    """The Spotify result to play. When nothing is close but kafka named an artist word
    for word, their own songs are searched too ("muse space dimension": whisper heard
    Space Dementia wrong, and Spotify's first answer was another band's song)."""
    results = api.search(asked.query)
    hit = pick(asked, results)
    artists = named_artists(asked, results) if hit is None else []
    for artist in artists[:1]:
        more = api.search(f"artist:{artist}", limit=ARTIST_LIMIT)
        hit = pick(asked, Results(results.hits + more.hits, results.top))
        if hit is None:
            rest = " ".join(w for w in fold(asked.query).split() if w not in tokens(artist))
            raise ActionFailed(f'I couldn\'t find "{rest}" by {artist} on Spotify')
    if hit is None:
        raise ActionFailed(f'Spotify has nothing close to "{asked.query}"')
    return hit


class PlayExecutor(Executor):
    """Searches Spotify, picks the closest result, starts it on a device, then reads back
    what plays. Opens Spotify first when none of kafka's devices is open."""

    action = Play

    def __init__(self, module: "MusicModule"):
        self._module = module

    def run(self, action: Play, ctx: Context) -> dict:
        api = self._module.spotify()
        if not api.ready():
            raise ActionFailed("I'm not logged in to Spotify yet; run `prada music login`")
        try:
            hit = find(api, Wish(action.query, action.want))
            device = choose(api.devices())
            if device is None:
                api.open_app()
                deadline = time.monotonic() + self._module.launch
                devices = []
                while device is None and time.monotonic() < deadline:
                    time.sleep(self._module.step)
                    devices = api.devices()
                    device = choose(devices)
                device = device or choose(devices, anywhere=True)
                if device is None:
                    raise ActionFailed("Spotify didn't show up as a device after I opened it")
            self._module.pressed(SPOTIFY_APP)
            api.play(hit, device)
            now = api.now()
            deadline = time.monotonic() + self._module.verify
            while not _plays(hit, now) and time.monotonic() < deadline:
                time.sleep(self._module.step)
                now = api.now()
        except SpotifyError as e:
            raise ActionFailed(str(e)) from e
        paused = self._module.quiet_corpus() if _plays(hit, now) else []  # one at a time
        return {"hit": vars(hit), "now": vars(now), "paused": paused}

    def check(self, action: Play, result: dict, ctx: Context) -> None:
        hit, now = Hit(**result["hit"]), Now(**result["now"])
        if not _plays(hit, now):
            raise ActionFailed(f"Spotify didn't start {hit.describe()}")

    def done(self, action: Play, result: dict) -> str:
        hit, now = Hit(**result["hit"]), Now(**result["now"])
        where = f" on {now.device}" if now.device else ""
        line = f"Playing {hit.describe()}{where}."
        if result.get("paused"):
            line += f" I paused {' and '.join(result['paused'])}."
        return line


class PlayCorpus(Action):
    """Corpus plays these files (mode "play"), or adds them to its queue ("queue", "next")."""

    kind: Literal["music.play_corpus"] = "music.play_corpus"
    label: str  # "Space Dementia by Muse", "something calm"
    paths: list[str]
    titles: list[str]  # what Windows will show for them
    mode: str = "play"
    shuffle: bool = False

    def describe(self) -> str:
        if self.mode == "play":
            return f"play {self.label} in Corpus"
        return f"add {self.label} to Corpus' queue"


def _corpus(status: Status) -> Player | None:
    return next((p for p in status.players if p.name == "Corpus"), None)


class PlayCorpusExecutor(Executor):
    """Hands the files to Corpus, then watches Windows' media controls until Corpus plays
    one of them (a queued song only needs Corpus playing). Spotify, if playing, is paused
    once Corpus has started."""

    action = PlayCorpus

    def __init__(self, module: "MusicModule"):
        self._module = module

    def _ok(self, action: PlayCorpus, player: Player | None) -> bool:
        if player is None or not player.playing or not _settled(player):
            return False
        return action.mode != "play" or key(player.title) in {key(t) for t in action.titles}

    def run(self, action: PlayCorpus, ctx: Context) -> dict:
        bridge = self._module.bridge()
        paused = []
        try:
            before = bridge.status()
            self._module.corpus().play(action.paths, action.mode, action.shuffle)
            deadline = time.monotonic() + self._module.play_verify
            while True:
                after = _corpus(bridge.status())
                if self._ok(action, after) or time.monotonic() >= deadline:
                    break
                time.sleep(VERIFY_STEP)
            if self._ok(action, after) and action.mode == "play":
                self._module.pressed(after.app)
                pressed = [
                    p.app
                    for p in before.players
                    if p.known
                    and p.playing
                    and p.name != "Corpus"
                    and bridge.control(p.app, "pause")
                ]
                paused = self._stopped(bridge, pressed)
        except CorpusError as e:
            raise ActionFailed(str(e)) from e
        except BridgeError as e:
            raise ActionFailed(f"I couldn't check Corpus ({e})") from e
        return {"after": vars(after) if after else None, "paused": paused}

    def _stopped(self, bridge, apps: list[str]) -> list[str]:
        """The names of the players among `apps` that show they stopped playing (up to
        the module's `verify`); a pause that didn't take is never claimed."""
        deadline = time.monotonic() + self._module.verify
        while apps:
            players = [bridge.status().get(app) for app in apps]
            stopped = [p for p in players if p and _settled(p) and not p.playing]
            if len(stopped) == len(apps) or time.monotonic() >= deadline:
                return [p.name for p in stopped]
            time.sleep(VERIFY_STEP)
        return []

    def check(self, action: PlayCorpus, result: dict, ctx: Context) -> None:
        after = Player(**result["after"]) if result["after"] else None
        if not self._ok(action, after):
            if after and after.playing:
                raise ActionFailed(f"Corpus is playing {after.describe()} instead")
            raise ActionFailed("Corpus didn't start playing")

    def done(self, action: PlayCorpus, result: dict) -> str:
        after = Player(**result["after"])
        if action.mode == "queue":
            line = f"Added {action.label} to Corpus' queue."
        elif action.mode == "next":
            line = f"{action.label[0].upper()}{action.label[1:]} plays next in Corpus."
        elif len(action.paths) == 1:
            line = f"Playing {after.describe()} in Corpus."
        else:
            order = " on shuffle" if action.shuffle else ""
            line = (
                f"Playing {action.label} in Corpus{order}, {len(action.paths)} songs, "
                f"starting with {after.describe()}."
            )
        if result["paused"]:
            line += f" I paused {' and '.join(result['paused'])}."
        return line


class _RealSpotify(SpotifyApi):
    def open_app(self) -> None:
        open_app()


CHOICES = {"song": "the song", "album": "the album", "artist": "everything by the artist"}
_ACTIONS = {"pause": Pause, "resume": Resume, "next": Next, "previous": Previous}


def targets(verb: str, status: Status, app: str | None) -> list[Player] | str:
    """The players a command presses, or what she says when there's none. Spotify and
    Corpus come before other players (a browser video); pause stops everything playing."""
    players = status.players
    if app:
        players = [p for p in players if p.name.lower() == app]
        if not players:
            return f"{app.capitalize()} isn't open."
    if not players:
        return "No music player is open right now."
    rank = sorted(players, key=lambda p: (not p.known, p.app != status.current))
    playing = [p for p in rank if p.playing]
    quiet = f"{players[0].name} isn't playing right now." if app else "Nothing's playing right now."
    if verb == "pause":
        return playing or quiet
    if verb == "resume":
        if playing:
            return f"It's already playing: {playing[0].describe()}."
        paused = [p for p in rank if p.state == "paused"] or [p for p in rank if p.known]
        return paused[:1] or "There's nothing paused to resume."
    # next / previous: the one playing, else the one Windows shows, if it's a music player
    chosen = playing[:1] or [p for p in rank if p.known and p.app == status.current][:1]
    chosen = chosen or [p for p in rank if p.known][:1]
    return chosen or quiet


def now_playing(status: Status, app: str | None) -> str:
    players = [p for p in status.players if p.title and (not app or p.name.lower() == app)]
    if not players:
        return "Nothing's playing right now."
    playing = sorted(
        (p for p in players if p.playing), key=lambda p: (not p.known, p.app != status.current)
    )
    if playing:
        listed = "; and ".join(f"{p.describe()}, on {p.name}" for p in playing)
        return f"That's {listed}."
    last = next((p for p in players if p.app == status.current), players[0])
    return f"Nothing's playing right now; {last.name} is paused on {last.describe()}."


class MusicModule(Module):
    name = "music"
    description = (
        "Plays a song, artist, album, genre or mood from Corpus (kafka's own files) or "
        "Spotify, adds songs to the music queue or shuffles them, pauses, resumes and skips "
        "the music playing on the PC and says what song is playing."
    )
    migrations = MIGRATIONS
    memory_note = "[music controls]"
    ability = (
        "play songs, albums, artists, playlists or moods from Spotify or the user's own music "
        "(Corpus); queue or shuffle in Corpus; pause, resume and skip what's playing; say what "
        "it is. Nothing else: it can't change the volume, playlists, likes or the Spotify library"
    )
    examples = ["play Space Dementia by Muse", "pause", "skip", "what's playing?"]

    def __init__(
        self,
        bridge=None,
        verify: float = VERIFY,
        spotify=None,
        launch: float = LAUNCH,
        step: float = SPOTIFY_STEP,
        corpus=None,
        play_verify: float = PLAY_VERIFY,
    ):
        """`bridge` replaces Windows' media controls, `spotify` the Spotify Web API and
        `corpus` Corpus' command line (tests); `verify` = seconds to wait for a player to
        show a change, `launch` = for Spotify to show up as a device after she opens it,
        `step` = between looks, `play_verify` = for Corpus to play what it was given."""
        self._bridge = bridge
        self._spotify = spotify
        self._corpus = corpus
        self._library: library.Library | None = None
        self.play_verify = play_verify
        self.verify = verify
        self.launch = launch
        self.step = step
        self._log: PlayLog | None = None
        self._watcher: Watcher | None = None
        self.executors = [
            PauseExecutor(self),
            ResumeExecutor(self),
            NextExecutor(self),
            PreviousExecutor(self),
            PlayExecutor(self),
            PlayCorpusExecutor(self),
        ]

    def bridge(self):
        if self._bridge is None:
            self._bridge = FakeBridge() if llm.fake_enabled() else WindowsBridge()
        return self._bridge

    def spotify(self):
        if self._spotify is None:
            self._spotify = FakeSpotify() if llm.fake_enabled() else _RealSpotify()
        return self._spotify

    def corpus(self):
        if self._corpus is None:
            fake = llm.fake_enabled()
            self._corpus = FakeCorpus(bridge=self.bridge()) if fake else CorpusCli()
        return self._corpus

    def library(self) -> library.Library:
        if self._library is None:
            self._library = library.Library(self.corpus())
        return self._library

    def vocabulary(self, db) -> list[str]:
        """What you play most (any app, from the play log), then the artists with the most
        songs in Corpus' catalog (PRADA's stored copy; Corpus itself is not asked)."""
        words: list[str] = []
        if db is not None:
            for row in favorites(db, time.time() - FAVORITE_DAYS * 86400, FAVORITES):
                words += [row["artist"], _bare_title(row["title"])]
        catalog = self.library().open()
        catalog.sync()
        return vocabulary.unique(words + catalog.artists(), EAR_WORDS)

    def start(self, db_path) -> None:
        """Log what plays and read the Corpus catalog while the node runs (real players
        only, never the fake ones)."""
        if self._bridge is None and not llm.fake_enabled():
            bridge = self.bridge()
            self._log = PlayLog(open_store(self.name, MIGRATIONS, db_path))
            self._watcher = Watcher(bridge.status, self._log)
            self._watcher.start()
        if self._corpus is None and not llm.fake_enabled():
            self.library().path = db_path
            self.library().catalog()  # a first read in the background

    def pressed(self, app: str) -> None:
        if self._log is not None:
            self._log.mark(app, time.time())

    def quiet_corpus(self) -> list[str]:
        """Pauses Corpus if it plays (Spotify just started); best effort. ["Corpus"] once
        Corpus shows it stopped (up to `verify`), else []: a pause is never claimed unseen."""
        try:
            bridge = self.bridge()
            corpus = _corpus(bridge.status())
            if not (corpus and corpus.playing and bridge.control(corpus.app, "pause")):
                return []
            deadline = time.monotonic() + self.verify
            while True:
                now = bridge.status().get(corpus.app)
                if now and _settled(now) and not now.playing:
                    return [now.name]
                if time.monotonic() >= deadline:
                    return []
                time.sleep(VERIFY_STEP)
        except BridgeError:
            return []

    def observe(self, state: dict, situation: seen.Situation, current: bool) -> None:
        """The players Windows shows and what they play (the watcher's last look while the
        node runs, so a hung bridge never holds a turn); the choice she asked about, while
        it can still be answered."""
        choice = state.get("choice")
        if current and isinstance(choice, library.Choice):
            situation.question = choice.question()
            for kind, found in choice.options.items():
                label = f"{CHOICES.get(kind, kind)}: {found.label}"
                situation.options.append(seen.Option(label, len(found.tracks), "Corpus"))
        if self._watcher is not None:
            seen_at, status = self._watcher.last
            if status is None or time.time() - seen_at > FRESH:
                why = self._watcher.error or "no answer"
                situation.players = None
                situation.broken.append(f"the music players can't be read ({seen.short(why)})")
                return
        else:
            try:
                status = self.bridge().status()
            except BridgeError as e:
                situation.players = None
                situation.broken.append(f"the music players can't be reached ({seen.short(e)})")
                return
        situation.players = [
            seen.Player(p.name, p.state, p.title, p.artist)
            for p in sorted(status.players, key=lambda p: not p.known)
        ]

    def claims(self, text: str, state: dict) -> bool:
        choice = state.get("choice")
        return choice is not None and choice.answer(text) is not None

    def wants(self, text: str) -> bool:
        if command(text) is not None or wish(text, strict=True) is not None:
            return True
        # "add X to the queue", "shuffle X": only when X is clearly in Corpus ("shuffle the
        # deck", "coloca o carro na fila" go to the router)
        asked = wish(text)
        return bool(asked and asked.queued and self._clear(self._in_corpus(asked)))

    def accepts(self, text: str) -> bool:
        return command(text) is not None or wish(text) is not None

    def handle(self, request: Request, ctx: Context) -> Reply:
        choice, mode = ctx.state.pop("choice", None), ctx.state.pop("mode", "play")
        if choice is not None and (picked := choice.answer(request.text)):
            return self._corpus_play(picked, mode)
        found = command(request.text)
        if found is None:
            asked = wish(request.text)
            if asked is None:
                return Reply(
                    module=self.name,
                    text="I can play something from Corpus or Spotify if you tell me what, "
                    "pause, resume or skip what's playing, or tell you what it is.",
                )
            return self._play(asked, ctx, request.text)
        verb, app = found
        try:
            status = self.bridge().status()
        except BridgeError as e:
            return Reply(module=self.name, text=f"I can't reach your music players: {e}.")
        if verb == "now":
            return Reply(module=self.name, text=now_playing(status, app))
        chosen = targets(verb, status, app)
        if isinstance(chosen, str):
            return Reply(module=self.name, text=chosen)
        action = _ACTIONS[verb]
        return Reply(module=self.name, actions=[action(app=p.app, player=p.name) for p in chosen])

    def _in_corpus(self, asked: Wish) -> "library.Found | library.Choice | str":
        """What Corpus has for the wish, or the line she says when it has nothing."""
        if asked.kind == "playlist":
            return "Playlists are on Spotify, not in Corpus."
        mood = None
        if asked.kind == "mood":
            mood = library.mood_of(asked.query)
            if mood is None:
                return f"I don't know which of your Corpus songs sound {asked.query}."
        kinds = {"track": "song", "album": "album", "artist": "artist"}
        try:
            catalog = self.library().catalog()
        except (CorpusError, OSError) as e:
            print(f"music: Corpus' catalog unavailable ({e})", file=sys.stderr)
            return "I can't read your Corpus library right now."
        if not len(catalog):
            return "I haven't read your Corpus library yet; `prada music import` reads it."
        return library.find(catalog, asked.query, kinds.get(asked.kind), mood, asked.shuffle)

    @staticmethod
    def _clear(found) -> bool:
        return isinstance(found, library.Found | library.Choice)

    def _play(self, asked: Wish, ctx: Context, text: str) -> Reply:
        """Corpus when the music is clearly in kafka's files (or Corpus is named), else
        Spotify; queueing is Corpus only."""
        if asked.app != "spotify" and (asked.mode, asked.shuffle) != ("play", None):
            # "play Who's Next": the words read as "next" may be part of a name in Corpus
            whole = wish(text, peel=False)
            if whole and whole.app != "spotify" and self._clear(self._in_corpus(whole)):
                asked = whole
        found = None if asked.app == "spotify" else self._in_corpus(asked)
        if asked.app == "corpus" or self._clear(found):
            if isinstance(found, str):
                return Reply(module=self.name, text=found)
            if isinstance(found, library.Choice):
                ctx.state["choice"], ctx.state["mode"] = found, asked.mode
                return Reply(module=self.name, text=found.question())
            return self._corpus_play(found, asked.mode)
        if asked.mode != "play":
            missing = f"{found} " if found else ""
            return Reply(module=self.name, text=f"{missing}I can only queue songs in Corpus.")
        return Reply(module=self.name, actions=[Play(query=asked.query, want=asked.kind)])

    def _corpus_play(self, found: "library.Found", mode: str) -> Reply:
        action = PlayCorpus(
            label=found.label,
            paths=[t.path for t in found.tracks],
            titles=[t.title for t in found.tracks],
            mode=mode,
            shuffle=found.shuffle,
        )
        return Reply(module=self.name, actions=[action])
