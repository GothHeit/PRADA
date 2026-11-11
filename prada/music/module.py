"""The music module: pause, resume, skip and "what's playing?" for Spotify and Corpus.

Plain commands ("pause", "next song", "próxima", "que música é essa?") are matched in
code, so they skip the router and never wait for the model. Each button press is a typed
SAFE action: the node presses it through Windows' media controls and checks the player
really changed (paused, playing, a new track) before she says so. "What's playing?" is
answered from the players only, never guessed.

While the node runs (`start`), a watcher logs every track played to `music_play`.
"""

import re
import time
import unicodedata
from typing import ClassVar, Literal

from prada import llm
from prada.actions import Action, ActionFailed, Executor
from prada.module import Context, Module, Reply, Request
from prada.music.bridge import BridgeError, FakeBridge, Player, Status, WindowsBridge
from prada.music.playlog import MIGRATIONS, PlayLog, Watcher
from prada.store import open_store

VERIFY = 3.0  # seconds to wait for the player to show the change
VERIFY_STEP = 0.15

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
_LEAD = r"(?:(?:hey |ok |oi )?prada |please |por favor |(?:can|could|would) you |(?:voce )?(?:pode|poderia|consegue) )*"
_TRAIL = r"(?: (?:please|por favor|prada|for me|pra mim|now|agora|ai))*"
_MATCHERS = [
    (verb, re.compile(rf"{_LEAD}(?:{pattern}){_TRAIL}"))
    for verb, patterns in _COMMANDS.items()
    for pattern in patterns
]


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
    chosen = playing[:1] or [p for p in rank if p.known or p.app == status.current][:1]
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
        "Pauses, resumes and skips the music playing on the PC (Spotify, Corpus) and says "
        "what song is playing. Not for playing a particular song or artist."
    )
    migrations = MIGRATIONS
    memory_note = "[music controls]"

    def __init__(self, bridge=None, verify: float = VERIFY):
        """`bridge` replaces Windows' media controls (tests); `verify` = seconds to wait for
        a player to show a change."""
        self._bridge = bridge
        self.verify = verify
        self._log: PlayLog | None = None
        self.executors = [
            PauseExecutor(self),
            ResumeExecutor(self),
            NextExecutor(self),
            PreviousExecutor(self),
        ]

    def bridge(self):
        if self._bridge is None:
            self._bridge = FakeBridge() if llm.fake_enabled() else WindowsBridge()
        return self._bridge

    def start(self, db_path) -> None:
        """Log what plays while the node runs (real players only, never the fake ones)."""
        if self._bridge is not None or llm.fake_enabled():
            return
        bridge = self.bridge()
        self._log = PlayLog(open_store(self.name, MIGRATIONS, db_path))
        Watcher(bridge.status, self._log).start()

    def pressed(self, app: str) -> None:
        if self._log is not None:
            self._log.mark(app, time.time())

    def wants(self, text: str) -> bool:
        return command(text) is not None

    def accepts(self, text: str) -> bool:
        return command(text) is not None

    def handle(self, request: Request, ctx: Context) -> Reply:
        found = command(request.text)
        if found is None:
            return Reply(
                module=self.name,
                text="I can pause, resume or skip what's playing, or tell you what it is; "
                "I can't pick songs yet.",
            )
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
