"""The voice loop and the pieces it is made of, independent of any audio library.

`Ears` waits for a wake word and then hears one request; `Mouth` speaks. `run` wires them to
a session (`prada.client`), so swapping in text fakes (`prada.voice.fake`) exercises the
whole loop without audio hardware or models.

Barge-in: while she thinks and speaks (in a thread), ears that can (`BargeEars.watch`) keep
listening for a wake word on the main thread, the only one that ever reads the mic; hearing one sets a `Barge`, she stops mid-sentence and listens again. Only the
wake word interrupts her, not loudness: her own voice reaches the mic, and a model trained on
a phrase won't fire on her speech unless she says that phrase, so a line that names a wake
word is spoken with the barge held. The node is told what was heard (`session.cut`), so her
memory doesn't assume she said the rest.
"""

import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from prada.client import NodeError
from prada.stream import rest
from tutor.languages import detect_language

SAMPLE_RATE = 16000  # what openWakeWord and whisper.cpp expect
FRAME = 1280  # samples per frame: 80 ms, openWakeWord's step
FRAME_SECONDS = FRAME / SAMPLE_RATE
MIN_SPEECH_RMS = 300.0  # int16 loudness below which nothing counts as speech
MAX_FOLLOW_UPS = 4  # replies in a row without a wake word (a TV can't chain them forever)
VERBOSE = False  # `prada voice --verbose`: say what the ears and mouth are doing on stderr

# What PRADA says when the request never got an answer.
CANT_REACH = "Hmm, I can't reach my thoughts right now. Is my node running?"
WENT_WRONG = "Sorry, something went wrong on my side. Could you try again?"


class Ears(Protocol):
    def wait_for_wake(self) -> str | None:
        """Block until a wake word is heard; returns its name, or None when input ended."""

    def listen(self, wait: float | None = None) -> str:
        """Hear one request and return its text ("" when nothing was said). `wait` is how many
        seconds to wait for speech to begin; given, this is a follow-up straight after a reply."""


class BargeEars(Ears, Protocol):
    def watch(self, barge: "Barge", ready: Callable[[], None]) -> None:
        """Call `ready()` once listening, then listen for a wake word until `barge.over` is set
        (the turn ended); hearing one (unless `barge.held`) sets `barge`."""


class Mouth(Protocol):
    def chime(self) -> None:
        """Signal that PRADA is listening."""

    def say(self, text: str, barge: "Barge | None" = None) -> bool | None:
        """Speak `text`; with `barge`, stop as soon as it is set and return False."""


def phrase(wake_word: str) -> str:
    """What a wake word model listens for, from its name: "wake_up_prada" -> "wake up prada"
    ("hey_jarvis_v0.1" and paths to .onnx files too)."""
    name = Path(wake_word).name.removesuffix(".onnx").removesuffix(".tflite")
    return " ".join(re.sub(r"_v\d.*$", "", name).replace("_", " ").replace("-", " ").split())


class Barge:
    """An interruption: set by the ears when they hear a wake word while she talks."""

    def __init__(self, wake_words=()):
        self.event = threading.Event()
        self.over = threading.Event()  # set too when the turn ends without an interruption
        self.mouth = threading.Lock()  # held while she speaks, so a chime never overlaps her
        self.held = False  # she is saying a wake phrase herself: don't take it as one
        words = [phrase(w).split() for w in wake_words]
        self._phrases = [
            re.compile(r"\b" + r"\W+".join(map(re.escape, w)) + r"\b", re.IGNORECASE)
            for w in words
            if w
        ]

    def set(self) -> None:
        debug("interrupted: she stops")
        self.event.set()
        self.over.set()

    def is_set(self) -> bool:
        return self.event.is_set()

    def names_wake_word(self, line: str) -> bool:
        return any(p.search(line) for p in self._phrases)


def debug(line: str) -> None:
    if VERBOSE:
        print(
            f"  ({time.strftime('%H:%M:%S')} {threading.current_thread().name}) {line}",
            file=sys.stderr,
        )


def track_noise(noise: float, rms: float) -> float:
    """Background loudness: follows quieter frames at once, louder ones only slowly."""
    return min(max(rms, 1.0), noise * 1.005)


class Endpointer:
    """Decides when a spoken request is over, one frame's loudness (RMS) at a time."""

    def __init__(
        self, noise: float, silence: float = 0.8, wait: float = 5.0, longest: float = 15.0
    ):
        self.level = max(noise * 3, MIN_SPEECH_RMS)
        self.silence_frames = max(1, round(silence / FRAME_SECONDS))
        self.wait_frames = round(wait / FRAME_SECONDS)
        self.longest_frames = round(longest / FRAME_SECONDS)
        self.heard = False
        self.frames = 0
        self.quiet = 0

    def push(self, rms: float) -> bool:
        """Add a frame; True once the request has ended (or never started in time)."""
        self.frames += 1
        if rms >= self.level:
            self.heard = True
            self.quiet = 0
        else:
            self.quiet += 1
        if not self.heard:
            return self.frames >= self.wait_frames
        return self.quiet >= self.silence_frames or self.frames >= self.longest_frames


def split_by_language(text: str, languages) -> list[tuple[str | None, str]]:
    """Cut a reply into runs of sentences, each tagged with its language when it is one of
    `languages` (else None), so each run can be spoken by the right voice."""
    runs: list[tuple[str | None, str]] = []
    for sentence in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", text.strip()):
        if not sentence:
            continue
        lang = detect_language(sentence)
        lang = lang if lang in languages else None
        if runs and runs[-1][0] == lang:
            runs[-1] = (lang, f"{runs[-1][1]} {sentence}")
        else:
            runs.append((lang, sentence))
    return runs


def respell(text: str, say_as: dict[str, str]) -> str:
    """Swap whole words for the spellings that make the voice say them right."""
    if not say_as:
        return text
    lookup = {word.lower(): spelling for word, spelling in say_as.items()}
    words = "|".join(re.escape(word) for word in sorted(say_as, key=len, reverse=True))
    return re.sub(
        rf"\b(?:{words})\b", lambda m: lookup[m.group().lower()], text, flags=re.IGNORECASE
    )


def _utter(mouth: Mouth, line: str, barge: Barge | None) -> bool:
    """Say `line`; False if she was interrupted before its end (or already had been)."""
    if barge is None:
        mouth.say(line)
        return True
    if barge.is_set():
        return False
    with barge.mouth:
        barge.held = barge.names_wake_word(line)
        try:
            return mouth.say(line, barge) is not False and not barge.is_set()
        finally:
            barge.held = False


def _speaker(mouth: Mouth, said: list[str], barge: Barge | None):
    """Says what the node streams at once; the reply's own sentences she finished go in
    `said`. Once interrupted she says nothing more."""

    def speak(kind: str, line: str) -> None:
        if _utter(mouth, line, barge) and kind == "part":
            said.append(line)

    return speak


def _log(module: str | None, line: str) -> None:
    """Her side of the console log, next to the "[wake word] you: ..." lines."""
    print(f"[{module or 'voice'}] prada: {line}", file=sys.stderr)


def _say(mouth: Mouth, module: str | None, line: str, barge: Barge | None = None) -> None:
    _log(module, line)
    _utter(mouth, line, barge)


def _turn(session, text: str, mouth: Mouth, barge: Barge | None, said: list[str]):
    """Send `text` and speak the reply as it comes; returns the Reply (None if the node
    couldn't be reached). The reply's sentences she finished go in `said`."""
    try:
        reply = session.send(text, speak=_speaker(mouth, said, barge))
    except NodeError as e:
        print(f"error: {e}", file=sys.stderr)
        _say(mouth, "", CANT_REACH, barge)
        return None
    if reply.error:
        print(f"error: {reply.error}", file=sys.stderr)
        _say(mouth, reply.module, WENT_WRONG, barge)
    else:
        _log(reply.module, reply.text)
        if (left := rest(reply.text, said)) and _utter(mouth, left, barge):
            said.append(left)
    return reply


def _cut(session, reply, said: list[str]) -> None:
    """Tell the node she was interrupted after the sentences in `said` (nothing to tell if
    she had said all of it)."""
    if reply is None or reply.error or not reply.text or not rest(reply.text, said):
        return
    heard = " ".join(said)
    print(
        f"[{reply.module or 'voice'}] (cut short{': ' + heard if heard else ''})", file=sys.stderr
    )
    try:
        session.cut(reply.text, heard)
    except NodeError as e:
        print(f"error: {e}", file=sys.stderr)


def run(
    session,
    ears: Ears,
    mouth: Mouth,
    once: bool = False,
    follow_up: float = 0.0,
    barge_in: bool = False,
    wake_words=(),
) -> int:
    """Wake word -> request -> node -> spoken reply, until input ends; returns an exit code.

    She speaks each sentence as soon as the node has it, and a filler line before a slow
    action. After a reply she keeps listening for `follow_up` seconds, so the next request
    needs no wake word (up to `MAX_FOLLOW_UPS` in a row); silence for that long sends her
    back to waiting for one (0 turns it off).

    With `barge_in` (`ears` must be `BargeEars`), a wake word while she thinks or speaks
    stops her; the node hears how much she said, and she listens for the new request at once.
    `wake_words` are her ears' models, so she can name one without interrupting herself."""
    while (word := ears.wait_for_wake()) is not None:
        mouth.chime()
        text = ears.listen()
        follow_ups = 0
        while text:
            print(f"[{word}] you: {text}", file=sys.stderr)
            said: list[str] = []
            if not barge_in:
                _turn(session, text, mouth, None, said)
            else:
                # The turn runs in a thread, so an interruption is answered at once, even
                # while the node is still thinking or running an action.
                barge = Barge(wake_words)
                result = []

                def turn(text=text, barge=barge, said=said, result=result) -> None:
                    try:
                        result.append(_turn(session, text, mouth, barge, said))
                    finally:
                        debug("she is done")
                        barge.over.set()

                worker = threading.Thread(target=turn, name="turn", daemon=True)
                debug("turn starts")
                ears.watch(barge, worker.start)
                debug("watch ended")
                if barge.is_set():
                    with barge.mouth:  # she has stopped
                        mouth.chime()
                    text = "" if once else ears.listen()
                    worker.join()  # the node must be done before the next request anyway
                    _cut(session, result[0] if result else None, said)
                    if once:
                        return 0
                    follow_ups = 0
                    continue
                worker.join()
            if once:
                return 0
            follow_ups += 1
            more = follow_up > 0 and follow_ups <= MAX_FOLLOW_UPS
            text = ears.listen(follow_up) if more else ""
    return 0
