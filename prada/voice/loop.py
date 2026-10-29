"""The voice loop and the pieces it is made of, independent of any audio library.

`Ears` waits for a wake word and then hears one request; `Mouth` speaks. `run` wires them to
a session (`prada.client`), so swapping in text fakes (`prada.voice.fake`) exercises the
whole loop without audio hardware or models.
"""

import re
import sys
from typing import Protocol

from prada.client import NodeError
from prada.stream import rest
from tutor.languages import detect_language

SAMPLE_RATE = 16000  # what openWakeWord and whisper.cpp expect
FRAME = 1280  # samples per frame: 80 ms, openWakeWord's step
FRAME_SECONDS = FRAME / SAMPLE_RATE
MIN_SPEECH_RMS = 300.0  # int16 loudness below which nothing counts as speech
MAX_FOLLOW_UPS = 4  # replies in a row without a wake word (a TV can't chain them forever)

# What PRADA says when the request never got an answer.
CANT_REACH = "Hmm, I can't reach my thoughts right now. Is my node running?"
WENT_WRONG = "Sorry, something went wrong on my side. Could you try again?"


class Ears(Protocol):
    def wait_for_wake(self) -> str | None:
        """Block until a wake word is heard; returns its name, or None when input ended."""

    def listen(self, wait: float | None = None) -> str:
        """Hear one request and return its text ("" when nothing was said). `wait` is how many
        seconds to wait for speech to begin; given, this is a follow-up straight after a reply."""


class Mouth(Protocol):
    def chime(self) -> None:
        """Signal that PRADA is listening."""

    def say(self, text: str) -> None: ...


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


def _speaker(mouth: Mouth, said: list[str]):
    """Says what the node streams at once; the reply's own sentences go in `said`."""

    def speak(kind: str, line: str) -> None:
        mouth.say(line)
        if kind == "part":
            said.append(line)

    return speak


def _log(module: str | None, line: str) -> None:
    """Her side of the console log, next to the "[wake word] you: ..." lines."""
    print(f"[{module or 'voice'}] prada: {line}", file=sys.stderr)


def _say(mouth: Mouth, module: str | None, line: str) -> None:
    _log(module, line)
    mouth.say(line)


def run(session, ears: Ears, mouth: Mouth, once: bool = False, follow_up: float = 0.0) -> int:
    """Wake word -> request -> node -> spoken reply, until input ends; returns an exit code.

    She speaks each sentence as soon as the node has it, and a filler line before a slow
    action. After a reply she keeps listening for `follow_up` seconds, so the next request
    needs no wake word (up to `MAX_FOLLOW_UPS` in a row); silence for that long sends her
    back to waiting for one (0 turns it off)."""
    while (word := ears.wait_for_wake()) is not None:
        mouth.chime()
        text = ears.listen()
        follow_ups = 0
        while text:
            print(f"[{word}] you: {text}", file=sys.stderr)
            said: list[str] = []
            try:
                reply = session.send(text, speak=_speaker(mouth, said))
            except NodeError as e:
                print(f"error: {e}", file=sys.stderr)
                _say(mouth, "", CANT_REACH)
            else:
                if reply.error:
                    print(f"error: {reply.error}", file=sys.stderr)
                    _say(mouth, reply.module, WENT_WRONG)
                else:
                    _log(reply.module, reply.text)
                    if left := rest(reply.text, said):
                        mouth.say(left)
            if once:
                return 0
            follow_ups += 1
            more = follow_up > 0 and follow_ups <= MAX_FOLLOW_UPS
            text = ears.listen(follow_up) if more else ""
    return 0
