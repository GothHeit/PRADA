"""The voice loop and the pieces it is made of, independent of any audio library.

`Ears` waits for a wake word and then hears one request; `Mouth` speaks. `run` wires them to
a session (`prada.client`), so swapping in text fakes (`prada.voice.fake`) exercises the
whole loop without audio hardware or models.
"""

import re
import sys
from typing import Protocol

from prada.client import NodeError
from tutor.languages import detect_language

SAMPLE_RATE = 16000  # what openWakeWord and whisper.cpp expect
FRAME = 1280  # samples per frame: 80 ms, openWakeWord's step
FRAME_SECONDS = FRAME / SAMPLE_RATE
MIN_SPEECH_RMS = 300.0  # int16 loudness below which nothing counts as speech

# What PRADA says when the request never got an answer.
CANT_REACH = "Hmm, I can't reach my thoughts right now. Is my node running?"
WENT_WRONG = "Sorry, something went wrong on my side. Could you try again?"


class Ears(Protocol):
    def wait_for_wake(self) -> str | None:
        """Block until a wake word is heard; returns its name, or None when input ended."""

    def listen(self) -> str:
        """Hear one request and return its text ("" when nothing was said)."""


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


def run(session, ears: Ears, mouth: Mouth, once: bool = False) -> int:
    """Wake word -> request -> node -> spoken reply, until input ends; returns an exit code."""
    while (word := ears.wait_for_wake()) is not None:
        mouth.chime()
        text = ears.listen()
        if not text:
            continue
        print(f"[{word}] you: {text}", file=sys.stderr)
        try:
            reply = session.send(text)
        except NodeError as e:
            print(f"prada: {e}", file=sys.stderr)
            mouth.say(CANT_REACH)
        else:
            if reply.error:
                print(f"prada: {reply.error}", file=sys.stderr)
            mouth.say(WENT_WRONG if reply.error else reply.text)
        if once:
            break
    return 0
