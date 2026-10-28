"""Text stand-ins for the microphone and speakers (PRADA_FAKE_LLM, tests, smoke runs).

Each stdin line is one request spoken right after a wake word; replies are printed.
"""

import sys


class TextEars:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._line = ""

    def wait_for_wake(self) -> str | None:
        line = self.stream.readline()
        if not line:
            return None
        self._line = line.strip()
        return "text"

    def listen(self, wait: float | None = None) -> str:
        if wait is not None:
            return ""  # follow-ups are for a microphone; every text line starts with a wake word
        return self._line


class TextMouth:
    def chime(self) -> None:
        pass

    def say(self, text: str) -> None:
        print(text, flush=True)
