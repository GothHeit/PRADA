"""Speaking a reply while the model is still writing it.

The model answers in JSON ({"reply": "..."}), so `FieldText` decodes one string field out of
the chunks as they arrive, and `Sentences` hands each finished sentence on, so the voice can
start on the first one while the rest is being written. `rest` is what is left to say of
the final reply once the streamed sentences were spoken.
"""

import json
import re
from collections.abc import Callable

MIN_SENTENCE = 8  # chars; a shorter sentence ("Hi!", "1.") waits for the next one

_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?:\s+|$)|[。！？]+|\n+")


class FieldText:
    """The text of one string field of a JSON object, decoded as the model writes it."""

    def __init__(self, name: str):
        self._key = re.compile(rf'"{re.escape(name)}"\s*:\s*"')
        self._raw = ""
        self._begin: int | None = None
        self._sent = 0

    def feed(self, chunk: str) -> str:
        """Add raw model output; returns the newly decoded text of the field ("" if none)."""
        self._raw += chunk
        if self._begin is None:
            if (m := self._key.search(self._raw)) is None:
                return ""
            self._begin = m.end()
        body = self._raw[self._begin :]
        i = 0
        while i < len(body) and body[i] != '"':
            if body[i] == "\\":
                step = 6 if body[i + 1 : i + 2] == "u" else 2
                if i + step > len(body):
                    break  # an escape cut in half: wait for the rest
                try:
                    high = step == 6 and 0xD800 <= int(body[i + 2 : i + 6], 16) < 0xDC00
                except ValueError:
                    return ""  # not a real escape; json.loads below would refuse it too
                if high:
                    if i + 12 > len(body):
                        break  # the first half of a surrogate pair
                    step = 12
                i += step
            else:
                i += 1
        try:
            text = json.loads(f'"{body[:i]}"')
        except ValueError:
            return ""
        new, self._sent = text[self._sent :], len(text)
        return new


class Sentences:
    """Collects text and calls `say` with each finished sentence; the unfinished end stays."""

    def __init__(self, say: Callable[[str], None]):
        self._say = say
        self._buf = ""

    def feed(self, text: str) -> None:
        self._buf += text
        start = 0
        for m in _END.finditer(self._buf):
            if m.end() == len(self._buf) and not m.group().endswith(("\n", " ", "。", "！", "？")):
                break  # "3." at the very end may still become "3.5"
            sentence = self._buf[start : m.end()].strip()
            if len(sentence) >= MIN_SENTENCE:
                self._say(sentence)
                start = m.end()
        self._buf = self._buf[start:]


def rest(text: str, said: list[str]) -> str:
    """What is left of `text` after the sentences in `said`, spoken in order from its start;
    all of `text` if they don't match it (better said twice than not at all)."""
    at = 0
    for part in said:
        found = text.find(part, at)
        if found < 0 or text[at:found].strip():
            return text
        at = found + len(part)
    return text[at:].strip()
