"""The status block: what PRADA's own code knows right now, for the model.

It goes at the end of the user's latest message, never in the system prompt, so the persona
stays the same text from one turn to the next, and the conversation too until its window
of recent messages starts to slide (Ollama reuses its cache for an unchanged start of the
prompt). It also lets her know the date and time.
Only the newest message carries it; the stored conversation never does.
"""

from datetime import datetime

OPEN, CLOSE = "<status>", "</status>"


def now_line(ts: float) -> str:
    return "Now: " + datetime.fromtimestamp(ts).astimezone().strftime("%A %d %B %Y, %H:%M")


def since_line(last: float | None, now: float) -> str:
    if last is None:
        return "You have never talked before."
    return f"You last talked {ago(now - last)}."


def ago(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 1:
        return "less than a minute ago"
    for size, unit in ((60 * 24, "day"), (60, "hour"), (1, "minute")):
        if minutes >= size:
            n = minutes // size
            return f"{n} {unit}{'s' if n > 1 else ''} ago"
    return ""


def attach(text: str, lines: list[str]) -> str:
    """`text` with the status block after it; `text` alone when there is nothing to say."""
    lines = [line for line in lines if line]
    if not lines:
        return text
    return f"{text}\n\n{OPEN}\n" + "\n".join(lines) + f"\n{CLOSE}"


def detach(text: str) -> str:
    """The user's own words, without the status block."""
    head, sep, _ = text.rpartition(f"\n\n{OPEN}\n")
    return head if sep else text
