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


_NUMBERS = "a two three four five six seven eight nine ten eleven twelve".split()


def _count(n: int, unit: str) -> str:
    word = _NUMBERS[n - 1] if 1 <= n <= len(_NUMBERS) else str(n)
    return f"{word} {unit}{'s' if n > 1 else ''}"


def _part(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 18:
        return "afternoon"
    return "evening" if 18 <= hour < 23 else "night"


def perceived(ts: float, now: float) -> str:
    """When something happened, as a person remembers it: "this morning", "yesterday
    evening", "on Tuesday afternoon", "last week, one evening", "about three months ago,
    one morning". Starts lower case."""
    then, today = datetime.fromtimestamp(ts), datetime.fromtimestamp(now)
    days = (today.date() - then.date()).days
    part = _part(then.hour)
    if days <= 0:
        if part == "night":
            # 02:00 seen from 10:00 is last night; from 03:00 or 23:30 it is tonight
            return "last night" if then.hour < 5 and today.hour >= 5 else "earlier tonight"
        return f"this {part}"
    if days == 1:
        return "last night" if part == "night" else f"yesterday {part}"
    if days < 7:
        return f"on {then.strftime('%A')} {part}"
    if days < 14:
        when = "last week"
    elif days < 60:
        when = f"about {_count(round(days / 7), 'week')} ago"
    elif days < 365:
        when = f"about {_count(round(days / 30.4), 'month')} ago"
    else:
        when = f"about {_count(round(days / 365), 'year')} ago"
    return f"{when}, one {part}"
