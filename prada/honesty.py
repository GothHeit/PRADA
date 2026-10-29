"""Telling when a reply says an action happened, so a reply with no action behind it can't."""

import re

# Words that say an action happened: "I've sent it", "I turned on the lights", "Sent!",
# "Your alarm is set for 7am", "Message sent to Marina", "the lights are on now".
_DONE = (
    r"(?:done|sent|set|saved|locked|unlocked|ordered|booked|placed|playing|scheduled|started|added)"
)
_CLAIM = re.compile(
    r"\bi(?:'ve|\s+have)?\s+(?:just\s+|already\s+)?(?:sent|emailed|texted|messaged|called"
    r"|turned|switched|set(?!\s+my\b)|made\s+(?:a\s+)?note|put|saved|noted|written|wrote|added|booked|ordered|played"
    r"|scheduled|created|dimmed|locked|unlocked|opened|started|searched|looked\s+(?:it\s+)?up)\b"
    r"|(?:^|[.!]\s+)(?:done|sent|ordered|booked|saved|scheduled|locked|playing|all\s+set)\b(?!\s*\?)"
    r"|\ball\s+set\b(?!\s*\?)"
    r"|(?:\b(?:is|are|was|were|has\s+been|have\s+been)|'s|'re)\s+(?:now\s+|all\s+)?(?:"
    + _DONE
    + r"|(?:turned|switched)\s+on)\b"
    r"|(?:\b(?:is|are)|'s|'re)\s+(?:now\s+)?on(?:\s+now)?\s*(?:[.!,]|$)"
    r"|\b(?:message|email|text|reminder|alarm|pizza|order|note)\s+" + _DONE + r"\b",
    re.IGNORECASE,
)


# Verbs of things to do in the world, asked at the start of a clause or after "please",
# "can you", ...: "turn on the lights", "can you text Marina", "lock the door, please".
_VERBS = (
    r"turn|switch|send|e-?mail|text|message|call|ring|play|put\s+on|order|buy|book|reserve"
    r"|set|remind|lock|unlock|open|close|dim|schedule|start|stop|pause|cancel|add|delete"
    r"|post|pay|transfer|record"
    # Notes and messages to pass on ("write me a poem" and "tell me" are chat, not a request).
    r"|write\s+(?:(?:it|this|that)\s+)?down|jot|note|save|remember|tell(?!\s+me\b)"
    r"|keep\s+(?:track|a\s+note|in\s+mind)"
    r"|liga|desliga|acende|apaga|manda|envia|toca|pede|compra|tranca|lembra|anota|salva|avisa"
)
_ASK = re.compile(
    r"(?:^|[.!?,;:]\s*|\b(?:please|pls|can\s+you|could\s+you|would\s+you|will\s+you"
    r"|i\s+want\s+you\s+to|i\s+need\s+you\s+to|i'd\s+like\s+you\s+to|go\s+ahead\s+and"
    r"|just|now|pode|você\s+pode|por\s+favor)\s+)(?:please\s+)?(?:" + _VERBS + r")\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?]*(?:[.!?]+|$)")


def asks_action(text: str) -> bool:
    """True if the user asks for something to be done: "turn on the lights", "can you
    order a pizza?"; False for chat that only mentions it ("my sister plays guitar")."""
    return bool(_ASK.search(text.replace("’", "'").strip()))


def claims_action(text: str) -> bool:
    """True if `text` says something was done: "I've sent it", "Done!", "the lights are on".
    A question is never a claim: "Is your alarm set for 7?", "Done with work already?"."""
    text = text.replace("’", "'")
    sentences = (m.group() for m in _SENTENCE.finditer(text))
    return any(_CLAIM.search(s) for s in sentences if s.strip() and not s.rstrip().endswith("?"))
