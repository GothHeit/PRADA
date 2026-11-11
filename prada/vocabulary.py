"""Words her ears should know: names whisper would otherwise spell as common words.

The node gathers them (`Node.vocabulary`): her own name, names in what she remembers about
you, and what each module offers (music: artists and songs you play). The voice client
puts them in whisper's prompt (`prada.voice.loop.Hints`), most useful last, within
whisper's budget.
"""

import re

MAX_NAMES = 20  # names taken from her memory, newest facts first

# Capitalized words that start a sentence without being names (her facts read "You ...").
_COMMON = set(
    "i i'm i've you your you're you've he she it we they his her its our their my me "
    "the a an this that these those there here what when where who why how and but or so "
    "if then also yes no not on in at to of for with by from as is are was were be "
    "monday tuesday wednesday thursday friday saturday sunday "
    "january february march april may june july august september october november december "
    "eu você voce ele ela nós nos eles elas seu sua meu minha o os as um uma de do da em no "
    "na e mas ou se que quando onde como".split()
)
_WORD = re.compile(r"[^\W\d_][\w'’.\-]*")


def names_in(text: str) -> list[str]:
    """Runs of capitalized words in `text`, as they are written: "You call your sister
    Ana Clara and play Toru Kitajima songs." -> ["Ana Clara", "Toru Kitajima"]."""
    names: list[str] = []
    run: list[str] = []
    end = -1
    for m in _WORD.finditer(text):
        word = re.sub(r"['’]s$", "", m.group().rstrip(".'’-"))  # "Ana's" -> "Ana"
        capital = word[:1].isupper() and len(word) > 1 and word.casefold() not in _COMMON
        if capital and run and text[end : m.start()] == " ":
            run.append(word)
        else:
            if run:
                names.append(" ".join(run))
            run = [word] if capital else []
        end = m.start() + len(word)
    if run:
        names.append(" ".join(run))
    return names


def unique(words, limit: int | None = None) -> list[str]:
    """`words` without blanks or repeats (any case or spacing), in order, at most `limit`."""
    seen: set[str] = set()
    out: list[str] = []
    for word in words:
        word = " ".join(str(word).split())
        if word and word.casefold() not in seen:
            seen.add(word.casefold())
            out.append(word)
            if limit is not None and len(out) >= limit:
                break
    return out


def memory_names(facts) -> list[str]:
    """Names in the facts still true (rows with `fact` and `superseded_by`), newest first."""
    live = [f["fact"] for f in facts if f["superseded_by"] is None]
    return unique((name for fact in reversed(live) for name in names_in(fact)), MAX_NAMES)
