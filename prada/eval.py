"""Score PRADA as an assistant on the fixed cases in prada/evalset/assistant.jsonl.

Each case runs through a fresh node (routing, modules, actions, memory) in a throwaway
database, and is scored on what has a right answer:
- routing: the module that answered ("anota: ..." reaches notes, "turn my PC on" wake).
- honesty: she never claims an action that didn't run (checked on every case).
- search: she offers to look up live or unknown things, and answers known facts directly.
- memory: she uses what an earlier, seeded conversation said, and says so when she doesn't
  know.
It also counts model replies that weren't valid JSON for their schema and times each turn.
How natural she sounds isn't scored: every exchange goes into a transcript file to read.

Run `prada eval --suite assistant [--model] [--think on|off]` on a machine with Ollama; web
searches use canned offline sources, so nothing leaves the house. With PRADA_FAKE_LLM=1 it
only exercises the plumbing.
"""

import argparse
import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prada import config, llm, store
from prada.memory import MIGRATIONS as MEMORY_MIGRATIONS
from prada.memory import Memory
from prada.module import Chat, Reply, Request
from prada.node import KEEP_ALIVE, Node, default_modules

EVALSET = Path(__file__).parent / "evalset" / "assistant.jsonl"
CATEGORIES = ("routing", "honesty", "search", "memory")
DAY = 24 * 3600
SESSION = "eval"

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


@dataclass
class Case:
    category: str
    text: str
    module: str | None = None  # the module that must answer
    before: list[str] = field(default_factory=list)  # earlier messages, same session, unscored
    # [said, replied] seeded as past conversations, one a day, the last one yesterday: the
    # last is what she remembers as-is, earlier ones only her memory search finds.
    memory: list[list[str]] = field(default_factory=list)
    offer: bool | None = None  # must (True) or must not (False) offer to search
    says: list[str] = field(default_factory=list)  # the reply has one of these words


@dataclass
class Result:
    case: Case
    reply: Reply
    seconds: float
    failure: str = ""  # "" when the case passed
    before: list[tuple[str, Reply]] = field(default_factory=list)


@dataclass
class Score:
    results: list[Result] = field(default_factory=list)
    calls: int = 0  # model calls
    invalid: int = 0  # model replies that weren't valid JSON for their schema


def load_cases(path: Path = EVALSET) -> list[Case]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case(**json.loads(line))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path.name}:{n}: bad eval case ({e})") from None
        if case.category not in CATEGORIES:
            raise ValueError(f"{path.name}:{n}: unknown category {case.category!r}")
        cases.append(case)
    return cases


def claims_action(text: str) -> bool:
    return bool(_CLAIM.search(text.replace("’", "'")))


def offered(reply: Reply) -> bool:
    """She offered to look it up: the assistant's offer, or the web module asking first."""
    return bool(reply.offer) or (reply.module == "web" and not reply.outcomes)


def _mentions(text: str, word: str) -> bool:
    text = text.replace("’", "'").casefold()
    return bool(re.search(rf"(?<!\w){re.escape(word.casefold())}(?!\w)", text))


def check(case: Case, reply: Reply) -> str:
    """Why the case failed, "" if it passed."""
    if reply.error:
        return f"error: {reply.error}"
    if case.module and reply.module != case.module:
        return f"went to {reply.module}, not {case.module}"
    if not any(o.ok for o in reply.outcomes) and claims_action(reply.text):
        return "claimed an action that didn't run"
    if case.offer is not None and offered(reply) != case.offer:
        return "offered to search a known fact" if not case.offer else "didn't offer to search"
    if case.says and not any(_mentions(reply.text, w) for w in case.says):
        return f"didn't say {' / '.join(case.says)}"
    return ""


class Counter:
    """Wraps chat functions to count model calls and replies that don't fit their schema."""

    def __init__(self, score: Score):
        self.score = score

    def wrap(self, chat: Chat) -> Chat:
        def counted(*args, **kwargs):
            schema = kwargs.get("schema", args[1] if len(args) > 1 else {})
            text = chat(*args, **kwargs)
            self.score.calls += 1
            try:
                data = json.loads(text)
                ok = isinstance(data, dict) and all(k in data for k in schema.get("required", []))
            except ValueError:
                ok = False
            self.score.invalid += not ok
            return text

        return counted


def eval_modules():
    """The node's modules, with web searches answered by canned offline sources."""
    from prada.web.module import FakeSource, WebModule

    web = WebModule(sources={name: FakeSource(name) for name in config.WEB_SOURCES})
    return [web if m.name == "web" else m for m in default_modules()]


def _seed(path: Path, memory: list[list[str]]) -> None:
    db = store.open_store("memory", MEMORY_MIGRATIONS, path)
    try:
        for i, (said, replied) in enumerate(memory):
            days = len(memory) - i
            Memory(db, clock=lambda days=days: time.time() - days * DAY).add(said, replied)
    finally:
        db.close()


def run_case(case: Case, client: Callable[[Chat], Chat], path: Path) -> Result:
    if case.memory:
        _seed(path, case.memory)
    node = Node(eval_modules(), db_path=path, background=False, client=client)
    before = [(t, node.handle(Request(text=t, session=SESSION))) for t in case.before]
    start = time.perf_counter()
    reply = node.handle(Request(text=case.text, session=SESSION))
    seconds = time.perf_counter() - start
    return Result(case, reply, seconds, check(case, reply), before)


def score(cases: list[Case] | None = None, client: Callable[[Chat], Chat] | None = None) -> Score:
    """Runs every case; `client(fake)` gives a module's chat (default: the configured model)."""
    result = Score()
    counter = Counter(result)
    base = client or (lambda fake: llm.client(fake, keep_alive=KEEP_ALIVE))
    with tempfile.TemporaryDirectory(prefix="prada-eval-") as tmp:
        for i, case in enumerate(cases if cases is not None else load_cases()):
            path = Path(tmp) / f"case{i}.db"
            result.results.append(run_case(case, lambda fake: counter.wrap(base(fake)), path))
    return result


def warm_up(client: Callable[[Chat], Chat]) -> None:
    """One untimed call, so loading the model isn't counted in the first case's time."""
    from prada.assistant import Answer, fake_chat

    try:
        client(fake_chat)(
            messages=[{"role": "user", "content": "hi"}], schema=Answer.model_json_schema()
        )
    except Exception:
        pass  # e.g. Ollama down: every case will say so


def _pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({100 * ok / total:.0f}%)" if total else "n/a"


def format_report(s: Score) -> str:
    lines = [f"[assistant] {len(s.results)} cases"]
    for cat in CATEGORIES:
        rs = [r for r in s.results if r.case.category == cat]
        lines.append(f"  {cat + ':':<16}{_pct(sum(not r.failure for r in rs), len(rs))}")
    claims = sum(r.failure.startswith("claimed") for r in s.results)
    turns = max(len(s.results), 1)
    lines += [
        f"  false action claims: {claims}",
        f"  invalid JSON:        {s.invalid} of {s.calls} model replies",
        f"  seconds per turn:    {sum(r.seconds for r in s.results) / turns:.1f}",
    ]
    failed = [r for r in s.results if r.failure]
    if failed:
        lines.append("  failed:")
        lines += [f"    [{r.case.category}] {r.case.text!r}: {r.failure}" for r in failed]
    return "\n".join(lines)


def _said(reply: Reply) -> str:
    return reply.text if not reply.error else f"(error: {reply.error})"


def format_transcript(s: Score, model: str) -> str:
    """Every exchange, to read how natural she sounds."""
    lines = [f"# PRADA assistant eval: {model}", "", time.strftime("%Y-%m-%d %H:%M"), ""]
    lines += ["```", format_report(s), "```"]
    for cat in CATEGORIES:
        lines += ["", f"## {cat}"]
        for n, r in enumerate((r for r in s.results if r.case.category == cat), 1):
            verdict = "pass" if not r.failure else f"FAIL: {r.failure}"
            lines += ["", f"### {n}. {verdict} ({r.seconds:.1f} s)", ""]
            for said, replied in r.case.memory:
                lines += [f"- *remembered:* you: {said} / PRADA: {replied}"]
            for text, reply in r.before:
                lines += [f"- **you:** {text}", f"- **PRADA ({reply.module}):** {_said(reply)}"]
            lines += [
                f"- **you:** {r.case.text}",
                f"- **PRADA ({r.reply.module}):** {_said(r.reply)}",
            ]
    return "\n".join(lines) + "\n"


def transcript_path(model: str) -> Path:
    name = re.sub(r"[^\w.-]+", "-", model).strip("-")
    return store.data_dir() / "eval" / f"assistant-{name}-{time.strftime('%Y%m%d-%H%M%S')}.md"


def run(model: str | None = None, think: str | None = None, transcript: str | None = None) -> int:
    think = None if think is None else think == "on"
    try:
        settings = config.llm_settings()
    except ValueError as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1
    if llm.fake_enabled():
        label = "fake"
    else:
        think = settings.think if think is None else think
        label = (model or settings.model) + (
            "" if think is None else f", think {'on' if think else 'off'}"
        )
    print(f"model: {label}")
    try:
        client = lambda fake: llm.client(fake, KEEP_ALIVE, model, think)  # noqa: E731
        warm_up(client)
        s = score(client=client)
    except Exception as e:  # one line, not a traceback
        print(f"eval: [assistant] stopped: {e}", file=sys.stderr)
        return 1
    print(format_report(s))
    path = Path(transcript) if transcript else transcript_path(label)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_transcript(s, label), encoding="utf-8")
        print(f"transcript: {path}")
    except OSError as e:
        print(f"eval: could not write the transcript ({e})", file=sys.stderr)
    if s.results and all(r.reply.error for r in s.results):
        print(f"eval: [assistant] every case failed: {s.results[0].reply.error}", file=sys.stderr)
        return 1  # e.g. Ollama not running
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m prada.eval", description="Score PRADA as an assistant"
    )
    parser.add_argument("--model", help="Ollama model to score instead of the configured one")
    parser.add_argument("--think", choices=["on", "off"], help="a thinking model's reasoning")
    parser.add_argument("--transcript", help="where to write every exchange (Markdown)")
    args = parser.parse_args(argv)
    return run(args.model, args.think, args.transcript)


if __name__ == "__main__":
    sys.exit(main())
