"""Hard cases: does she understand what kafka wants done in the situation she is in?

Each case in prada/evalset/understand.jsonl sets a situation (what the players are doing, her
last line, a search she offered), plays `before` turns through the node, then says one more
thing and checks which *calls* she made: the tool and its key arguments, or no action at all.
Today's pipeline (regex shortcuts, router, modules) is scored as it is: the actions it ran or
held, a memory fix and an offer to search become calls, so the first run on the real model
is the baseline of the rules that a model-led mind has to beat.

Metrics: right calls, false actions (a call nobody asked for), missed actions, wrong calls,
false claims (she says something was done when nothing ran: such a case fails even with the
right calls), invalid JSON, and milliseconds per stage (each model caller, then the code and
actions around them).

Run `prada eval --suite understand [--model] [--think on|off] [--backend NAME]`. Players,
Spotify, Corpus and the web are fakes, and the PC is never woken: nothing outside the house
changes. With PRADA_FAKE_LLM=1 it only exercises the plumbing (and the code-only paths).
"""

import json
import re
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from prada import backends, config, llm, router, store
from prada.eval import SESSION, _seed, offered, warm_up
from prada.honesty import claims_action
from prada.memory import MIGRATIONS as MEMORY_MIGRATIONS
from prada.memory import Memory, corrections
from prada.memory import fake_chat as memory_fake_chat
from prada.module import Chat, Reply, Request
from prada.node import KEEP_ALIVE, Node

EVALSET = Path(__file__).parent / "evalset" / "understand.jsonl"
GROUPS = (
    "correction",
    "polite",
    "reference",
    "change",
    "not_music",
    "notes_wake_search",
    "multi",
    "portuguese",
)
# The calls a case can expect, with the arguments that can be checked.
TOOLS = {
    "music.pause": {"app"},
    "music.resume": {"app"},
    "music.next": {"app"},
    "music.previous": {"app"},
    "music.play": {"what", "app", "mode", "tracks"},
    "notes.save": {"text"},
    "wake.send": set(),
    "web.search": {"query"},
    "web.offer": {"query"},  # she asked whether to search instead of searching
    "memory.fix": {"wrong", "right"},
}
APPS = {"Spotify": "Spotify.exe", "Corpus": "Corpus.Corpus"}
# Corpus' fake files plus a second song on the album "Wave", so "play Wave" has to ask
# "the song or the album?" (a pending choice).
EXTRA_SONGS = [
    {
        "path": "E:\\Music\\Jobim\\Wave\\02 Triste.mp3",
        "title": "Triste",
        "artist": "Antonio Carlos Jobim",
        "album": "Wave",
        "genre": "Bossa Nova",
        "tags": [],
        "duration": 190.0,
    }
]
FAIL_KINDS = ("false action", "missed action", "wrong call", "false claim")


@dataclass
class Case:
    group: str
    text: str
    # [{"tool": "music.play", "what": "space dementia", "app": "Corpus"}, ...]; [] = no action
    calls: list[dict]
    # "players": [{"app": "Spotify", "state": "playing", "title": "Unravel"}] (titles from
    # the fake tracks; [] = nothing open; missing = Spotify playing, Corpus paused);
    # "last": [kafka's line, her reply] as the last exchange of the conversation;
    # "offer": what she just offered to look up online.
    situation: dict = field(default_factory=dict)
    before: list[str] = field(default_factory=list)  # earlier messages, same session
    memory: list[list[str]] = field(default_factory=list)  # past conversations, as in eval.py
    note: str = ""  # why the case is hard, for the person reading the report


@dataclass
class Result:
    case: Case
    reply: Reply
    calls: list[dict]
    seconds: float
    stages: dict[str, float]  # ms per stage of the scored turn
    failure: str = ""
    claimed: bool = False  # said something was done when nothing ran
    before: list[tuple[str, Reply]] = field(default_factory=list)


@dataclass
class Score:
    results: list[Result] = field(default_factory=list)
    calls: int = 0  # model calls
    invalid: int = 0


def _check_case(case: Case) -> None:
    if case.group not in GROUPS:
        raise ValueError(f"unknown group {case.group!r}")
    for call in case.calls:
        tool = call.get("tool")
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        if extra := set(call) - TOOLS[tool] - {"tool"}:
            raise ValueError(f"{tool} has no argument {sorted(extra)}")
    if extra := set(case.situation) - {"players", "last", "offer"}:
        raise ValueError(f"unknown situation {sorted(extra)}")
    from prada.music.bridge import FAKE_TRACKS

    titles = {t[0] for t in FAKE_TRACKS}
    for p in case.situation.get("players", []):
        if p.get("title", FAKE_TRACKS[0][0]) not in titles:
            raise ValueError(f"no fake track {p['title']!r}")
        if p.get("state") not in ("playing", "paused", "stopped"):
            raise ValueError(f"player state {p.get('state')!r}")


def load_cases(path: Path = EVALSET) -> list[Case]:
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case(**json.loads(line))
            _check_case(case)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path.name}:{n}: bad eval case ({e})") from None
        cases.append(case)
    return cases


# --- what she did, as calls -------------------------------------------------------------


def _call(action: dict) -> dict:
    kind = action.get("kind", "")
    if kind in ("music.pause", "music.resume", "music.next", "music.previous"):
        return {"tool": kind, "app": action["player"]}
    if kind == "music.play":
        return {"tool": kind, "what": action["query"], "app": "Spotify", "mode": "play"}
    if kind == "music.play_corpus":
        return {
            "tool": "music.play",
            "what": action["label"],
            "app": "Corpus",
            "mode": action["mode"],
            "tracks": len(action["paths"]),
        }
    if kind == "notes.save":
        return {"tool": kind, "text": action["text"]}
    if kind == "web.search":
        return {"tool": kind, "query": action["query"]}
    return {"tool": kind}


def calls_of(text: str, reply: Reply, held: list[dict]) -> list[dict]:
    """The calls today's pipeline made for `text`: actions it ran (ok or not), actions held
    for a yes, a memory fix, and an offer to search."""
    calls = [_call(o.action) for o in reply.outcomes] + [_call(a) for a in held]
    if reply.data.get("fixed"):
        pairs = corrections(text) + ([reply.correction] if reply.correction else [])
        wrong, right = pairs[0] if pairs else ("", "")
        calls.append({"tool": "memory.fix", "wrong": wrong, "right": right})
    if offered(reply):
        calls.append({"tool": "web.offer", "query": reply.offer or text})
    return calls


def _words(text: str) -> set[str]:
    from prada.music.play import fold

    return set(fold(str(text)).split())


def matches(expected: dict, call: dict) -> bool:
    """The call is the expected tool and every expected argument holds: words all said (in
    any order), apps and modes equal, numbers equal."""
    if call.get("tool") != expected["tool"]:
        return False
    for key, want in expected.items():
        if key == "tool":
            continue
        got = call.get(key)
        if isinstance(want, int):
            if got != want:
                return False
        elif key in ("app", "mode"):
            if str(got).casefold() != str(want).casefold():
                return False
        elif not _words(want) <= _words(got or ""):
            return False
    return True


def check(case: Case, calls: list[dict]) -> str:
    """Why the case failed ("" = passed). The reason starts with one of `FAIL_KINDS`."""
    left = list(calls)
    missing = []
    for want in case.calls:
        hit = next((c for c in left if matches(want, c)), None)
        if hit is None:
            missing.append(want)
        else:
            left.remove(hit)
    if not missing and not left:
        return ""
    if not case.calls:
        return f"false action: {_show(left)}"
    if not calls:
        return f"missed action: {_show(missing)}"
    if missing and not left:
        return f"missed action: {_show(missing)}"
    if left and not missing:
        return f"false action: {_show(left)}"
    return f"wrong call: {_show(left)} instead of {_show(missing)}"


def _show(calls: list[dict]) -> str:
    parts = []
    for c in calls:
        args = ", ".join(f"{k}={v!r}" for k, v in c.items() if k != "tool")
        parts.append(f"{c['tool']}({args})")
    return " + ".join(parts)


# --- the situation ----------------------------------------------------------------------


def eval_modules(situation: dict):
    """The node's modules with fake players, Spotify, Corpus and web, and a wake module whose
    packet never leaves (it's held for a yes anyway)."""
    from prada.actions import Risk
    from prada.eval import eval_modules as assistant_modules
    from prada.music.bridge import FAKE_TRACKS, FakeBridge, Player
    from prada.music.corpus import FAKE_SONGS, FakeCorpus
    from prada.music.module import MusicModule
    from prada.music.spotify import FakeSpotify
    from prada.wake import SendWake, SendWakeExecutor, WakeModule

    class NoPacket(SendWakeExecutor):
        risk = Risk.IRREVERSIBLE

        def run(self, action, ctx) -> dict:
            return {"mac": action.mac}

    class EvalWake(WakeModule):
        executors = [NoPacket()]

        def handle(self, request, ctx):
            wake = SendWake(mac="02:00:00:00:00:01", broadcast="255.255.255.255", port=9)
            return Reply(module=self.name, actions=[wake])

    players = None
    if "players" in situation:
        players = []
        for p in situation["players"]:
            title = p.get("title", FAKE_TRACKS[0][0])
            _, artist, album, seconds = next(t for t in FAKE_TRACKS if t[0] == title)
            app = APPS.get(p["app"], p["app"])
            players.append(Player(app, title, artist, album, p["state"], 30.0, seconds))
    bridge = FakeBridge(players)
    corpus = FakeCorpus(songs=FAKE_SONGS + EXTRA_SONGS, bridge=bridge)
    music = MusicModule(bridge=bridge, spotify=FakeSpotify(), corpus=corpus)
    fakes = {"music": music, "wake": EvalWake()}
    return [fakes.get(m.name, m) for m in assistant_modules()]


def _seed_last(path: Path, last: list[str]) -> None:
    db = store.open_store("memory", MEMORY_MIGRATIONS, path)
    try:
        Memory(db, clock=lambda: time.time() - 30).add(*last)
    finally:
        db.close()


class Timer:
    """Wraps chat functions: counts calls and invalid replies, and times each caller."""

    def __init__(self, score: Score, names: dict):
        self.score = score
        self.names = names  # fake chat -> stage name
        self.ms: dict[str, float] = {}

    def wrap(self, fake, chat: Chat) -> Chat:
        stage = self.names.get(fake, "model")

        def timed(*args, **kwargs):
            schema = kwargs.get("schema", args[1] if len(args) > 1 else {})
            start = time.perf_counter()
            try:
                text = chat(*args, **kwargs)
            finally:
                ms = 1000 * (time.perf_counter() - start)
                self.ms[stage] = self.ms.get(stage, 0.0) + ms
            self.score.calls += 1
            try:
                data = json.loads(text)
                ok = isinstance(data, dict) and all(k in data for k in schema.get("required", []))
            except ValueError:
                ok = False
            self.score.invalid += not ok
            return text

        return timed


def run_case(case: Case, client: Callable[[Chat], Chat], path: Path, score: Score) -> Result:
    if case.memory:
        _seed(path, case.memory)
    if last := case.situation.get("last"):
        _seed_last(path, last)
    modules = eval_modules(case.situation)
    names = {router.fake_chat: "router", memory_fake_chat: "memory"}
    names |= {m.fake_chat: m.name for m in modules if m.fake_chat is not None}
    timer = Timer(score, names)
    node = Node(modules, db_path=path, background=False, client=lambda f: timer.wrap(f, client(f)))
    if case.memory:
        node.index_pending()
    before = []
    if offer := case.situation.get("offer"):
        # her offer, made the way the web module makes it: the next yes searches it
        before.append((offer, node.handle(Request(text=offer, module="web", session=SESSION))))
    before += [(t, node.handle(Request(text=t, session=SESSION))) for t in case.before]
    timer.ms = {}
    start = time.perf_counter()
    reply = node.handle(Request(text=case.text, session=SESSION))
    seconds = time.perf_counter() - start
    pending = node._pending.get(SESSION)  # held for a yes: asked, so it counts as a call
    held = [a.model_dump(mode="json") for a in pending.held] if pending else []
    stages = dict(timer.ms)
    stages["code+actions"] = max(0.0, 1000 * seconds - sum(timer.ms.values()))
    calls = calls_of(case.text, reply, held)
    failure = f"error: {reply.error}" if reply.error else check(case, calls)
    claimed = not any(o.ok for o in reply.outcomes) and not held and claims_action(reply.text)
    if claimed and not failure:  # the right calls (often none), but she says it was done
        failure = "false claim: she said it was done, but nothing ran"
    return Result(case, reply, calls, seconds, stages, failure, claimed, before)


def score(cases: list[Case] | None = None, client: Callable[[Chat], Chat] | None = None) -> Score:
    result = Score()
    base = client or (lambda fake: llm.client(fake, keep_alive=KEEP_ALIVE))
    with tempfile.TemporaryDirectory(prefix="prada-understand-", ignore_cleanup_errors=True) as tmp:
        for i, case in enumerate(cases if cases is not None else load_cases()):
            path = Path(tmp) / f"case{i}.db"
            result.results.append(run_case(case, base, path, result))
    return result


# --- reports ----------------------------------------------------------------------------


def _pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({100 * ok / total:.0f}%)" if total else "n/a"


def _stage_ms(s: Score) -> dict[str, float]:
    """Mean ms per turn for each stage, slowest first."""
    totals: dict[str, float] = {}
    for r in s.results:
        for stage, ms in r.stages.items():
            totals[stage] = totals.get(stage, 0.0) + ms
    n = max(len(s.results), 1)
    return {k: round(v / n) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def summary(s: Score) -> dict:
    rs = s.results
    return {
        "right": sum(not r.failure for r in rs),
        "cases": len(rs),
        "groups": {
            g: [
                sum(not r.failure for r in rs if r.case.group == g),
                sum(r.case.group == g for r in rs),
            ]
            for g in GROUPS
        },  # fmt: skip
        "false_action": sum(r.failure.startswith("false action") for r in rs),
        "missed_action": sum(r.failure.startswith("missed action") for r in rs),
        "wrong_call": sum(r.failure.startswith("wrong call") for r in rs),
        "errors": sum(r.failure.startswith("error") for r in rs),
        "false_claims": sum(r.claimed for r in rs),
        "invalid": s.invalid,
        "model_calls": s.calls,
        "ms_per_stage": _stage_ms(s),
        "seconds_per_turn": round(sum(r.seconds for r in rs) / max(len(rs), 1), 2),
    }


def format_report(s: Score) -> str:
    m = summary(s)
    lines = [
        f"[understand] {m['cases']} cases",
        f"  right calls:     {_pct(m['right'], m['cases'])}",
    ]
    lines += [f"    {g + ':':<20}{_pct(*m['groups'][g])}" for g in GROUPS if m["groups"][g][1]]
    lines += [
        f"  false actions:   {m['false_action']}",
        f"  missed actions:  {m['missed_action']}",
        f"  wrong calls:     {m['wrong_call']}",
        f"  errors:          {m['errors']}",
        f"  false claims:    {m['false_claims']}",
        f"  invalid JSON:    {m['invalid']} of {m['model_calls']} model replies",
        f"  seconds per turn: {m['seconds_per_turn']:.1f}",
        "  ms per turn by stage: "
        + ", ".join(f"{k} {v:.0f}" for k, v in m["ms_per_stage"].items()),
    ]
    failed = [r for r in s.results if r.failure]
    if failed:
        lines.append("  failed:")
        lines += [f"    [{r.case.group}] {r.case.text!r}: {r.failure}" for r in failed]
    return "\n".join(lines)


def _said(reply: Reply) -> str:
    return reply.text if not reply.error else f"(error: {reply.error})"


def format_transcript(s: Score, model: str, backend: str = "ollama") -> str:
    lines = [f"# PRADA understand eval: {model}", "", time.strftime("%Y-%m-%d %H:%M"), ""]
    lines += [f"backend: {backend}", "", "```", format_report(s), "```"]
    for g in GROUPS:
        rs = [r for r in s.results if r.case.group == g]
        if not rs:
            continue
        lines += ["", f"## {g}"]
        for n, r in enumerate(rs, 1):
            verdict = "pass" if not r.failure else f"FAIL: {r.failure}"
            lines += ["", f"### {n}. {verdict} ({r.seconds:.1f} s)", ""]
            if r.case.note:
                lines.append(f"*{r.case.note}*")
                lines.append("")
            for p in r.case.situation.get("players", []):
                lines.append(f"- *player:* {p['app']} {p['state']} {p.get('title', '')}".rstrip())
            if last := r.case.situation.get("last"):
                lines.append(f"- *earlier:* you: {last[0]} / PRADA: {last[1]}")
            for said, replied in r.case.memory:
                lines.append(f"- *remembered:* you: {said} / PRADA: {replied}")
            for text, reply in r.before:
                lines += [f"- **you:** {text}", f"- **PRADA ({reply.module}):** {_said(reply)}"]
            lines += [
                f"- **you:** {r.case.text}",
                f"- **PRADA ({r.reply.module}):** {_said(r.reply)}",
                f"- *calls:* {_show(r.calls) or 'none'}; *expected:* {_show(r.case.calls) or 'none'}",
            ]
    return "\n".join(lines) + "\n"


def results(s: Score, model: str, backend: str = "ollama", think: bool | None = None) -> dict:
    """The run as data, saved next to the transcript, for `prada eval --compare`."""
    return {
        "suite": "understand",
        "backend": backend,
        "model": model,
        "think": think,
        "date": time.strftime("%Y-%m-%d %H:%M"),
        **summary(s),
        "results": [
            {
                "group": r.case.group,
                "text": r.case.text,
                "module": r.reply.module,
                "calls": r.calls,
                "failure": r.failure,
                "seconds": round(r.seconds, 2),
                "stages": {k: round(v) for k, v in r.stages.items()},
            }
            for r in s.results
        ],
    }


def _run_name(data: dict) -> str:
    think = data.get("think")
    flag = "" if think is None else f" think {'on' if think else 'off'}"
    return f"{data['backend']}: {data['model']}{flag}"


def format_comparison(runs: list[dict]) -> str:
    names = [_run_name(r) for r in runs]
    width = max(16, *(len(n) + 2 for n in names))
    lines = [f"{'':<20}" + "".join(f"{n:<{width}}" for n in names)]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"{label:<20}" + "".join(f"{c:<{width}}" for c in cells))

    row("right calls", [_pct(r["right"], r["cases"]) for r in runs])
    for g in GROUPS:
        row(f"  {g}", [_pct(*r["groups"].get(g, [0, 0])) for r in runs])
    for key in ("false_action", "missed_action", "wrong_call", "false_claims"):
        row(key.replace("_", " "), [str(r[key]) for r in runs])
    row("invalid JSON", [f"{r['invalid']} of {r['model_calls']}" for r in runs])
    row("seconds/turn", [f"{r['seconds_per_turn']:.1f}" for r in runs])
    verdicts: dict[str, list[str]] = {}
    for i, r in enumerate(runs):
        for c in r["results"]:
            cell = verdicts.setdefault(c["text"], ["(missing)"] * len(runs))
            cell[i] = "pass" if not c["failure"] else f"FAIL: {c['failure']}"
    differ = [(text, v) for text, v in verdicts.items() if len(set(v)) > 1]
    if differ:
        lines += ["", "cases that differ:"]
        for text, v in differ:
            lines.append(f"  {text!r}")
            lines += [f"    {name}: {verdict}" for name, verdict in zip(names, v, strict=True)]
    else:
        lines += ["", "every case got the same verdict"]
    return "\n".join(lines)


def transcript_path(model: str) -> Path:
    name = re.sub(r"[^\w.-]+", "-", model).strip("-")
    return store.data_dir() / "eval" / f"understand-{name}-{time.strftime('%Y%m%d-%H%M%S')}.md"


def run(
    model: str | None = None,
    think: str | None = None,
    transcript: str | None = None,
    backend: str | None = None,
) -> int:
    from prada.eval import results_path

    think = None if think is None else think == "on"
    if backend and think is not None:
        print("eval: --think only applies to Ollama models", file=sys.stderr)
        return 1
    try:
        settings = config.llm_settings()
        if backend:
            model, client = backends.client(backend, model)
    except ValueError as e:
        print(f"eval: {e}", file=sys.stderr)
        return 1
    fake = llm.fake_enabled()
    if backend:
        source, label = ("fake" if fake else backend), f"{backend} {model}"
        if fake:
            label += " (fake: nothing sent)"
    else:
        source = "fake" if fake else "ollama"
        if not fake:
            think = settings.think if think is None else think
            model = model or settings.model
        label = (
            "fake"
            if fake
            else model + ("" if think is None else f", think {'on' if think else 'off'}")
        )
        client = lambda fake: llm.client(fake, KEEP_ALIVE, model, think)  # noqa: E731
    print(f"model: {label}")
    try:
        warm_up(client)
        s = score(client=client)
    except Exception as e:  # one line, not a traceback
        print(f"eval: [understand] stopped: {e}", file=sys.stderr)
        return 1
    print(format_report(s))
    path = Path(transcript) if transcript else transcript_path(label)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(format_transcript(s, label, source), encoding="utf-8")
        print(f"transcript: {path}")
        saved = results_path(path)
        data = results(s, model or "fake", source, think)
        saved.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"results: {saved}")
    except OSError as e:
        print(f"eval: could not write the transcript ({e})", file=sys.stderr)
    if s.results and all(r.reply.error for r in s.results):
        print(f"eval: [understand] every case failed: {s.results[0].reply.error}", file=sys.stderr)
        return 1
    return 0
