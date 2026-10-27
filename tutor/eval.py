"""Score the tutor on the labeled learner sentences in tutor/evalset/<lang>.jsonl.

Each line: {"text", "is_correct", and for wrong sentences "error_type" and "correction"}.
Run `prada eval` (or `python -m tutor.eval`) on a machine with Ollama to measure a prompt or
model change; with PRADA_FAKE_LLM=1 it only exercises the plumbing.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from tutor.languages import LANGUAGES, get_language
from tutor.orchestrator import run_turn

EVALSET_DIR = Path(__file__).parent / "evalset"


@dataclass
class Case:
    text: str
    is_correct: bool
    error_type: str | None = None
    correction: str | None = None


@dataclass
class Score:
    cases: int = 0
    failed: int = 0  # model reply was not valid JSON for the schema
    verdict_ok: int = 0  # is_correct matches
    wrong_cases: int = 0  # labeled-wrong sentences
    category_ok: int = 0  # wrong sentences with the expected error_type
    correction_ok: int = 0  # wrong sentences with the expected correction (normalized)
    format_ok: int = 0  # spoken ends with reply_target verbatim
    by_category: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0]))


def load_cases(lang: str) -> list[Case]:
    path = EVALSET_DIR / f"{get_language(lang).code}.jsonl"
    cases = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = Case(**json.loads(line))
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path.name}:{n}: bad eval case ({e})") from None
        if not case.is_correct and not (case.error_type and case.correction):
            raise ValueError(f"{path.name}:{n}: wrong sentence needs error_type and correction")
        cases.append(case)
    return cases


def normalize(text: str | None) -> str:
    """Casefold and drop punctuation/spacing so trivial formatting differences don't count."""
    return re.sub(r"[\W_]+", "", (text or "").casefold())


def score_language(lang: str, chat=None) -> Score:
    score = Score()
    for case in load_cases(lang):
        score.cases += 1
        try:
            turn = run_turn(case.text, lang, chat=chat)
        except (ValidationError, ValueError):
            score.failed += 1
            if not case.is_correct:
                score.wrong_cases += 1
                score.by_category[case.error_type][1] += 1
            continue
        score.verdict_ok += turn.is_correct == case.is_correct
        score.format_ok += turn.spoken.rstrip().endswith(turn.reply_target.strip())
        if not case.is_correct:
            score.wrong_cases += 1
            hit = turn.error_type is not None and turn.error_type.value == case.error_type
            score.category_ok += hit
            score.correction_ok += normalize(turn.correction) == normalize(case.correction)
            score.by_category[case.error_type][0] += hit
            score.by_category[case.error_type][1] += 1
    return score


def _pct(ok: int, total: int) -> str:
    return f"{ok}/{total} ({100 * ok / total:.0f}%)" if total else "n/a"


def format_report(lang: str, s: Score) -> str:
    lines = [
        f"[{lang}] {s.cases} sentences, {s.failed} unparseable replies",
        f"  verdict (correct vs wrong): {_pct(s.verdict_ok, s.cases)}",
        f"  error category:             {_pct(s.category_ok, s.wrong_cases)}",
        f"  exact correction:           {_pct(s.correction_ok, s.wrong_cases)}",
        f"  spoken ends with question:  {_pct(s.format_ok, s.cases)}",
    ]
    for cat in sorted(s.by_category):
        ok, total = s.by_category[cat]
        lines.append(f"    {cat}: {_pct(ok, total)}")
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        action="append",
        help="language to score (repeatable; default: all)",
    )


def run(langs: list[str] | None) -> None:
    for lang in langs or sorted(LANGUAGES):
        print(format_report(lang, score_language(lang)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tutor.eval", description="Score the tutor")
    add_arguments(parser)
    run(parser.parse_args(argv).lang)


if __name__ == "__main__":
    main()
