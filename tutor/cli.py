import argparse
import os
import sqlite3
import sys

from tutor import memory
from tutor.languages import LANGUAGES
from tutor.orchestrator import run_turn


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        default=None,
        help="language (default: ru; for stats, all)",
    )


def show_stats(lang: str | None = None) -> None:
    print(memory.format_stats(memory.stats(memory.open_memory(), lang)))


def run(lang: str) -> None:
    if os.environ.get("PRADA_FAKE_LLM"):
        print("tutor: PRADA_FAKE_LLM is set, using the canned offline client", file=sys.stderr)
    history = []
    try:
        db = memory.open_memory()
    except (sqlite3.Error, OSError) as e:
        print(f"tutor: practice will not be logged ({e})", file=sys.stderr)
        db = None
    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt.strip():
            continue
        turn = run_turn(prompt, lang, history)
        if db:
            try:
                memory.log_exchange(db, lang, prompt, turn)
            except sqlite3.Error as e:
                print(f"tutor: could not log ({e})", file=sys.stderr)
                db = None
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": turn.spoken})
        print(turn.spoken)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tutor.cli", description="Offline language tutor"
    )
    add_arguments(parser)
    run(parser.parse_args(argv).lang or "ru")


if __name__ == "__main__":
    main()
