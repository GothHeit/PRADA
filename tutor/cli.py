import argparse

from tutor.languages import LANGUAGES
from tutor.orchestrator import run_turn


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang", choices=sorted(LANGUAGES), default="ru", help="language to practice (default: ru)"
    )


def run(lang: str) -> None:
    history = []
    while True:
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not prompt.strip():
            continue
        turn = run_turn(prompt, lang, history)
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": turn.spoken})
        print(turn.spoken)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tutor.cli", description="Offline language tutor"
    )
    add_arguments(parser)
    run(parser.parse_args(argv).lang)


if __name__ == "__main__":
    main()
