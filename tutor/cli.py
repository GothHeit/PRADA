import argparse
import sys

from tutor import memory
from tutor.languages import LANGUAGES


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        default=None,
        help="language (default: ru; for stats, all)",
    )


def show_stats(lang: str | None = None) -> None:
    print(memory.format_stats(memory.stats(memory.open_memory(), lang)))


def run(lang: str) -> int:
    """Practice `lang` with the tutor module, through the home node if one is running."""
    from prada import client

    return client.repl(client.connect(), "tutor", lang=lang)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tutor.cli", description="Offline language tutor"
    )
    add_arguments(parser)
    return run(parser.parse_args(argv).lang or "ru")


if __name__ == "__main__":
    sys.exit(main())
