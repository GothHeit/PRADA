import argparse
import sys

from prada import config, wol
from tutor.languages import LANGUAGES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prada", description="PRADA")
    sub = parser.add_subparsers(dest="command", required=True)

    wake = sub.add_parser("wake", help="wake the home PC with a Wake-on-LAN packet")
    wake.add_argument("--mac", help="MAC address of the PC")
    wake.add_argument("--broadcast", help="broadcast address (default 255.255.255.255)")
    wake.add_argument("--port", type=int, help="UDP port (default 9)")

    chat = sub.add_parser("chat", help="talk to PRADA; each message goes to the right module")
    chat.add_argument("--port", type=int, help="home node port (default 7723)")

    node = sub.add_parser("node", help="run the home node that hosts the modules (localhost)")
    node.add_argument("--port", type=int, help="port to listen on (default 7723)")

    tutor = sub.add_parser("tutor", help="practice a language with the offline tutor")
    from tutor.cli import add_arguments

    add_arguments(tutor)
    tutor_sub = tutor.add_subparsers(dest="tutor_command")
    stats = tutor_sub.add_parser("stats", help="show your logged practice, errors by category")
    stats.add_argument("--lang", dest="stats_lang", choices=sorted(LANGUAGES), help="one language")

    evaluate = sub.add_parser("eval", help="score the tutor on the labeled eval set")
    from tutor.eval import add_arguments as add_eval_arguments

    add_eval_arguments(evaluate)

    args = parser.parse_args(argv)
    if args.command == "wake":
        try:
            s = config.wake_settings(args.mac, args.broadcast, args.port)
            wol.send_magic_packet(s["mac"], s["broadcast"], s["port"])
        except (ValueError, OSError) as e:
            print(f"prada: {e}", file=sys.stderr)
            return 1
        print(f"Magic packet sent to {s['mac']} via {s['broadcast']}:{s['port']}")
    elif args.command == "chat":
        from prada import client

        return client.repl(client.connect(args.port))
    elif args.command == "node":
        from prada import server

        try:
            server.serve(config.node_port(args.port))
        except (ValueError, OSError) as e:
            print(f"prada: {e}", file=sys.stderr)
            return 1
    elif args.command == "eval":
        from tutor.eval import run as run_eval

        return run_eval(args.lang)
    elif args.tutor_command == "stats":
        from tutor.cli import show_stats

        show_stats(args.stats_lang or args.lang)
    else:
        from tutor.cli import run

        return run(args.lang or "ru")
    return 0


if __name__ == "__main__":
    sys.exit(main())
