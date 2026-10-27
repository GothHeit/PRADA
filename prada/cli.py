import argparse
import sys

from prada import config, wol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prada", description="PRADA")
    sub = parser.add_subparsers(dest="command", required=True)

    wake = sub.add_parser("wake", help="wake the home PC with a Wake-on-LAN packet")
    wake.add_argument("--mac", help="MAC address of the PC")
    wake.add_argument("--broadcast", help="broadcast address (default 255.255.255.255)")
    wake.add_argument("--port", type=int, help="UDP port (default 9)")

    tutor = sub.add_parser("tutor", help="practice a language with the offline tutor")
    from tutor.cli import add_arguments

    add_arguments(tutor)

    args = parser.parse_args(argv)
    if args.command == "wake":
        try:
            s = config.wake_settings(args.mac, args.broadcast, args.port)
            wol.send_magic_packet(s["mac"], s["broadcast"], s["port"])
        except (ValueError, OSError) as e:
            print(f"prada: {e}", file=sys.stderr)
            return 1
        print(f"Magic packet sent to {s['mac']} via {s['broadcast']}:{s['port']}")
    else:
        from tutor.cli import run

        run(args.lang)
    return 0


if __name__ == "__main__":
    sys.exit(main())
