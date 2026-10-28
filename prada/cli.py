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

    health = sub.add_parser("health", help="check that the home node answers (exit 1 if not)")
    health.add_argument("--port", type=int, help="home node port (default 7723)")
    health.add_argument("--wait", type=float, default=0, help="keep trying for this many seconds")

    voice = sub.add_parser("voice", help="talk to PRADA out loud: wake word, then speak")
    voice.add_argument("--port", type=int, help="home node port (default 7723)")
    voice.add_argument("--once", action="store_true", help="stop after one reply")
    voice_sub = voice.add_subparsers(dest="voice_command")
    voice_sub.add_parser("setup", help="download the wake word, speech and voice models")
    voice_sub.add_parser("devices", help="list microphones and speakers")
    train = voice_sub.add_parser("train", help="train your own wake word model (Linux/WSL)")
    train.add_argument("phrase", nargs="?", help='what you will say, e.g. "wake up prada"')
    train.add_argument("--setup", action="store_true", help="download the training data once")
    train.add_argument("--name", help="model name (default: the phrase, e.g. wake_up_prada)")
    train.add_argument("--samples", type=int, help="synthetic clips of the phrase (default 25000)")
    train.add_argument("--steps", type=int, help="training steps (default 50000)")
    train.add_argument("--out", help="folder or .onnx path for the model (default: voice models)")

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
    elif args.command == "health":
        from prada import client

        try:
            url = client.node_url(args.port)
        except ValueError as e:
            print(f"prada: {e}", file=sys.stderr)
            return 1
        modules = client.wait_for_node(url, args.wait)
        if modules is None:
            print(f"prada: no node answering at {url}", file=sys.stderr)
            return 1
        print(f"PRADA node up at {url} (modules: {', '.join(modules)})")
    elif args.command == "voice":
        return _voice(args)
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


def _voice(args) -> int:
    from prada import client, llm
    from prada.voice.loop import run

    if args.voice_command == "train":
        return _train(args)
    if llm.fake_enabled() and args.voice_command is None:
        from prada.voice.fake import TextEars, TextMouth

        return run(client.connect(args.port), TextEars(), TextMouth(), args.once)
    try:
        settings = config.voice_settings()
    except ValueError as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    from prada.voice import devices

    try:
        if args.voice_command == "setup":
            devices.setup(settings)
            return 0
        if args.voice_command == "devices":
            print(devices.list_devices())
            return 0
        # Always the node: the voice side (e.g. Windows) has no model of its own.
        session = client.RemoteSession(client.node_url(args.port))
        ears, mouth = devices.build(settings)
    except ImportError as e:
        print(f'prada: voice needs `pip install "prada[voice]"` ({e})', file=sys.stderr)
        return 1
    except devices.errors() as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    words = ", ".join(settings.wake_words)
    print(f"PRADA is listening for: {words} (Ctrl+C to stop)", file=sys.stderr)
    try:
        return run(session, ears, mouth, args.once)
    except KeyboardInterrupt:
        return 0
    except devices.errors() as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1


def _train(args) -> int:
    from prada.voice import train

    try:
        if args.setup:
            train.setup()
            if not args.phrase:
                return 0
        elif not args.phrase:
            print('prada: what should I train? e.g. "wake up prada"', file=sys.stderr)
            return 1
        options = {"samples": args.samples, "steps": args.steps}
        options = {k: v for k, v in options.items() if v is not None}
        path = train.train(args.phrase, args.name, out=args.out, **options)
    except ImportError as e:
        print(f'prada: training needs `pip install "prada[wakeword]"` ({e})', file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, OSError) as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("prada: stopped; run the same command to resume", file=sys.stderr)
        return 130
    print(f"Wake word model saved to {path}")
    print(f'Put it in the voice models folder and add "{path.stem}" to [voice] wake_words.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
