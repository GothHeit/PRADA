import argparse
import dataclasses
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
    voice.add_argument(
        "--verbose", action="store_true", help="log what the mic and speakers are doing"
    )
    voice_sub = voice.add_subparsers(dest="voice_command")
    setup = voice_sub.add_parser("setup", help="download the wake word, speech and voice models")
    setup.add_argument("--tts", choices=config.TTS_ENGINES, help="voice engine to fetch too")
    voice_sub.add_parser("devices", help="list microphones and speakers")
    trial = voice_sub.add_parser("try", help="hear her voice say test lines, timed")
    trial.add_argument("text", nargs="*", help="what to say (default: the listening test)")
    trial.add_argument("--tts", choices=config.TTS_ENGINES, help="engine (default: [voice] tts)")
    design = voice_sub.add_parser("design", help="make a Qwen3-TTS voice from a description")
    design.add_argument("description", help='e.g. "a warm, playful young woman, soft voice"')
    design.add_argument("--name", default="her", help="clip name (default: her)")
    design.add_argument("--takes", type=int, default=3, help="clips to choose from (default 3)")
    train = voice_sub.add_parser("train", help="train your own wake word model (Linux/WSL)")
    train.add_argument("phrase", nargs="?", help='what you will say, e.g. "wake up prada"')
    train.add_argument("--setup", action="store_true", help="download the training data once")
    train.add_argument("--name", help="model name (default: the phrase, e.g. wake_up_prada)")
    train.add_argument("--samples", type=int, help="synthetic clips of the phrase (default 25000)")
    train.add_argument("--steps", type=int, help="training steps (default 50000)")
    train.add_argument("--out", help="folder or .onnx path for the model (default: voice models)")

    music = sub.add_parser(
        "music", help="show what Windows' media controls see (Spotify, Corpus, ...)"
    )
    music_sub = music.add_subparsers(dest="music_command")
    music_sub.add_parser("login", help="log in to Spotify once, so she can play songs by name")
    music_sub.add_parser("spotify", help="show the Spotify login, devices and what plays")
    music_sub.add_parser("import", help="read the music catalog from Corpus now")

    mem = sub.add_parser(
        "memory", help="developer tool: read and fix what she remembers (facts by default)"
    )
    mem.add_argument("--all", action="store_true", help="also list facts that were replaced")
    mem_sub = mem.add_subparsers(dest="memory_command")
    find = mem_sub.add_parser("search", help="facts, summaries and messages containing TEXT")
    find.add_argument("text")
    edit = mem_sub.add_parser("edit", help="change what fact ID says (its quote stays)")
    edit.add_argument("id", type=int)
    edit.add_argument("text")
    forget = mem_sub.add_parser("forget", help="delete fact ID and the versions it replaced")
    forget.add_argument("id", type=int)
    fix = mem_sub.add_parser(
        "fix", help="replace a misheard word in the last few conversations, as by voice"
    )
    fix.add_argument("wrong")
    fix.add_argument("right")

    tr = sub.add_parser(
        "trace", help="show her last turns: what she heard, why, what ran, how long"
    )
    tr.add_argument("--last", type=int, default=10, help="how many turns (default 10)")
    tr.add_argument("--case", type=int, metavar="N", help="print turn N as an eval case line")

    persona = sub.add_parser("persona", help="show who PRADA is and where her persona file is")
    persona_sub = persona.add_subparsers(dest="persona_command")
    init = persona_sub.add_parser("init", help="write a persona file to edit (outside the repo)")
    init.add_argument("--force", action="store_true", help="overwrite an existing persona file")

    tutor = sub.add_parser("tutor", help="practice a language with the offline tutor")
    from tutor.cli import add_arguments

    add_arguments(tutor)
    tutor_sub = tutor.add_subparsers(dest="tutor_command")
    stats = tutor_sub.add_parser("stats", help="show your logged practice, errors by category")
    stats.add_argument("--lang", dest="stats_lang", choices=sorted(LANGUAGES), help="one language")

    evaluate = sub.add_parser("eval", help="score a model as tutor or assistant on fixed cases")
    from tutor.eval import add_arguments as add_eval_arguments

    add_eval_arguments(evaluate)
    evaluate.add_argument(
        "--suite",
        choices=["tutor", "assistant"],
        default="tutor",
        help="tutor: labeled learner sentences (--lang); assistant: routing, honesty, "
        "search offers and memory, plus a transcript of every exchange",
    )
    evaluate.add_argument("--transcript", help="assistant suite: where to write the transcript")
    evaluate.add_argument(
        "--compare",
        nargs="+",
        metavar="JSON",
        help="assistant suite: show saved runs (the .json next to each transcript) side by side",
    )

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
    elif args.command == "memory":
        return _memory(args)
    elif args.command == "persona":
        return _persona(args)
    elif args.command == "trace":
        return _trace(args)
    elif args.command == "eval" and args.compare:
        from prada.eval import compare

        return compare(args.compare)
    elif args.command == "music" and args.music_command == "import":
        return _music_import()
    elif args.command == "music" and args.music_command:
        return _spotify(args.music_command)
    elif args.command == "music":
        return _music()
    elif args.command == "eval" and args.suite == "assistant":
        from prada.eval import run as run_assistant_eval

        return run_assistant_eval(args.model, args.think, args.transcript, args.backend)
    elif args.command == "eval":
        from tutor.eval import run as run_eval

        return run_eval(args.lang, args.model, args.think, args.backend)
    elif args.tutor_command == "stats":
        from tutor.cli import show_stats

        show_stats(args.stats_lang or args.lang)
    else:
        from tutor.cli import run

        return run(args.lang or "ru")
    return 0


def _music() -> int:
    from prada import llm
    from prada.music.bridge import BridgeError, FakeBridge, WindowsBridge

    bridge = FakeBridge() if llm.fake_enabled() else WindowsBridge()
    try:
        status = bridge.status()
    except BridgeError as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    finally:
        if isinstance(bridge, WindowsBridge):
            bridge.close()
    if not status.players:
        print("No player is showing anything to Windows.")
    for p in status.players:
        mark = "*" if p.app == status.current else " "
        where = f"{p.position:.0f}/{p.duration:.0f} s" if p.duration else "no timeline"
        print(
            f"{mark} {p.name} [{p.app}] {p.state}: {p.describe()} ({p.album or 'no album'}; {where})"
        )
    return 0


def _spotify(command: str) -> int:
    from prada.music import spotify

    try:
        api = spotify.SpotifyApi()
        if command == "login":
            spotify.login(api)
            print(f"Logged in to Spotify; the token is in {api.token_path}.")
            return 0
        if not api.ready():
            print("Not logged in to Spotify: run `prada music login`.")
            return 1
        devices = api.devices()
        now = api.now()
    except (ValueError, OSError, spotify.SpotifyError) as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    print("Logged in to Spotify.")
    if not devices:
        print("No Spotify device is open.")
    for d in devices:
        print(f"{'*' if d.active else ' '} {d.name} ({d.kind})")
    if now.title:
        state = "playing" if now.playing else "paused"
        print(f"{state}: {now.title} by {now.artists} ({now.album}) on {now.device}")
    return 0


def _music_import() -> int:
    from prada import llm
    from prada.music.bridge import FakeBridge
    from prada.music.corpus import CorpusCli, CorpusError, FakeCorpus
    from prada.music.library import Library

    library = Library(FakeCorpus(bridge=FakeBridge()) if llm.fake_enabled() else CorpusCli())
    try:
        songs = library.refresh(wait=True)
    except CorpusError as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    print(f"Read {songs} songs from Corpus.")
    return 0


def _memory(args) -> int:
    """`prada memory`: a developer's window into her memory (the node may be running)."""
    import sqlite3

    from prada.memory import MIGRATIONS, Memory
    from prada.store import open_store

    try:
        memory = Memory(open_store("memory", MIGRATIONS))
    except (sqlite3.Error, OSError, RuntimeError) as e:
        print(f"prada: can't open her memory ({e})", file=sys.stderr)
        return 1
    command = args.memory_command
    if command == "search":
        found = memory.find(args.text)
        for kind, r in found:
            if kind == "fact":
                print(_fact_line(r))
            elif kind == "summary":
                print(f"summary {r['id']} ({_date(r['ts'])}): {r['summary']}")
            else:
                print(f"message {r['id']} ({_date(r['ts'])}, {r['role']}): {r['content']}")
        if not found:
            print(f"Nothing in her memory contains {args.text!r}.")
    elif command == "edit":
        if not memory.edit_fact(args.id, args.text):
            print(f"prada: no fact {args.id}", file=sys.stderr)
            return 1
        print(f"Fact {args.id} now says: {' '.join(args.text.split())}")
    elif command == "forget":
        gone = memory.forget_fact(args.id)
        if not gone:
            print(f"prada: no fact {args.id}", file=sys.stderr)
            return 1
        print(f"Forgot fact {', '.join(map(str, gone))}.")
    elif command == "fix":
        from prada.memory import TooBroad

        try:
            fixed = memory.correct(args.wrong, args.right, args.right, manual=True)
        except TooBroad as e:
            print(f"prada: nothing changed, {e} (use search and edit)", file=sys.stderr)
            return 1
        print(f"Fixed {fixed} place(s): {args.right} instead of {args.wrong}.")
    else:
        facts = [f for f in memory.facts() if args.all or f["superseded_by"] is None]
        for f in facts:
            print(_fact_line(f))
        if not facts:
            print("She keeps no facts yet.")
    return 0


def _date(ts: float) -> str:
    import time

    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _fact_line(f) -> str:
    old = f" (replaced by {f['superseded_by']})" if f["superseded_by"] is not None else ""
    fake = " [fake]" if f["model"] == "fake" else ""
    return f'fact {f["id"]} ({_date(f["ts"])}){old}{fake}: {f["fact"]} — "{f["quote"]}"'


def _trace(args) -> int:
    """Reads the flight recorder (prada.trace) straight from the store, node running or not."""
    import sqlite3

    from prada import trace
    from prada.store import open_store

    try:
        log = trace.Trace(open_store("trace", trace.MIGRATIONS))
        if args.case is None:
            turns = log.last(args.last)
            if not turns:
                print("No turns recorded yet.")
            print("\n\n".join(trace.describe(t) for t in turns))
            return 0
        turn = log.get(args.case)
    except (sqlite3.Error, OSError, RuntimeError) as e:
        print(f"prada: can't read the trace ({e})", file=sys.stderr)
        return 1
    if turn is None:
        print(
            f"prada: no turn #{args.case} (only the last {trace.KEEP_DAYS} days are kept)",
            file=sys.stderr,
        )
        return 1
    print(trace.case_line(turn, log.before(turn)))
    print(
        "Check `module` (or use `says`) before pasting: it is what she did, not what was right.",
        file=sys.stderr,
    )
    return 0


def _persona(args) -> int:
    from prada import persona

    path = persona.persona_path()
    if args.persona_command == "init":
        if path.exists() and not args.force:
            print(f"prada: {path} already exists (--force overwrites it)", file=sys.stderr)
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(persona.template(persona.CURRENT), encoding="utf-8")
        print(f"Persona written to {path}; edit it, then restart the node.")
        return 0
    try:
        current = persona.load(path)
    except ValueError as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    source = path if path.is_file() else f"the default (no {path})"
    print(f"Persona from {source}:\n")
    print(persona.render(current))
    print(f"\nTone: {current.tone}")
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
            if args.tts:
                settings = dataclasses.replace(settings, tts=args.tts)
            devices.setup(settings)
            return 0
        if args.voice_command == "devices":
            print(devices.list_devices())
            return 0
        if args.voice_command == "try":
            if args.tts:
                settings = dataclasses.replace(settings, tts=args.tts)
            devices.try_voice(settings, [" ".join(args.text)] if args.text else None)
            return 0
        if args.voice_command == "design":
            return _design(args)
        # Always the node: the voice side (e.g. Windows) has no model of its own.
        session = client.RemoteSession(client.node_url(args.port))
        ears, mouth = devices.build(settings)
    except ImportError as e:
        tts = "qwen" if args.voice_command == "design" else settings.tts
        extra = "voice" if tts == "piper" else f"voice,{tts}"
        print(f'prada: voice needs `pip install "prada[{extra}]"` ({e})', file=sys.stderr)
        return 1
    except devices.errors() as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    from prada.voice import loop

    loop.VERBOSE = args.verbose
    words = ", ".join(settings.wake_words)
    print(f"PRADA is listening for: {words} (Ctrl+C to stop)", file=sys.stderr)
    try:
        return run(
            session,
            ears,
            mouth,
            args.once,
            settings.follow_up,
            settings.barge_in,
            settings.wake_words,
            settings.max_follow_ups,
        )
    except KeyboardInterrupt:
        return 0
    except devices.errors() as e:
        print(f"prada: {e}", file=sys.stderr)
        return 1
    finally:
        for device in (ears, mouth):
            try:
                device.close()
            except devices.errors():
                pass  # already gone with the error that ended the loop


def _design(args) -> int:
    from prada.voice import tts

    if args.takes < 1:
        print("prada: --takes must be at least 1", file=sys.stderr)
        return 1
    print("Designing her voice (first run downloads ~4 GB) ...", file=sys.stderr)
    for path in tts.design(args.description, args.name, args.takes):
        print(path)
    print(
        f"Listen, then rename the one you like to {args.name}.wav (and its .txt), or set\n"
        f'qwen_voice = "{args.name}-2" in [voice], with tts = "qwen".'
    )
    return 0


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
