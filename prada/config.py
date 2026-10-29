"""PRADA settings: command line > environment > config file.

The config file is TOML at $PRADA_CONFIG, or ~/.config/prada/config.toml:

    [wake]
    mac = "aa:bb:cc:dd:ee:ff"
    broadcast = "192.168.1.255"
    port = 9

    [node]
    port = 7723

    [voice]
    wake_words = ["hey_jarvis", "~/prada/wake_up_prada.onnx"]

    [voice.voices]
    ru = "ru_RU-irina-medium"

    [web]
    github_token = "github_pat_..."
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def config_path() -> Path:
    return Path(os.environ.get("PRADA_CONFIG") or Path.home() / ".config" / "prada" / "config.toml")


def load_config() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def wake_settings(mac: str | None, broadcast: str | None, port: int | None) -> dict:
    """Resolve wake settings; raises ValueError if the MAC is not set anywhere."""
    file = load_config().get("wake", {})
    if not isinstance(file, dict):
        raise ValueError(f"[wake] in {config_path()} must be a table")
    env = os.environ
    mac = mac or env.get("PRADA_WAKE_MAC") or file.get("mac")
    if not mac:
        raise ValueError(
            f"no MAC address: pass --mac, set PRADA_WAKE_MAC, or add [wake] mac to {config_path()}"
        )
    return {
        "mac": mac,
        "broadcast": broadcast
        or env.get("PRADA_WAKE_BROADCAST")
        or file.get("broadcast", "255.255.255.255"),
        "port": port
        if port is not None
        else int(env.get("PRADA_WAKE_PORT") or file.get("port", 9)),
    }


DEFAULT_NODE_PORT = 7723  # P-R-A-D on a phone keypad


def node_port(port: int | None = None) -> int:
    """The home node's localhost port: CLI > PRADA_NODE_PORT > [node] port > 7723."""
    if port is not None:
        return port
    if env := os.environ.get("PRADA_NODE_PORT"):
        return int(env)
    file = load_config().get("node", {})
    if not isinstance(file, dict):
        raise ValueError(f"[node] in {config_path()} must be a table")
    return int(file.get("port", DEFAULT_NODE_PORT))


@dataclass(frozen=True)
class VoiceSettings:
    """The `[voice]` table: what the voice loop listens for and how it speaks."""

    # openWakeWord pretrained names (e.g. "hey_jarvis") or paths to custom .onnx models.
    wake_words: tuple[str, ...] = ("hey_jarvis",)
    threshold: float = 0.5  # wake word score (0-1) that counts as heard
    whisper_model: str = "small"  # whisper.cpp model name or path to a ggml .bin
    language: str = "auto"  # what you speak: "auto" or a code such as "en", "ru"
    # Text whisper reads before each request, e.g. names it should spell right ("Kitajima").
    whisper_prompt: str = ""
    piper_voice: str = "en_US-amy-medium"  # Piper voice name or path to its .onnx
    # Voices for sentences in other languages, e.g. {"ru": "ru_RU-irina-medium"} ([voice.voices]).
    voices: dict[str, str] = field(default_factory=dict)
    # How to spell words for the voice so she says them right ([voice.say_as]), e.g.
    # {"Kafka": "Kahfka"}; whole words, any case. Only what she speaks changes.
    say_as: dict[str, str] = field(default_factory=dict)
    input_device: str | int | None = None  # sounddevice name or index; None = default
    output_device: str | int | None = None
    silence: float = 0.8  # seconds of quiet that end what you're saying
    follow_up: float = 6.0  # seconds she keeps listening after a reply, no wake word (0 = off)


def voice_settings() -> VoiceSettings:
    """Resolve the voice settings from `[voice]` in the config file (defaults otherwise)."""
    file = load_config().get("voice", {})
    if not isinstance(file, dict):
        raise ValueError(f"[voice] in {config_path()} must be a table")
    fields = VoiceSettings.__dataclass_fields__
    if unknown := sorted(set(file) - set(fields)):
        raise ValueError(f"unknown [voice] setting(s) in {config_path()}: {', '.join(unknown)}")
    voices = file.get("voices", {})
    if not isinstance(voices, dict) or not all(isinstance(v, str) for v in voices.values()):
        raise ValueError(f"[voice.voices] in {config_path()} must map languages to voices")
    say_as = file.get("say_as", {})
    if not isinstance(say_as, dict) or not all(isinstance(v, str) for v in say_as.values()):
        raise ValueError(f"[voice.say_as] in {config_path()} must map words to spellings")
    words = file.get("wake_words", VoiceSettings.wake_words)
    if isinstance(words, str):
        words = [words]
    if not words or not all(isinstance(w, str) and w.strip() for w in words):
        raise ValueError(f"[voice] wake_words in {config_path()} must be a list of names or paths")
    try:
        follow_up = float(file.get("follow_up", VoiceSettings.follow_up))
        if follow_up < 0:
            raise ValueError("follow_up must not be negative")
        return VoiceSettings(
            **{
                **file,
                "wake_words": tuple(words),
                "threshold": float(file.get("threshold", VoiceSettings.threshold)),
                "silence": float(file.get("silence", VoiceSettings.silence)),
                "follow_up": follow_up,
            }
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad [voice] setting in {config_path()}: {e}") from e


@dataclass(frozen=True)
class WebSettings:
    """The `[web]` table: where she searches. Secrets stay in the file, never in logs."""

    searxng_url: str = "http://127.0.0.1:8888"  # your own SearXNG (scripts/install-searxng.sh)
    github_token: str = ""  # optional read-only token: 30 searches a minute instead of 10
    max_pages: int = 3  # most pages she reads per question (0-3)
    sources: tuple[str, ...] = ("web", "reddit", "github")  # the ones she may use


WEB_SOURCES = ("web", "reddit", "github")


def web_settings() -> WebSettings:
    """Resolve the web search settings from `[web]` in the config file (defaults otherwise)."""
    file = load_config().get("web", {})
    if not isinstance(file, dict):
        raise ValueError(f"[web] in {config_path()} must be a table")
    fields = WebSettings.__dataclass_fields__
    if unknown := sorted(set(file) - set(fields)):
        raise ValueError(f"unknown [web] setting(s) in {config_path()}: {', '.join(unknown)}")
    sources = file.get("sources", WebSettings.sources)
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list | tuple) or not set(sources) <= set(WEB_SOURCES):
        raise ValueError(
            f"[web] sources in {config_path()} must be some of {', '.join(WEB_SOURCES)}"
        )
    try:
        max_pages = int(file.get("max_pages", WebSettings.max_pages))
        if not 0 <= max_pages <= 3:
            raise ValueError("max_pages must be 0 to 3")
        return WebSettings(
            searxng_url=str(file.get("searxng_url", WebSettings.searxng_url)).rstrip("/"),
            github_token=str(file.get("github_token", "")),
            max_pages=max_pages,
            sources=tuple(sources),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad [web] setting in {config_path()}: {e}") from e
