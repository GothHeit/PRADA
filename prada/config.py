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
    piper_voice: str = "en_US-amy-medium"  # Piper voice name or path to its .onnx
    # Voices for sentences in other languages, e.g. {"ru": "ru_RU-irina-medium"} ([voice.voices]).
    voices: dict[str, str] = field(default_factory=dict)
    input_device: str | int | None = None  # sounddevice name or index; None = default
    output_device: str | int | None = None
    silence: float = 0.8  # seconds of quiet that end what you're saying


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
    words = file.get("wake_words", VoiceSettings.wake_words)
    if isinstance(words, str):
        words = [words]
    if not words or not all(isinstance(w, str) and w.strip() for w in words):
        raise ValueError(f"[voice] wake_words in {config_path()} must be a list of names or paths")
    try:
        return VoiceSettings(
            **{
                **file,
                "wake_words": tuple(words),
                "threshold": float(file.get("threshold", VoiceSettings.threshold)),
                "silence": float(file.get("silence", VoiceSettings.silence)),
            }
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"bad [voice] setting in {config_path()}: {e}") from e
