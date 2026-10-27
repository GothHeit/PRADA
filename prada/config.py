"""PRADA settings: command line > environment > config file.

The config file is TOML at $PRADA_CONFIG, or ~/.config/prada/config.toml:

    [wake]
    mac = "aa:bb:cc:dd:ee:ff"
    broadcast = "192.168.1.255"
    port = 9

    [node]
    port = 7723
"""

import os
import tomllib
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
