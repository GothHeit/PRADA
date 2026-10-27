"""Wake-on-LAN: build and send a magic packet using only the standard library."""

import re
import socket

_MAC_RE = re.compile(r"^[0-9a-f]{12}$")


def parse_mac(mac: str) -> bytes:
    """Parse 'aa:bb:cc:dd:ee:ff' (also '-' or '.' separated, or bare hex) into 6 bytes."""
    digits = re.sub(r"[:.\-]", "", mac.strip().lower())
    if not _MAC_RE.match(digits):
        raise ValueError(f"invalid MAC address: {mac!r}")
    return bytes.fromhex(digits)


def magic_packet(mac: str) -> bytes:
    return b"\xff" * 6 + parse_mac(mac) * 16


def send_magic_packet(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> None:
    packet = magic_packet(mac)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.sendto(packet, (broadcast, port))
