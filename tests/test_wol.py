import socket

import pytest

from prada import wol


def test_parse_mac_accepts_common_formats():
    expected = bytes.fromhex("aabbccddeeff")
    for mac in ("aa:bb:cc:dd:ee:ff", "AA-BB-CC-DD-EE-FF", "aabb.ccdd.eeff", "aabbccddeeff"):
        assert wol.parse_mac(mac) == expected


@pytest.mark.parametrize("mac", ["", "aa:bb:cc", "zz:bb:cc:dd:ee:ff", "aa:bb:cc:dd:ee:ff:00"])
def test_parse_mac_rejects_bad_input(mac):
    with pytest.raises(ValueError):
        wol.parse_mac(mac)


def test_magic_packet_layout():
    packet = wol.magic_packet("aa:bb:cc:dd:ee:ff")
    assert len(packet) == 102
    assert packet[:6] == b"\xff" * 6
    assert packet[6:] == bytes.fromhex("aabbccddeeff") * 16


def test_send_magic_packet_over_udp():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(2)
    port = receiver.getsockname()[1]
    wol.send_magic_packet("aa:bb:cc:dd:ee:ff", "127.0.0.1", port)
    data, _ = receiver.recvfrom(512)
    receiver.close()
    assert data == wol.magic_packet("aa:bb:cc:dd:ee:ff")
