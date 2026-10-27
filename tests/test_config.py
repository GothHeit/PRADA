import pytest

from prada import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("PRADA_WAKE_MAC", "PRADA_WAKE_BROADCAST", "PRADA_WAKE_PORT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "config.toml"))


def test_missing_mac_is_an_error():
    with pytest.raises(ValueError, match="no MAC"):
        config.wake_settings(None, None, None)


def test_defaults():
    s = config.wake_settings("aa:bb:cc:dd:ee:ff", None, None)
    assert s == {"mac": "aa:bb:cc:dd:ee:ff", "broadcast": "255.255.255.255", "port": 9}


def test_precedence_cli_over_env_over_file(monkeypatch, tmp_path):
    (tmp_path / "config.toml").write_text(
        '[wake]\nmac = "00:00:00:00:00:01"\nbroadcast = "10.0.0.255"\nport = 7\n'
    )
    assert config.wake_settings(None, None, None)["mac"] == "00:00:00:00:00:01"
    monkeypatch.setenv("PRADA_WAKE_MAC", "00:00:00:00:00:02")
    monkeypatch.setenv("PRADA_WAKE_PORT", "8")
    s = config.wake_settings(None, None, None)
    assert (s["mac"], s["broadcast"], s["port"]) == ("00:00:00:00:00:02", "10.0.0.255", 8)
    assert config.wake_settings("00:00:00:00:00:03", None, 9)["mac"] == "00:00:00:00:00:03"


def test_port_zero_is_kept(monkeypatch):
    monkeypatch.setenv("PRADA_WAKE_PORT", "7")
    assert config.wake_settings("aa:bb:cc:dd:ee:ff", None, 0)["port"] == 0
