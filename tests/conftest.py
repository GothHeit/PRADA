import pytest


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Keep tests off the user's real database, config and running node."""
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("PRADA_NODE", "local")
