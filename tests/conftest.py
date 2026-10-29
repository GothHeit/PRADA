import os

import pytest

# prada.persona reads the persona file on import: tests always get the default one.
os.environ["PRADA_PERSONA"] = os.path.join(os.devnull, "persona.toml")


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """Keep tests off the user's real database, config and running node."""
    monkeypatch.setenv("PRADA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PRADA_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("PRADA_NODE", "local")
    monkeypatch.delenv("PRADA_LLM_MODEL", raising=False)
    monkeypatch.delenv("PRADA_LLM_THINK", raising=False)
