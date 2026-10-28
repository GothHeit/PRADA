import json
import sys
import types
from subprocess import CompletedProcess

import pytest

from prada.cli import main
from prada.voice import train


@pytest.fixture
def extra(monkeypatch, tmp_path):
    """Stand-ins for the wakeword extra's packages (the real ones are several GB)."""
    oww = types.ModuleType("openwakeword")
    oww.__file__ = str(tmp_path / "site" / "openwakeword" / "__init__.py")
    for name, module in {
        "openwakeword": oww,
        "openwakeword.train": types.ModuleType("openwakeword.train"),
        "piper": types.ModuleType("piper"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return tmp_path / "site" / "openwakeword"


def fake_assets():
    root = train.assets_dir()
    for d in train.DOWNLOADS:
        path = root / d.name
        if d.kind != "file":
            path.mkdir(parents=True)
            (path / "clip.wav").write_bytes(b"")
            path.with_name(path.name + ".done").write_text("1 clips\n")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")


def test_slug():
    assert train.slug("Wake up, Prada!") == "wake_up_prada"
    assert train.slug("  hey   PRADA 2 ") == "hey_prada_2"
    with pytest.raises(ValueError):
        train.slug("?!")


def test_missing_and_background_dirs():
    assert train.missing() == list(train.DOWNLOADS)
    fake_assets()
    assert train.missing() == []
    root = train.assets_dir() / "background"
    (root / "empty").mkdir()
    (root / "mine").mkdir()
    (root / "mine" / "room.wav").write_bytes(b"")
    assert [p.name for p in train.background_dirs()] == ["audioset", "mine"]


def test_train_needs_the_data(extra):
    with pytest.raises(ValueError, match="prada voice train --setup"):
        train.train("wake up prada")


def test_train_runs_openwakeword_and_saves_the_model(extra, tmp_path):
    fake_assets()
    calls = []

    def run(command, cwd):
        calls.append(command)
        if "--train_model" in command:
            (cwd / "wake_up_prada.onnx").write_bytes(b"model")
        return CompletedProcess(command, 0)

    path = train.train("Wake up, Prada", samples=500, steps=200, run=run)
    assert path == tmp_path / "data" / "voice" / "wake_up_prada.onnx"
    assert path.read_bytes() == b"model"
    work = train.assets_dir() / "runs" / "wake_up_prada"
    assert calls[0][1] == str(work / "run_trainer.py")
    compile((work / "run_trainer.py").read_text(), "run_trainer.py", "exec")
    assert calls[0][-1] == "--generate_clips"  # then the clips are resampled to 16 kHz
    assert calls[1][-2:] == ["--augment_clips", "--train_model"]
    config = json.loads((work / "config.yaml").read_text())
    assert config["target_phrase"] == ["Wake up, Prada"] and config["model_name"] == "wake_up_prada"
    assert config["n_samples"] == 500 and config["n_samples_val"] == 100 and config["steps"] == 200
    assert config["piper_sample_generator_path"] == str(work)
    assert config["background_paths"] == [str(train.assets_dir() / "background" / "audioset")]
    shim = (work / "generate_samples.py").read_text()
    assert str(train.assets_dir() / train.GENERATOR.name) in shim
    compile(shim, "generate_samples.py", "exec")
    assert (extra / "resources" / train.PHONEMIZER.name).is_file()  # no download mid-training

    out = train.train("wake up prada", out=str(tmp_path / "win"), samples=500, steps=200, run=run)
    assert out == tmp_path / "win" / "wake_up_prada.onnx"


def test_train_failure_says_how_to_resume(extra):
    fake_assets()
    with pytest.raises(RuntimeError, match="resume"):
        train.train("hey prada", run=lambda command, cwd: CompletedProcess(command, 1))


def test_train_ignores_the_tflite_conversion_error(extra, capsys):
    fake_assets()

    def run(command, cwd):  # the model is exported, then the TFLite copy fails
        if "--train_model" not in command:
            return CompletedProcess(command, 0)
        (cwd / "hey_prada.onnx").write_bytes(b"model")
        return CompletedProcess(command, 1)

    assert train.train("hey prada", run=run).read_bytes() == b"model"
    assert "TFLite" in capsys.readouterr().err


def test_cli_train(monkeypatch, capsys):
    assert main(["voice", "train"]) == 1
    assert "wake up prada" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "openwakeword.train", None)  # extra not installed
    assert main(["voice", "train", "wake up prada"]) == 1
    assert "prada[wakeword]" in capsys.readouterr().err


def test_drop_partial_features(tmp_path):
    (tmp_path / "positive_features_train.npy").write_bytes(b"")
    train._drop_partial_features(tmp_path)
    assert not any(tmp_path.iterdir())
    for name in train.FEATURE_FILES:
        (tmp_path / f"{name}.npy").write_bytes(b"")
    train._drop_partial_features(tmp_path)
    assert len(list(tmp_path.iterdir())) == 4


def test_to_16k_wav(tmp_path):
    np = pytest.importorskip("numpy")
    soundfile = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    stereo = np.zeros((32000, 2), dtype="float32")
    source = tmp_path / "in.wav"
    soundfile.write(source, stereo, 32000)
    train.to_16k_wav(source.read_bytes(), tmp_path / "out.wav")
    audio, rate = soundfile.read(tmp_path / "out.wav")
    assert rate == 16000 and audio.shape == (16000,)
    train.resample_folder(tmp_path)
    assert soundfile.info(source).samplerate == 16000


def test_fetch_checks_the_checksum(monkeypatch, tmp_path):
    import hashlib
    import io

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(train.urllib.request, "urlopen", lambda *a, **k: Response(b"data"))
    good = hashlib.sha256(b"data").hexdigest()
    train._fetch("https://x/y", tmp_path / "ok", good)
    assert (tmp_path / "ok").read_bytes() == b"data"
    with pytest.raises(ValueError, match="checksum mismatch"):
        train._fetch("https://x/y", tmp_path / "bad", "0" * 64)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()


def test_downloads_are_pinned():
    for d in (train.FEATURES, train.VALIDATION, train.PHONEMIZER, train.BACKGROUND):
        assert len(d.sha256) == 64 and "/resolve/main/" not in d.url


def test_fetch_accepts_a_part_that_is_already_whole(monkeypatch, tmp_path):
    import hashlib
    import urllib.error

    def refuse(*a, **k):
        raise urllib.error.HTTPError("https://x/y", 416, "Range Not Satisfiable", {}, None)

    monkeypatch.setattr(train.urllib.request, "urlopen", refuse)
    (tmp_path / "f.part").write_bytes(b"data")
    train._fetch("https://x/y", tmp_path / "f", hashlib.sha256(b"data").hexdigest())
    assert (tmp_path / "f").read_bytes() == b"data"
    (tmp_path / "g.part").write_bytes(b"data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        train._fetch("https://x/y", tmp_path / "g", "0" * 64)
    with pytest.raises(urllib.error.HTTPError):  # nothing downloaded yet: a real error
        train._fetch("https://x/y", tmp_path / "h")


def test_install_phonemizer_replaces_an_unchecked_copy(extra, monkeypatch, tmp_path):
    import hashlib

    ours = train.assets_dir() / train.PHONEMIZER.name
    ours.parent.mkdir(parents=True)
    ours.write_bytes(b"ours")
    monkeypatch.setattr(
        train,
        "PHONEMIZER",
        train.Download(train.PHONEMIZER.name, "u", "p", sha256=hashlib.sha256(b"ours").hexdigest()),
    )
    target = extra / "resources" / train.PHONEMIZER.name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"from the dead bucket")
    train._install_phonemizer()
    assert target.read_bytes() == b"ours"
