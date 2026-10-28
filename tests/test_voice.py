import io

import pytest

from prada import config
from prada.cli import main
from prada.client import NodeError
from prada.module import Reply
from prada.voice import loop
from prada.voice.devices import _wake_model, _whisper_model, clean_transcript
from prada.voice.fake import TextEars


class Recorder:
    def __init__(self):
        self.said = []
        self.chimes = 0

    def chime(self):
        self.chimes += 1

    def say(self, text):
        self.said.append(text)


class Session:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def send(self, text, module=None, **options):
        self.sent.append(text)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_loop_speaks_replies_and_skips_silence():
    session = Session(Reply(text="Hi!"), NodeError("down"), Reply(error="boom"))
    mouth = Recorder()
    ears = TextEars(io.StringIO("hello\n\nare you there\nbreak\n"))
    assert loop.run(session, ears, mouth) == 0
    assert session.sent == ["hello", "are you there", "break"]
    assert mouth.said == ["Hi!", loop.CANT_REACH, loop.WENT_WRONG]
    assert mouth.chimes == 4  # every wake word gets a chime, even if nothing follows


def test_loop_once_stops_after_one_reply():
    session = Session(Reply(text="one"), Reply(text="two"))
    mouth = Recorder()
    loop.run(session, TextEars(io.StringIO("a\nb\n")), mouth, once=True)
    assert mouth.said == ["one"]


def frames(seconds):
    return round(seconds / loop.FRAME_SECONDS)


def test_endpointer_ends_after_silence():
    end = loop.Endpointer(noise=100, silence=0.8)
    assert end.level == 300
    assert not any(end.push(50) for _ in range(5))  # waiting for speech
    assert not any(end.push(2000) for _ in range(frames(1)))  # speaking
    quiet = [end.push(100) for _ in range(frames(0.8))]
    assert quiet[-1] and not any(quiet[:-1]) and end.heard


def test_endpointer_gives_up_without_speech_and_caps_length():
    end = loop.Endpointer(noise=1000, wait=1)
    assert end.level == 3000
    results = [end.push(2500) for _ in range(frames(1))]
    assert results[-1] and not end.heard
    end = loop.Endpointer(noise=100, longest=2)
    assert [end.push(5000) for _ in range(frames(2))][-1]


def test_noise_tracks_quiet_fast_and_loud_slowly():
    assert loop.track_noise(200, 50) == 50
    assert loop.track_noise(200, 5000) == pytest.approx(201)
    assert loop.track_noise(200, 0) == 1


def test_clean_transcript():
    assert clean_transcript(" [BLANK_AUDIO] ") == ""
    assert clean_transcript(" (music) What time is it?  ") == "What time is it?"


def test_model_paths(monkeypatch, tmp_path):
    assert _wake_model("hey_jarvis") == "hey_jarvis"
    with pytest.raises(ValueError, match="must be .onnx"):
        _wake_model(str(tmp_path / "wake_up_prada.tflite"))
    with pytest.raises(ValueError, match="wake word model not found"):
        _wake_model(str(tmp_path / "wake_up_prada.onnx"))
    with pytest.raises(ValueError, match="prada voice setup"):
        _whisper_model("small")
    model = tmp_path / "data" / "voice" / "ggml-small.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")
    assert _whisper_model("small") == model
    own = model.parent / "wake_up_prada.onnx"  # a bare name finds a model trained here
    own.write_bytes(b"")
    assert _wake_model("wake_up_prada") == str(own)


def test_voice_settings(tmp_path):
    path = tmp_path / "config.toml"
    assert config.voice_settings() == config.VoiceSettings()
    path.write_text('[voice]\nwake_words = ["hey_jarvis", "~/w.onnx"]\nthreshold = 0.7\n')
    s = config.voice_settings()
    assert s.wake_words == ("hey_jarvis", "~/w.onnx") and s.threshold == 0.7
    path.write_text('[voice]\nwake_words = "alexa"\n[voice.voices]\nru = "ru_RU-irina-medium"\n')
    s = config.voice_settings()
    assert s.wake_words == ("alexa",) and s.voices == {"ru": "ru_RU-irina-medium"}
    for bad in ["wake_words = []", "volume = 3", 'threshold = "loud"', "wake_words = [1]"]:
        path.write_text(f"[voice]\n{bad}\n")
        with pytest.raises(ValueError):
            config.voice_settings()


def test_cli_voice_runs_on_text_fakes(monkeypatch, capsys):
    monkeypatch.setenv("PRADA_FAKE_LLM", "1")
    monkeypatch.setattr("sys.stdin", io.StringIO("what time is it\n"))
    assert main(["voice"]) == 0
    out = capsys.readouterr()
    assert "(fake assistant) what time is it" in out.out
    assert "you: what time is it" in out.err


def test_cli_voice_reports_bad_config(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("PRADA_FAKE_LLM", raising=False)
    (tmp_path / "config.toml").write_text("[voice]\nwake_words = []\n")
    assert main(["voice"]) == 1
    assert "wake_words" in capsys.readouterr().err


def test_split_by_language():
    text = 'Almost: say "книгу". Что ты любишь читать? Nice. 本を読みます。何を読みますか？'
    assert loop.split_by_language(text, {"ru", "ja"}) == [
        ("ru", 'Almost: say "книгу". Что ты любишь читать?'),
        (None, "Nice."),
        ("ja", "本を読みます。 何を読みますか？"),
    ]
    assert loop.split_by_language("Что? Hi.", set()) == [(None, "Что? Hi.")]
    assert loop.split_by_language("  ", {"ru"}) == []


def test_voice_switches_off_onnxruntime_telemetry():
    import os

    import prada.voice  # noqa: F401

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"


def test_no_telemetry_on_windows_either(monkeypatch):
    import sys
    import types

    from prada.voice.devices import _no_telemetry

    calls = []
    ort = types.SimpleNamespace(disable_telemetry_events=lambda: calls.append(1))
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    _no_telemetry()
    assert calls == [1]
