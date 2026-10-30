import sys
import types

import pytest

from prada import config
from prada.cli import main
from prada.voice import tts

np = pytest.importorskip("numpy")


class FakePiperVoice:
    def __init__(self, path):
        self.path = path

    @classmethod
    def load(cls, path):
        return cls(path)

    def synthesize(self, text, syn_config=None):
        self.config = syn_config
        yield types.SimpleNamespace(audio_int16_array=f"piper:{text}", sample_rate=22050)


class FakeKokoro:
    def __init__(self, model, voices):
        self.made = []

    def get_voices(self):
        return ["af_heart", "ff_siwis"]

    def create(self, text, voice, speed, lang):
        self.made.append((text, voice, lang))
        return np.zeros(10, dtype=np.float32), 24000


class FakeQwen:
    calls = []

    @classmethod
    def from_pretrained(cls, model, **kwargs):
        cls.loaded = (model, kwargs)
        return cls()

    def warmup(self):
        self.warm = True

    def generate_voice_clone(self, text, language, **kwargs):
        self.calls.append((text, language, kwargs))
        return [np.array([0.5, -2.0], dtype=np.float32)], 24000


@pytest.fixture
def engines_installed(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "piper",
        types.SimpleNamespace(PiperVoice=FakePiperVoice, SynthesisConfig=types.SimpleNamespace),
    )
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setitem(
        sys.modules, "faster_qwen3_tts", types.SimpleNamespace(FasterQwen3TTS=FakeQwen)
    )
    folder = tts.models_dir()
    folder.mkdir(parents=True)
    for name in ("en_US-amy-medium.onnx", "ru_RU-irina-medium.onnx", *tts.kokoro_files()):
        (folder / name).write_bytes(b"")
    return folder


def test_tts_settings(tmp_path):
    path = tmp_path / "config.toml"
    s = config.voice_settings()
    assert (s.tts, s.speed, s.qwen_voice) == ("piper", 1.0, "her")
    path.write_text('[voice]\ntts = "kokoro"\nspeed = 1.1\nkokoro_voice = "af_bella"\n')
    s = config.voice_settings()
    assert (s.tts, s.speed, s.kokoro_voice) == ("kokoro", 1.1, "af_bella")
    path.write_text("[voice]\nqwen_temperature = 0.5\n")
    assert config.voice_settings().qwen_temperature == 0.5
    for bad in ['tts = "elevenlabs"', "speed = 0", 'speed = "fast"', "qwen_temperature = 3"]:
        path.write_text(f"[voice]\n{bad}\n")
        with pytest.raises(ValueError):
            config.voice_settings()


def test_piper_is_the_default_and_speed_sets_length(engines_installed):
    s = config.VoiceSettings(speed=2.0, voices={"ru": "ru_RU-irina-medium"})
    main, voices = tts.engines(s)
    assert isinstance(main, tts.Piper) and set(voices) == {"ru"}
    assert list(main.speak("Hi.", None)) == [("piper:Hi.", 22050)]
    assert main.config.length_scale == 0.5


def test_kokoro_speaks_french_itself_and_leaves_russian_to_piper(engines_installed):
    s = config.VoiceSettings(tts="kokoro", voices={"ru": "ru_RU-irina-medium"})
    main, voices = tts.engines(s)
    assert isinstance(main, tts.Kokoro)
    assert voices["fr"] is main and isinstance(voices["ru"], tts.Piper) and "ja" not in voices
    chunks = list(main.speak("One. Two!", None)) + list(main.speak("Bonjour.", "fr"))
    assert all(samples.dtype == np.int16 and rate == 24000 for samples, rate in chunks)
    assert main.model.made == [
        ("One.", "af_heart", "en-us"),
        ("Two!", "af_heart", "en-us"),
        ("Bonjour.", "ff_siwis", "fr-fr"),
    ]


def test_qwen_clones_her_clip_in_every_language(engines_installed):
    (engines_installed / "her.wav").write_bytes(b"")
    (engines_installed / "her.txt").write_text("Hi, it's me.\n")
    main, voices = tts.engines(config.VoiceSettings(tts="qwen"))
    assert isinstance(main, tts.Qwen) and voices["ja"] is voices["ru"] is main
    assert FakeQwen.loaded[1] == {"local_files_only": True}  # never downloads at runtime
    assert main.model.warm and FakeQwen.calls[0][0] == "Hi."  # a silent first line at load
    [(samples, rate)] = main.speak("Hello.", None)
    assert samples.tolist() == [16383, -32767] and rate == 24000
    assert len(list(main.speak("こんにちは。元気？", "ja"))) == 2  # one sentence at a time
    (_, lang, first), (_, ja, second) = FakeQwen.calls[-3:-1]
    assert first["ref_text"] == "Hi, it's me." and not first["xvec_only"]
    assert (lang, ja, second["xvec_only"]) == ("auto", "Japanese", True)
    assert first["temperature"] == 0.8
    assert first["max_new_tokens"] == tts.QWEN_MIN_STEPS + tts.QWEN_STEPS_PER_CHAR * 6


def test_a_voice_that_wont_load_falls_back_to_piper(engines_installed, capsys):
    main, _ = tts.engines(config.VoiceSettings(tts="qwen"))  # no her.wav yet
    assert isinstance(main, tts.Piper)
    assert "qwen didn't load, using Piper" in capsys.readouterr().err
    sys.modules["kokoro_onnx"] = None  # not installed: importing it raises ImportError
    main, voices = tts.engines(config.VoiceSettings(tts="kokoro"))
    assert isinstance(main, tts.Piper) and voices == {}
    main, _ = tts.engines(config.VoiceSettings(tts="kokoro", kokoro_voice="af_nobody"))
    assert isinstance(main, tts.Piper)


def test_mouth_gives_each_language_its_engine(engines_installed):
    from prada.voice.devices import SpeakerMouth

    mouth = SpeakerMouth.__new__(SpeakerMouth)
    mouth.voice, mouth.voices = tts.engines(
        config.VoiceSettings(tts="kokoro", voices={"ru": "ru_RU-irina-medium"})
    )
    mouth.say_as = {}
    played = []
    mouth._play = lambda samples, rate, barge=None: played.append(rate) or True
    mouth.speaker = types.SimpleNamespace(drain=lambda: None)
    assert mouth.say("Hello there. Я читаю книгу. Je suis là.")
    assert played == [24000, 22050, 24000]


def test_listening_test_covers_her_languages():
    from tutor.languages import detect_language

    found = {detect_language(line) for line in tts.LISTENING_TEST}
    assert {"ja", "fr", "ru", None} <= found


def test_design_needs_a_take(capsys):
    assert main(["voice", "design", "warm", "--takes", "0"]) == 1
    assert "--takes" in capsys.readouterr().err


def test_try_voice_times_each_line(engines_installed, monkeypatch, capsys):
    from prada.voice import devices

    monkeypatch.setattr(devices, "_no_telemetry", lambda: None)
    monkeypatch.setattr(devices.Speaker, "play", lambda self, samples, rate, barge=None: True)
    monkeypatch.setattr(devices.Speaker, "drain", lambda self: None)
    devices.try_voice(config.VoiceSettings(tts="kokoro"), ["Hi there."])
    out = capsys.readouterr()
    assert "kokoro: loaded in" in out.err and out.out.strip().endswith("s  Hi there.")


def test_qwen_splits_sentences_only_where_they_end():
    split = [s for s in tts.QWEN_SENTENCES.split("Hmm... It costs 3.50 today. 元気？うん。") if s]
    assert split == ["Hmm...", "It costs 3.50 today.", "元気？", "うん。"]


def test_qwen_copies_only_the_timbre_of_a_clip_without_transcript(engines_installed):
    (engines_installed / "monika.wav").write_bytes(b"")
    main, _ = tts.engines(config.VoiceSettings(tts="qwen", qwen_voice="monika"))
    assert isinstance(main, tts.Qwen) and main.ref_text == ""
    list(main.speak("Hello.", None))
    assert FakeQwen.calls[-1][2]["xvec_only"]
