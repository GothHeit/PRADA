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


class FollowEars:
    """Wake words and follow-ups from a script: ("wake", [first, follow-up...]) per wake."""

    def __init__(self, *wakes):
        self.wakes = list(wakes)
        self.heard = []
        self.waits = []

    def wait_for_wake(self):
        if not self.wakes:
            return None
        self.heard = list(self.wakes.pop(0))
        return "hey"

    def listen(self, wait=None):
        self.waits.append(wait)
        return self.heard.pop(0) if self.heard else ""


def test_follow_up_keeps_listening_until_silence():
    session = Session(Reply(text="one"), Reply(text="two"), Reply(text="three"))
    mouth = Recorder()
    ears = FollowEars(["a", "b"], ["c"])
    loop.run(session, ears, mouth, follow_up=4.0)
    assert session.sent == ["a", "b", "c"]
    assert mouth.chimes == 2  # follow-ups need no wake word and no chime
    assert ears.waits == [None, 4.0, 4.0, None, 4.0]


def test_follow_ups_have_no_limit_at_zero():
    replies = [Reply(text="ok") for _ in range(30)]
    ears = FollowEars(["a"] + ["more"] * 29)
    loop.run(Session(*replies), ears, Recorder(), follow_up=4.0, max_follow_ups=0)
    assert ears.waits == [None] + [4.0] * 30  # every one answered, then silence


def test_follow_up_off_returns_to_the_wake_word():
    session = Session(Reply(text="one"))
    ears = FollowEars(["a", "b"])
    loop.run(session, ears, Recorder(), follow_up=0)
    assert session.sent == ["a"]
    assert ears.waits == [None]


def test_follow_up_respects_once():
    session = Session(Reply(text="one"), Reply(text="two"))
    ears = FollowEars(["a", "b"])
    loop.run(session, ears, Recorder(), once=True, follow_up=4.0)
    assert session.sent == ["a"]


def test_text_ears_have_no_follow_up():
    ears = TextEars(io.StringIO("hi\n"))
    assert ears.wait_for_wake() == "text"
    assert ears.listen() == "hi"
    assert ears.listen(5.0) == ""


def test_follow_up_setting(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    monkeypatch.setenv("PRADA_CONFIG", str(path))
    assert config.voice_settings().follow_up == 6.0
    assert config.voice_settings().max_follow_ups == 20
    path.write_text("[voice]\nfollow_up = 0\n")
    assert config.voice_settings().follow_up == 0
    path.write_text("[voice]\nfollow_up = -1\n")
    with pytest.raises(ValueError, match="follow_up"):
        config.voice_settings()
    path.write_text("[voice]\nmax_follow_ups = 0\n")
    assert config.voice_settings().max_follow_ups == 0
    for bad in ("-1", "2.5", "true"):
        path.write_text(f"[voice]\nmax_follow_ups = {bad}\n")
        with pytest.raises(ValueError, match="max_follow_ups"):
            config.voice_settings()


def test_chime_is_long_enough_to_hear_and_padded():
    np = pytest.importorskip("numpy")
    from prada.voice.devices import chime_samples

    tone = chime_samples()
    assert tone.dtype == np.int16
    assert tone.size >= 0.4 * loop.SAMPLE_RATE
    lead = int(0.1 * loop.SAMPLE_RATE)
    assert not tone[:lead].any() and np.abs(tone).max() > 10000


def test_speaker_keeps_streams_open(monkeypatch):
    import sys
    import types

    opened = []

    class Stream:
        latency = 0.0

        active = False

        def __init__(self, **kw):
            opened.append(kw)
            self.written = []

        def start(self):
            self.active = True

        def write(self, samples):
            self.written.append(samples)

    sd = types.SimpleNamespace(OutputStream=Stream)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    from prada.voice.devices import Speaker

    speaker = Speaker("out")
    for _ in range(3):
        speaker.play([0], 16000)
    speaker.play([0], 22050)
    assert [kw["samplerate"] for kw in opened] == [16000, 22050]
    assert speaker.streams[16000].written == [[0]] * 3


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
    assert [end.push(5000) for _ in range(frames(2))][-1] and end.too_long
    end = loop.Endpointer(noise=100, silence=0.8)
    end.push(5000)
    assert not any(end.push(50) for _ in range(frames(0.8) - 1)) and not end.too_long


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
    path.write_text('[voice]\nwhisper_prompt = "Kitajima"\n[voice.say_as]\nKafka = "Kahfka"\n')
    s = config.voice_settings()
    assert s.whisper_prompt == "Kitajima" and s.say_as == {"Kafka": "Kahfka"}
    for bad in [
        "say_as = 1",
        "wake_words = []",
        "volume = 3",
        'threshold = "loud"',
        "wake_words = [1]",
    ]:
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
    assert "[assistant] prada: (fake assistant) what time is it" in out.err


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


def test_speaker_drains_before_another_rate_and_closes(monkeypatch):
    import sys
    import types

    events = []

    class Stream:
        latency = 0.0

        active = False

        def __init__(self, samplerate, **kw):
            self.rate = samplerate

        def start(self):
            self.active = True

        def write(self, samples):
            events.append(("write", self.rate))

        def close(self):
            events.append(("close", self.rate))

    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(OutputStream=Stream))
    from prada.voice.devices import Speaker

    speaker = Speaker("out")
    monkeypatch.setattr(speaker, "drain", lambda: events.append(("drain", None)))
    speaker.play([0], 16000)
    speaker.play([0], 16000)
    speaker.play([0], 22050)
    speaker.close()
    assert events == [
        ("write", 16000),
        ("write", 16000),
        ("drain", None),
        ("write", 22050),
        ("close", 16000),
        ("close", 22050),
    ]
    assert speaker.streams == {}


def test_respell_swaps_whole_words_in_any_case():
    say_as = {"Kafka": "Kahfka", "PRADA": "Prah-da"}
    assert loop.respell("Hi kafka! I'm PRADA, not Kafkaesque.", say_as) == (
        "Hi Kahfka! I'm Prah-da, not Kafkaesque."
    )
    assert loop.respell("unchanged", {}) == "unchanged"
    assert loop.respell("I like C++ and Dr. Who.", {"C++": "C plus plus", "Dr.": "Doctor"}) == (
        "I like C plus plus and Doctor Who."
    )
    assert loop.respell("abc", {"": "x"}) == "abc"


def _fake_mic_ears(monkeypatch, *texts, language="en", guesses=()):
    """MicEars on a fake mic (a short burst of speech, then quiet) and a fake whisper that
    says `texts` in turn and guesses languages `guesses` ((code, likelihood) in turn)."""
    np = pytest.importorskip("numpy")
    import contextlib
    import sys
    import types

    from prada.voice import devices

    utils = types.ModuleType("pywhispercpp.utils")
    utils.redirect_stderr = lambda to=None: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "pywhispercpp", types.ModuleType("pywhispercpp"))
    monkeypatch.setitem(sys.modules, "pywhispercpp.utils", utils)

    class Stream:
        active, read_available = False, 0

        def __init__(self):
            self.frames = 0

        def start(self):
            self.active = True

        def stop(self):
            self.active = False

        def read(self, n):
            self.frames += 1
            loud = self.frames % 40 < 10  # a short burst, then quiet
            return np.full(n, 3000 if loud else 0, dtype=np.int16).tobytes(), False

    class Whisper:
        def __init__(self):
            self.texts, self.guesses, self.calls = list(texts), list(guesses), []

        def transcribe(self, audio, language=None, initial_prompt=None):
            self.calls.append((language, initial_prompt))
            return [types.SimpleNamespace(text=self.texts.pop(0))]

        def auto_detect_language(self, audio):
            return self.guesses.pop(0), {}

    ears = devices.MicEars.__new__(devices.MicEars)
    ears.stream, ears.noise, ears.silence, ears.language = Stream(), 100.0, 0.8, language
    ears.stt, ears.hints, ears.prompt, ears.cue, ears.last_language = (
        Whisper(),
        None,
        "",
        None,
        None,
    )
    return ears


def test_speech_without_words_is_reported_and_heard_again(monkeypatch, capsys):
    ears = _fake_mic_ears(monkeypatch, "[BLANK_AUDIO]", "what time is it")
    cues = []
    ears.cue = lambda: cues.append("chime")
    assert ears.listen(6.0) == "what time is it"
    assert "heard something but no words; listening again" in capsys.readouterr().err
    assert cues == ["chime"]  # she lets you know she didn't get it

    ears = _fake_mic_ears(monkeypatch, "", "(music)")
    assert ears.listen() == ""
    err = capsys.readouterr().err
    assert err.count("heard something but no words") == 2 and not ears.stream.active


def test_whisper_reads_the_hints_before_each_request(monkeypatch):
    class Node:
        def __init__(self):
            self.words = ["Monika", "Muse"]

        def vocabulary(self):
            if self.words is None:
                raise NodeError("node down")
            return self.words

    node = Node()
    ears = _fake_mic_ears(monkeypatch, "play space dementia", "again", "once more")
    ears.hints = loop.Hints(node, ["hey_jarvis"], "Kitajima")
    ears.hints.refresh()  # as the ears start
    ears.hints.asking.join()
    ears.listen()
    assert ears.stt.calls[-1] == ("en", "Muse, Monika, Hey Jarvis. Kitajima")
    ears.hints.asking.join()  # asked again in the background, used by the next request
    node.words = ["Ana"]
    ears.listen()
    assert ears.stt.calls[-1][1] == "Muse, Monika, Hey Jarvis. Kitajima"  # never waits
    ears.hints.asking.join()
    node.words = None
    ears.listen()
    assert ears.stt.calls[-1][1] == "Ana, Hey Jarvis. Kitajima"
    ears.hints.asking.join()
    assert ears.hints.words == ["Ana"]  # node down: the last list is kept


def test_whisper_reading_its_prompt_back_is_no_words(monkeypatch):
    ears = _fake_mic_ears(monkeypatch, "Muse, Toru Kitajima, Monika.", "Space Dementia",
                          "Monika, Hey Jarvis", "Muse, Monika", "play Muse, then Monika")  # fmt: skip
    ears.prompt = "Space Dementia, Muse, Toru Kitajima, Monika, Hey Jarvis. Kitajima"
    assert ears.listen() == "Space Dementia"  # an echo is heard again; one name is an answer
    assert ears.listen() == ""  # echoes only: no request
    assert ears.listen() == "play Muse, then Monika"
    assert not loop.echoes("Muse", ears.prompt)
    assert loop.echoes("muse,  MONIKA", ears.prompt)


def test_auto_language_falls_back_to_the_last_one_when_unsure(monkeypatch):
    guesses = [("en", 0.9), ("nl", 0.3), ("ru", 0.8)]
    ears = _fake_mic_ears(monkeypatch, "hello", "play muse", "привет", language="auto",
                          guesses=guesses)  # fmt: skip
    for _ in guesses:
        ears.listen()
    assert [lang for lang, _ in ears.stt.calls] == ["en", "en", "ru"]
    assert ears.last_language == "ru"


def test_auto_language_takes_the_guess_with_no_last_one():
    assert loop.pick_language("nl", 0.2, None) == "nl"
    assert loop.pick_language("nl", 0.2, "pt") == "pt"
    assert loop.pick_language("ja", 0.7, "pt") == "ja"


def test_whisper_prompt_keeps_the_most_useful_names_within_budget():
    words = ["Prada", "Monika", "prada", "  ", "Toru  Kitajima", "A" * 30]
    assert loop.whisper_prompt(words) == f"{'A' * 30}, Toru Kitajima, Monika, Prada."
    assert loop.whisper_prompt(words, "Velvet", budget=40) == "Toru Kitajima, Monika, Prada. Velvet"
    assert loop.whisper_prompt([], "  Velvet ") == "Velvet"
    assert loop.whisper_prompt([]) == ""
    long = loop.whisper_prompt([f"Name{i}" for i in range(500)])
    assert len(long) <= loop.PROMPT_CHARS and long.endswith("Name1, Name0.")
