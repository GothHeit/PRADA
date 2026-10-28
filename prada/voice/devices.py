"""The real microphone, speakers and engines (needs `pip install "prada[voice]"`).

openWakeWord listens for the wake words, whisper.cpp (pywhispercpp) transcribes, Piper
speaks, and sounddevice moves the audio. Models live in `<data dir>/voice` and are fetched
once by `prada voice setup`; after that nothing here touches the network.
"""

import re
import time
from pathlib import Path

from prada.config import VoiceSettings
from prada.store import data_dir
from prada.voice.loop import (
    FRAME,
    FRAME_SECONDS,
    SAMPLE_RATE,
    Endpointer,
    split_by_language,
    track_noise,
)

SETTLE = 0.3  # seconds of mic ignored after she speaks, before listening for a follow-up
MIN_NOISE = 100.0  # starting guess for the room's background loudness (int16 RMS)


def models_dir() -> Path:
    return data_dir() / "voice"


def _is_path(name: str) -> bool:
    return name.endswith((".onnx", ".tflite", ".bin")) or "/" in name or "\\" in name


def _wake_model(word: str) -> str:
    if not _is_path(word):
        own = models_dir() / f"{word}.onnx"  # e.g. trained by `prada voice train`
        return str(own) if own.is_file() else word  # else pretrained, e.g. "hey_jarvis"
    path = Path(word).expanduser()
    if path.suffix != ".onnx":
        raise ValueError(f"wake word models must be .onnx files: {path}")
    if not path.is_file():
        raise ValueError(f"wake word model not found: {path}")
    return str(path)


def _whisper_model(name: str) -> Path:
    path = Path(name).expanduser() if _is_path(name) else models_dir() / f"ggml-{name}.bin"
    if not path.is_file():
        raise ValueError(f"whisper model not found: {path} (run `prada voice setup`)")
    return path


def _piper_voice(name: str) -> Path:
    path = Path(name).expanduser() if _is_path(name) else models_dir() / f"{name}.onnx"
    if not path.is_file():
        raise ValueError(f"Piper voice not found: {path} (run `prada voice setup`)")
    return path


def setup(s: VoiceSettings) -> None:
    """Download the models the settings name (skips files that are already there)."""
    import openwakeword.utils
    import pywhispercpp.utils
    from piper.download_voices import download_voice

    target = models_dir()
    target.mkdir(parents=True, exist_ok=True)
    print("openWakeWord models ...")
    pretrained = [w for w in s.wake_words if not _is_path(w) and _wake_model(w) == w]
    # An empty list downloads every pretrained model; a name matching none of them fetches
    # only the shared feature models.
    openwakeword.utils.download_models(pretrained or ["features-only"])
    if not _is_path(s.whisper_model):
        print(f"whisper.cpp model {s.whisper_model} ...")
        if not pywhispercpp.utils.download_model(s.whisper_model, str(target)):
            raise ValueError(f"unknown whisper model {s.whisper_model!r}")
    for voice in (s.piper_voice, *s.voices.values()):
        if not _is_path(voice):
            print(f"Piper voice {voice} ...")
            download_voice(voice, target)
    print(f"Voice models ready in {target}")


def errors() -> tuple[type[Exception], ...]:
    """What a bad setting, missing model or audio device raises. sounddevice's PortAudioError
    (e.g. a device that refuses 16 kHz, or an unplugged mic) is not an OSError."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):
        return (ValueError, OSError)
    return (ValueError, OSError, sd.PortAudioError)


def _no_telemetry() -> None:
    """ORT_DISABLE_TELEMETRY (see prada.voice) covers Linux and macOS; on Windows onnxruntime
    writes ETW telemetry events instead, which only this call switches off."""
    import onnxruntime

    onnxruntime.disable_telemetry_events()


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def rms(frame) -> float:
    import numpy as np

    return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))


class MicEars:
    """Listens on the microphone: wake words with openWakeWord, then whisper.cpp."""

    def __init__(self, s: VoiceSettings):
        import sounddevice as sd
        from openwakeword.model import Model as WakeModel
        from pywhispercpp.model import Model as WhisperModel

        _no_telemetry()
        self.threshold = s.threshold
        self.silence = s.silence
        self.language = s.language
        self.wake = WakeModel(
            wakeword_models=[_wake_model(w) for w in s.wake_words], inference_framework="onnx"
        )
        self.stt = WhisperModel(
            str(_whisper_model(s.whisper_model)), redirect_whispercpp_logs_to=None
        )
        self.stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME,
            channels=1,
            dtype="int16",
            device=s.input_device,
        )
        self.noise = MIN_NOISE

    def _frames(self):
        import numpy as np

        if not self.stream.active:
            self.stream.start()
        while True:
            data, _overflowed = self.stream.read(FRAME)
            yield np.frombuffer(data, dtype=np.int16)

    def wait_for_wake(self) -> str | None:
        self.wake.reset()  # forget the last detection and our own voice
        for frame in self._frames():
            self.noise = track_noise(self.noise, rms(frame))
            scores = self.wake.predict(frame)
            word, score = max(scores.items(), key=lambda item: item[1])
            if score >= self.threshold:
                return word
        return None

    def listen(self, wait: float | None = None) -> str:
        import numpy as np

        if not self.stream.active:
            self.stream.start()
        # Drop what was captured during the chime, so it isn't taken for speech.
        if waiting := self.stream.read_available:
            self.stream.read(waiting)
        if wait is not None:  # right after she spoke: let the room's echo of her voice die out
            for _ in range(round(SETTLE / FRAME_SECONDS)):
                self.stream.read(FRAME)
        end = Endpointer(self.noise, self.silence, *([] if wait is None else [wait]))
        frames = []
        for frame in self._frames():
            frames.append(frame)
            if end.push(rms(frame)):
                break
        self.stream.stop()  # deaf while thinking and speaking
        if not end.heard:
            return ""
        audio = np.concatenate(frames).astype(np.float32) / 32768
        segments = self.stt.transcribe(audio, language=self.language)
        return clean_transcript(" ".join(segment.text for segment in segments))


def clean_transcript(text: str) -> str:
    """Drop whisper's non-speech tags such as [BLANK_AUDIO] or (music)."""
    return " ".join(re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text).split())


def chime_samples(rate: int = SAMPLE_RATE):
    """The wake chime: a rising two-note ping, 0.45 s, led by a little silence so a sound
    device that takes a moment to start (Voicemeeter) doesn't swallow the notes."""
    import numpy as np

    def note(freq: float, seconds: float):
        t = np.arange(int(rate * seconds)) / rate
        return np.sin(2 * np.pi * freq * t) * np.hanning(t.size) * 0.5

    lead = np.zeros(int(rate * 0.15))
    tone = np.concatenate([lead, note(660, 0.15), note(880, 0.3)])
    return (tone * 32767).astype(np.int16)


class Speaker:
    """Plays int16 mono audio through output streams that stay open, one per sample rate: a
    stream opened per sound loses its first moments to the device's start-up latency."""

    def __init__(self, device):
        self.device = device
        self.streams = {}
        self.last = None

    def play(self, samples, rate: int) -> None:
        import sounddevice as sd

        stream = self.streams.get(rate)
        if stream is None:
            stream = self.streams[rate] = sd.OutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.device
            )
            stream.start()
        stream.write(samples)  # returns once buffered, not once heard
        self.last = stream

    def drain(self) -> None:
        """Wait out what is still in the device's buffer, so the mic doesn't hear its tail."""
        if self.last is not None:
            time.sleep(self.last.latency + 0.05)


class PiperMouth:
    """Speaks with Piper voices through the speakers: the main one, and per-language ones
    for sentences in another language (e.g. the tutor's Russian)."""

    def __init__(self, s: VoiceSettings):
        from piper import PiperVoice

        _no_telemetry()
        self.voice = PiperVoice.load(_piper_voice(s.piper_voice))
        self.voices = {lang: PiperVoice.load(_piper_voice(v)) for lang, v in s.voices.items()}
        self.speaker = Speaker(s.output_device)

    def _play(self, samples, rate: int) -> None:
        self.speaker.play(samples, rate)

    def chime(self) -> None:
        self._play(chime_samples(), SAMPLE_RATE)
        self.speaker.drain()

    def say(self, text: str) -> None:
        for lang, part in split_by_language(text, self.voices):
            voice = self.voices.get(lang, self.voice)
            for chunk in voice.synthesize(part):  # one sentence at a time
                self._play(chunk.audio_int16_array, chunk.sample_rate)
        self.speaker.drain()


def build(s: VoiceSettings) -> tuple[MicEars, PiperMouth]:
    return MicEars(s), PiperMouth(s)
