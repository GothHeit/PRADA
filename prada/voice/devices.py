"""The real microphone, speakers and engines (needs `pip install "prada[voice]"`).

openWakeWord listens for the wake words, whisper.cpp (pywhispercpp) transcribes, Piper
speaks, and sounddevice moves the audio. Models live in `<data dir>/voice` and are fetched
once by `prada voice setup`; after that nothing here touches the network.
"""

import re
from pathlib import Path

from prada.config import VoiceSettings
from prada.store import data_dir
from prada.voice.loop import FRAME, SAMPLE_RATE, Endpointer, split_by_language, track_noise

MIN_NOISE = 100.0  # starting guess for the room's background loudness (int16 RMS)


def models_dir() -> Path:
    return data_dir() / "voice"


def _is_path(name: str) -> bool:
    return name.endswith((".onnx", ".tflite", ".bin")) or "/" in name or "\\" in name


def _wake_model(word: str) -> str:
    if not _is_path(word):
        return word  # an openWakeWord pretrained model, e.g. "hey_jarvis"
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
    openwakeword.utils.download_models([w for w in s.wake_words if not _is_path(w)])
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

    def listen(self) -> str:
        import numpy as np

        # Drop what was captured during the chime, so it isn't taken for speech.
        if waiting := self.stream.read_available:
            self.stream.read(waiting)
        end = Endpointer(self.noise, self.silence)
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


class PiperMouth:
    """Speaks with Piper voices through the speakers: the main one, and per-language ones
    for sentences in another language (e.g. the tutor's Russian)."""

    def __init__(self, s: VoiceSettings):
        from piper import PiperVoice

        self.voice = PiperVoice.load(_piper_voice(s.piper_voice))
        self.voices = {lang: PiperVoice.load(_piper_voice(v)) for lang, v in s.voices.items()}
        self.device = s.output_device

    def _play(self, samples, rate: int) -> None:
        import sounddevice as sd

        sd.play(samples, rate, device=self.device)
        sd.wait()

    def chime(self) -> None:
        import numpy as np

        t = np.arange(int(SAMPLE_RATE * 0.12)) / SAMPLE_RATE
        tone = np.sin(2 * np.pi * 880 * t) * np.hanning(t.size) * 0.3
        self._play(tone.astype(np.float32), SAMPLE_RATE)

    def say(self, text: str) -> None:
        for lang, part in split_by_language(text, self.voices):
            voice = self.voices.get(lang, self.voice)
            for chunk in voice.synthesize(part):  # one sentence at a time
                self._play(chunk.audio_int16_array, chunk.sample_rate)


def build(s: VoiceSettings) -> tuple[MicEars, PiperMouth]:
    return MicEars(s), PiperMouth(s)
