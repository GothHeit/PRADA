"""The real microphone, speakers and engines (needs `pip install "prada[voice]"`).

openWakeWord listens for the wake words, whisper.cpp (pywhispercpp) transcribes, a TTS
engine from `prada.voice.tts` speaks, and sounddevice moves the audio. Models live in
`<data dir>/voice` and are fetched once by `prada voice setup`; after that nothing here
touches the network.
"""

import re
import sys
import time
from pathlib import Path

from prada.config import VoiceSettings
from prada.voice import tts
from prada.voice.loop import (
    FRAME,
    FRAME_SECONDS,
    SAMPLE_RATE,
    Barge,
    Endpointer,
    debug,
    respell,
    split_by_language,
    track_noise,
)

RETRIES = 1  # listens again after speech whisper made no words of, this many times
SETTLE = 0.3  # seconds of mic ignored after she speaks, before listening for a follow-up
SLICE = 0.1  # seconds of audio written at a time, so an interruption cuts her off this fast
MIN_NOISE = 100.0  # starting guess for the room's background loudness (int16 RMS)


models_dir = tts.models_dir
_is_path = tts.is_path


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
    tts.setup(s)
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
        hint = {"initial_prompt": s.whisper_prompt} if s.whisper_prompt else {}
        self.stt = WhisperModel(
            str(_whisper_model(s.whisper_model)),
            redirect_whispercpp_logs_to=None,
            print_progress=False,
            **hint,
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
        debug("waiting for a wake word")
        for frame in self._frames():
            self.noise = track_noise(self.noise, rms(frame))
            scores = self.wake.predict(frame)
            word, score = max(scores.items(), key=lambda item: item[1])
            if score >= self.threshold:
                debug(f"heard {word} ({score:.2f})")
                return word
        return None

    def watch(self, barge: Barge, ready) -> None:
        """Listen for a wake word while the turn runs (she thinks and speaks in a thread; this
        is the main thread, the mic's only reader); hearing one sets `barge`. While
        `barge.held` (she says a wake phrase herself) the model is reset on every frame
        instead, so her words can't add up to a detection. Returns once `barge.over` is set."""
        if not self.stream.active:
            self.stream.start()  # before her thread starts: no two audio calls at once
        self.wake.reset()
        debug("watching for the wake word")
        ready()
        try:
            for frame in self._frames():
                if barge.over.is_set():
                    break
                if barge.held:
                    self.wake.reset()
                    continue
                word, score = max(self.wake.predict(frame).items(), key=lambda item: item[1])
                if score >= self.threshold:
                    debug(f"heard {word} ({score:.2f}) while she talked")
                    barge.set()
                    break
        except errors() as e:  # a lost mic must not take her voice with it
            print(f"voice: stopped listening for interruptions ({e})", file=sys.stderr)
            barge.over.wait()
        self.wake.reset()
        debug("stopped watching")

    def listen(self, wait: float | None = None) -> str:
        if not self.stream.active:
            self.stream.start()
        # Drop what was captured during the chime, so it isn't taken for speech.
        if waiting := self.stream.read_available:
            self.stream.read(waiting)
        if wait is not None:  # right after she spoke: let the room's echo of her voice die out
            for _ in range(round(SETTLE / FRAME_SECONDS)):
                self.stream.read(FRAME)
        for tries in range(1 + RETRIES):
            heard, text = self._hear(wait)
            if text or not heard:
                break
            # Speech whisper made no words of: say so and listen again (the mic kept
            # running meanwhile, so nothing said since is lost).
            again = tries < RETRIES
            print(
                f"voice: heard something but no words{'; listening again' if again else ''}",
                file=sys.stderr,
            )
        self.stream.stop()  # deaf while thinking and speaking
        return text

    def _hear(self, wait: float | None) -> tuple[bool, str]:
        """Record one request; (whether there was speech, its words)."""
        import numpy as np

        debug("listening" + ("" if wait is None else f" for a follow-up ({wait:g} s)"))
        end = Endpointer(self.noise, self.silence, *([] if wait is None else [wait]))
        frames = []
        for frame in self._frames():
            frames.append(frame)
            if end.push(rms(frame)):
                break
        debug(
            f"listened {end.frames * FRAME_SECONDS:.1f} s, {'speech' if end.heard else 'nothing'}"
        )
        if not end.heard:
            return False, ""
        if end.too_long:
            print(
                "voice: you were still talking at the length limit; the rest went unheard",
                file=sys.stderr,
            )
        audio = np.concatenate(frames).astype(np.float32) / 32768
        from pywhispercpp.utils import redirect_stderr

        with redirect_stderr(to=None):  # whisper.cpp logs every language detection
            segments = self.stt.transcribe(audio, language=self.language)
        raw = " ".join(segment.text for segment in segments)
        debug(f"whisper: {raw.strip()!r}")
        return True, clean_transcript(raw)

    def close(self) -> None:
        self.stream.close()


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

    def play(self, samples, rate: int, barge: Barge | None = None) -> bool:
        """Queue `samples`; with `barge`, a little at a time, and once it is set what is still
        queued is dropped and False returned."""
        import sounddevice as sd

        stream = self.streams.get(rate)
        if stream is None:
            stream = self.streams[rate] = sd.OutputStream(
                samplerate=rate, channels=1, dtype="int16", device=self.device
            )
        if not stream.active:  # new, or stopped by an interruption
            stream.start()
        if self.last is not None and self.last is not stream:
            self.drain()  # another voice's rate: let the old stream finish first
        self.last = stream
        if barge is None:
            stream.write(samples)  # returns once buffered, not once heard
            return True
        step = max(1, int(rate * SLICE))
        for at in range(0, len(samples), step):
            if barge.is_set():
                self.stop()
                return False
            stream.write(samples[at : at + step])
        return True

    def stop(self) -> None:
        """Silence at once: drop what the device still has queued."""
        if self.last is not None:
            debug("speaker stopped")
            self.last.abort()
            self.last = None

    def drain(self) -> None:
        """Wait out what is still in the device's buffer, so the mic doesn't hear its tail."""
        if self.last is not None:
            time.sleep(self.last.latency + 0.05)

    def close(self) -> None:
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
        self.last = None


class SpeakerMouth:
    """Speaks through the speakers with her TTS engine (`[voice] tts`), and with the voice set
    for sentences in another language (e.g. the tutor's Russian)."""

    def __init__(self, s: VoiceSettings):
        _no_telemetry()
        self.voice, self.voices = tts.engines(s)
        self.speaker = Speaker(s.output_device)
        self.say_as = s.say_as

    def _play(self, samples, rate: int, barge: Barge | None = None) -> bool:
        return self.speaker.play(samples, rate, barge)

    def chime(self) -> None:
        self._play(chime_samples(), SAMPLE_RATE)
        self.speaker.drain()

    def say(self, text: str, barge: Barge | None = None) -> bool:
        for lang, part in split_by_language(respell(text, self.say_as), self.voices):
            for samples, rate in self.voices.get(lang, self.voice).speak(part, lang):
                if not self._play(samples, rate, barge):
                    return False
        self.speaker.drain()
        return barge is None or not barge.is_set()

    def close(self) -> None:
        self.speaker.close()


def build(s: VoiceSettings) -> tuple[MicEars, SpeakerMouth]:
    return MicEars(s), SpeakerMouth(s)


def try_voice(s: VoiceSettings, lines: list[str] | None = None) -> None:
    """Say `lines` (default: the listening test) with her voice, printing how long each took
    to start, so engines can be compared by ear and by speed."""
    start = time.monotonic()
    mouth = SpeakerMouth(s)
    print(f"{s.tts}: loaded in {time.monotonic() - start:.1f} s", file=sys.stderr)
    play = mouth._play
    first = []

    def timed(samples, rate, barge=None):
        if not first:
            first.append(time.monotonic())
        return play(samples, rate, barge)

    mouth._play = timed
    try:
        for line in lines or tts.LISTENING_TEST:
            first.clear()
            start = time.monotonic()
            mouth.say(line)
            print(f"{first[0] - start if first else 0:5.2f} s  {line}")
    finally:
        mouth.close()
