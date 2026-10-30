"""The engines that give her a voice. Each turns text in one language into int16 audio chunks.

Piper (the `voice` extra) is the default and the fallback. Kokoro (`kokoro` extra) runs on
the CPU and sounds more natural. Qwen3-TTS (`qwen` extra, NVIDIA GPU) speaks every language
in one voice, cloned from a short clip that `prada voice design` makes from a description.
Models live in `models_dir()`; nothing here downloads at runtime.
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from prada.config import VoiceSettings
from prada.store import data_dir

KOKORO_MODEL = "kokoro-v1.0.onnx"
KOKORO_VOICES = "voices-v1.0.bin"
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
# Kokoro's voice names start with their language, e.g. "af_heart" (American English).
KOKORO_LANGS = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "f": "fr-fr",
    "p": "pt-br",
    "e": "es",
    "i": "it",
    "h": "hi",
    "z": "cmn",
}
# Who says sentences in the tutor's other languages when Kokoro is her voice. The rest go to
# the Piper voices in [voice.voices]: Kokoro has no Russian, and kokoro-onnx reads Japanese
# through espeak-ng, which spells kanji out as "Chinese letter".
KOKORO_FOREIGN = {"fr": "ff_siwis"}

SENTENCES = re.compile(r"(?<=[.!?])\s+")

QWEN_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
QWEN_LANGS = {
    "en": "English",
    "ja": "Japanese",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "ko": "Korean",
    "zh": "Chinese",
}
# Qwen's streamed chunks click where they join, so she says each sentence whole instead.
QWEN_SENTENCES = re.compile(r"(?<=[.!?。！？])\s+|(?<=[。！？])")  # as split_by_language
# What a designed voice says in her reference clip: varied sounds, a question, some warmth.
DESIGN_LINE = (
    "Hi, it's me. I was just thinking about you, actually. "
    "Did you sleep well? Come on, tell me everything."
)

# `prada voice try`: the same lines for every engine, so they can be compared by ear.
LISTENING_TEST = (
    "Good morning, Kafka. It's seven thirty, and it's raining a little outside.",
    "Oh, you're back! I missed you. Did you get some rest today?",
    "Hmm... I'm not sure about that. Let me think for a second.",
    "Your favorite singer is Toru Kitajima, right? I like his voice too.",
    "Okay, okay, I'll stop teasing you. For now.",
    "In French, you'd say: Je suis fatiguée, mais contente.",
    "おはようございます。今日もがんばりましょう。",
    "Я читаю книгу.",
    "That's three hundred and twelve euros, or about fifty-five percent of the budget.",
)


class Engine(Protocol):
    def speak(self, text: str, lang: str | None) -> Iterator[tuple[object, int]]:
        """(int16 samples, sample rate) chunks for `text`; `lang` None = the main language."""


def models_dir() -> Path:
    return data_dir() / "voice"


def is_path(name: str) -> bool:
    return name.endswith((".onnx", ".tflite", ".bin", ".wav")) or "/" in name or "\\" in name


def _int16(audio):
    import numpy as np

    return (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16)


def piper_voice(name: str) -> Path:
    path = Path(name).expanduser() if is_path(name) else models_dir() / f"{name}.onnx"
    if not path.is_file():
        raise ValueError(f"Piper voice not found: {path} (run `prada voice setup`)")
    return path


class Piper:
    def __init__(self, voice: str, speed: float = 1.0):
        from piper import PiperVoice, SynthesisConfig

        self.voice = PiperVoice.load(piper_voice(voice))
        self.config = SynthesisConfig(length_scale=1 / speed)

    def speak(self, text, lang):
        for chunk in self.voice.synthesize(text, self.config):  # one sentence at a time
            yield chunk.audio_int16_array, chunk.sample_rate


def kokoro_files() -> tuple[Path, Path]:
    return models_dir() / KOKORO_MODEL, models_dir() / KOKORO_VOICES


class Kokoro:
    """Kokoro-82M through onnxruntime on the CPU: no GPU memory taken from the brain."""

    def __init__(self, voice: str, speed: float = 1.0):
        from kokoro_onnx import Kokoro as Model

        model, voices = kokoro_files()
        if not (model.is_file() and voices.is_file()):
            raise ValueError(f"Kokoro model not found in {models_dir()} (run `prada voice setup`)")
        self.model = Model(str(model), str(voices))
        if voice not in self.model.get_voices() or voice[0] not in KOKORO_LANGS:
            raise ValueError(f"unknown Kokoro voice {voice!r}")
        self.voice = voice
        self.speed = speed

    def speak(self, text, lang):
        voice = KOKORO_FOREIGN.get(lang, self.voice) if lang else self.voice
        for sentence in SENTENCES.split(text.strip()):  # the first one plays sooner
            if sentence:
                audio, rate = self.model.create(
                    sentence, voice=voice, speed=self.speed, lang=KOKORO_LANGS[voice[0]]
                )
                yield _int16(audio), rate


def qwen_voice(name: str) -> tuple[Path, str]:
    """Her reference clip and what it says (`<name>.wav` + `<name>.txt`)."""
    wav = Path(name).expanduser() if is_path(name) else models_dir() / f"{name}.wav"
    text = wav.with_suffix(".txt")
    if not wav.is_file() or not text.is_file():
        raise ValueError(f"Qwen voice not found: {wav} and {text.name} (run `prada voice design`)")
    return wav, text.read_text(encoding="utf-8").strip()


class Qwen:
    """Qwen3-TTS cloning her reference clip on the GPU, a sentence at a time."""

    def __init__(self, model: str, voice: str):
        from faster_qwen3_tts import FasterQwen3TTS

        self.ref_audio, self.ref_text = qwen_voice(voice)
        self.model = FasterQwen3TTS.from_pretrained(model, local_files_only=True)
        self.model.warmup()  # captures the CUDA graphs now, not during her first reply

    def speak(self, text, lang):
        for sentence in QWEN_SENTENCES.split(text.strip()):
            if not sentence:
                continue
            audio, rate = self.model.generate_voice_clone(
                sentence,
                QWEN_LANGS.get(lang, "auto") if lang else "auto",
                ref_audio=str(self.ref_audio),
                ref_text=self.ref_text,
                # Another language: only her timbre, so her English clip doesn't bleed an accent.
                xvec_only=lang is not None,
            )
            yield _int16(audio[0]), rate


def engines(s: VoiceSettings) -> tuple[Engine, dict[str, Engine]]:
    """Her main voice and the voice for each other language. A newer engine that won't load
    falls back to Piper, with a notice."""
    piper = {lang: Piper(v, s.speed) for lang, v in s.voices.items()}
    try:
        if s.tts == "kokoro":
            main = Kokoro(s.kokoro_voice, s.speed)
            return main, {**piper, **dict.fromkeys(KOKORO_FOREIGN, main)}
        if s.tts == "qwen":
            main = Qwen(s.qwen_model, s.qwen_voice)
            return main, {**piper, **dict.fromkeys(QWEN_LANGS, main)}
    except (ImportError, ValueError, OSError, RuntimeError) as e:
        print(f"voice: {s.tts} didn't load, using Piper ({e})", file=sys.stderr)
    return Piper(s.piper_voice, s.speed), piper


def setup(s: VoiceSettings) -> None:
    """Fetch what the chosen engine needs (Piper's voices are fetched by the caller)."""
    if s.tts == "kokoro":
        import urllib.request

        models_dir().mkdir(parents=True, exist_ok=True)
        for path in kokoro_files():
            if not path.is_file():
                print(f"Kokoro {path.name} ...")
                part = path.with_suffix(".part")
                urllib.request.urlretrieve(KOKORO_URL + path.name, part)
                part.replace(path)
    elif s.tts == "qwen":
        from huggingface_hub import snapshot_download

        print(f"Qwen3-TTS {s.qwen_model} ...")
        snapshot_download(s.qwen_model)


def design(description: str, name: str, takes: int, line: str = DESIGN_LINE) -> list[Path]:
    """Make reference clips for a voice described in words, with Qwen3-TTS VoiceDesign: one
    `<name>-<n>.wav` + `.txt` per take, each a little different. Downloads the design model
    (~4 GB) on first use; needs a few GB of free GPU memory, so stop the brain first."""
    import wave

    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(QWEN_DESIGN_MODEL)
    target = models_dir()
    target.mkdir(parents=True, exist_ok=True)
    made = []
    for n in range(1, takes + 1):
        audio, rate = model.generate_voice_design(line, description, "English")
        path = target / f"{name}-{n}.wav"
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(_int16(audio[0]).tobytes())
        path.with_suffix(".txt").write_text(line + "\n", encoding="utf-8")
        made.append(path)
    return made
