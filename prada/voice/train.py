"""Train your own openWakeWord models, such as "wake up prada" (needs `pip install "prada[wakeword]"`).

The recipe is openWakeWord's own: piper-sample-generator speaks the phrase (and phrases that
sound like it) in hundreds of synthetic voices, the clips are mixed with room echo and background
noise, and a small model learns to tell them apart from ~2,000 hours of everyday audio.

`prada voice train --setup` downloads that data once into `<data dir>/wakeword` (~17 GB). After
that, `prada voice train "wake up prada"` runs offline and writes `wake_up_prada.onnx` into
`<data dir>/voice`, where `wake_words = ["wake_up_prada"]` finds it. Runs are resumable: the same
command picks up where an interrupted one stopped. Linux only (WSL is fine); a GPU makes it fast.
"""

import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from prada.store import data_dir
from prada.voice.loop import SAMPLE_RATE

SAMPLES = 25_000  # synthetic clips of the phrase (openWakeWord recommends 20,000 or more)
STEPS = 50_000  # training steps (openWakeWord's default)
HF = "https://huggingface.co/datasets"
# Hugging Face URLs name a commit, not `main`, so a file can't change under its checksum.
FEATURES_REV = "985bf1b47e7f19c07741af82bfe32d5a9dc56096"  # davidscripka/openwakeword_features
AUDIOSET_REV = "5a2fa42a1506470d275a47ff8e1fdac5b364e6ef"  # agkphysics/AudioSet, before Parquet
PHONEMIZER_REV = "b5336bb41d8b1fbf32b3fd9cab779ef17d221520"  # the Space below


@dataclass(frozen=True)
class Download:
    name: str  # file, or folder for archives, under assets_dir()
    url: str
    about: str  # what it is and roughly how big, for the setup log
    # "file"; "audio": an archive of clips, unpacked as 16 kHz mono WAVs; "code": a source tarball
    kind: str = "file"
    sha256: str = ""  # checked after the download when set (empty: the host publishes no hash)


# Its PyPI package leaves out `piper_train`, which the generator model needs, so use the source.
SAMPLE_GENERATOR = Download(
    "piper-sample-generator",
    "https://github.com/rhasspy/piper-sample-generator/archive/refs/tags/v3.2.0.tar.gz",
    "piper-sample-generator 3.2.0 source",
    kind="code",
)
GENERATOR = Download(  # next to its .json config, which the source tarball ships
    "piper-sample-generator/models/en_US-libritts_r-medium.pt",
    "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
    "Piper LibriTTS-R voice generator (200 MB)",
)
FEATURES = Download(
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    f"{HF}/davidscripka/openwakeword_features/resolve/{FEATURES_REV}/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "2,000 hours of everyday audio as openWakeWord features (~16 GB)",
    sha256="721a66d0682c65a1b5c1da0aa109409cede1d20e28b15235c344b000cbb7654f",
)
VALIDATION = Download(
    "validation_set_features.npy",
    f"{HF}/davidscripka/openwakeword_features/resolve/{FEATURES_REV}/validation_set_features.npy",
    "11 hours of speech, noise and music to count false wakes",
    sha256="a56a8a0f8e0efb91900acc6de4c0cdf4c564842e8475a7d49b36c039e17a690f",
)
# DeepPhonemizer's own S3 bucket (what openWakeWord downloads from) now answers AccessDenied, and
# the project publishes no other copy. This is a copy of the same file in a Hugging Face Space
# (not official: hence the pinned commit and checksum). The checkpoint is a pickle, so setup
# only accepts it if it loads with `weights_only=True` and the allowlist in `PHONEMIZER_CLASSES`.
PHONEMIZER = Download(
    "en_us_cmudict_forward.pt",
    f"https://huggingface.co/spaces/mostafaashahin/Phone-aid/resolve/{PHONEMIZER_REV}/models/d_phonemizer/en_us_cmudict_forward.pt",
    "DeepPhonemizer model, for words missing from the CMU dictionary (67 MB)",
    sha256="c2e1fb223d7e027bf7b33052540c6f71d19db6d7fd87ab8671152b8b114501c2",
)
PHONEMIZER_CLASSES = (
    "Preprocessor",
    "SequenceTokenizer",
    "LanguageTokenizer",
)  # dp.preprocessing.text
RIRS = Download(
    "rirs",
    "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip",
    "MIT room impulse responses (270 real rooms)",
    kind="audio",
)
BACKGROUND = Download(
    "background/audioset",
    f"{HF}/agkphysics/AudioSet/resolve/{AUDIOSET_REV}/data/bal_train09.tar",  # later turned to Parquet
    "an AudioSet slice of background noise and music (970 MB)",
    kind="audio",
    sha256="9ab88550e77c8289acf71d19ae3f254fadad16f3cea305ef08a16949a928acf9",
)
DOWNLOADS = (SAMPLE_GENERATOR, GENERATOR, FEATURES, VALIDATION, PHONEMIZER, RIRS, BACKGROUND)


def assets_dir() -> Path:
    return data_dir() / "wakeword"


def slug(phrase: str) -> str:
    """The model's name for a phrase: "Wake up, Prada!" -> "wake_up_prada"."""
    name = re.sub(r"[^a-z0-9]+", "_", phrase.lower()).strip("_")
    if not name:
        raise ValueError(f"no letters or digits in the wake phrase {phrase!r}")
    return name


def _ready(d: Download) -> bool:
    path = assets_dir() / d.name
    return path.is_file() if d.kind == "file" else _done(path).is_file()


def _done(folder: Path) -> Path:
    """Marks an unpacked archive as complete; beside it, as the trainer reads every file in it."""
    return folder.with_name(folder.name + ".done")


def missing() -> list[Download]:
    return [d for d in DOWNLOADS if not _ready(d)]


def background_dirs() -> list[Path]:
    """Every folder under `background/` that holds WAVs: the download, plus any of your own
    16 kHz recordings (e.g. `background/mine/`) to teach the model what your room sounds like."""
    root = assets_dir() / "background"
    dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    return [p for p in dirs if any(p.glob("*.wav"))]


def setup() -> None:
    """Download what training needs (skips what is already there; resumes partial files)."""
    root = assets_dir()
    for d in DOWNLOADS:
        if _ready(d):
            continue
        print(f"{d.about} ...", file=sys.stderr)
        target = root / d.name
        if d.kind == "file":
            _fetch(d.url, target, d.sha256)
            if d is PHONEMIZER:
                _check_phonemizer(target)
            continue
        archive = root / "downloads" / d.url.rsplit("/", 1)[-1]
        _fetch(d.url, archive, d.sha256)
        (_unpack_audio if d.kind == "audio" else _unpack_code)(archive, target)
        archive.unlink()
    _feature_models()
    print(f"Training data ready in {root}", file=sys.stderr)


def _feature_models() -> None:
    """openWakeWord's trainer needs its melspectrogram and embedding models (GitHub release
    assets) inside its package, and stops with a file error when they are missing."""
    import openwakeword.utils

    openwakeword.utils.download_models()


def allow_phonemizer_classes() -> None:
    """Let `torch.load(weights_only=True)` unpickle the phonemizer checkpoint's own classes
    (and nothing else). torch 2.6 made that the default and added `add_safe_globals`, which is
    why the `wakeword` extra needs torch>=2.6: DeepPhonemizer's own `torch.load` passes no flag."""
    import torch.serialization
    from dp.preprocessing import text

    torch.serialization.add_safe_globals([getattr(text, name) for name in PHONEMIZER_CLASSES])


def _check_phonemizer(path: Path) -> None:
    """Refuse a phonemizer checkpoint that holds anything but data and the allowlisted classes."""
    import torch

    allow_phonemizer_classes()
    try:
        torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"{path.name} is not a plain DeepPhonemizer checkpoint ({e}); deleted"
        ) from e


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, target: Path, sha256: str = "") -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:  # the server ignored the range: start over
                have = 0
            total = have + int(response.headers.get("Content-Length") or 0)
            with part.open("ab" if have else "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    have += len(chunk)
                    print(f"\r  {have >> 20} / {total >> 20} MB", end="", file=sys.stderr)
        print(file=sys.stderr)
    except urllib.error.HTTPError as e:
        if e.code != 416 or not have:  # 416: the .part is already whole; the checksum decides
            raise
    if sha256 and (got := _sha256(part)) != sha256:
        part.unlink()
        raise ValueError(
            f"{target.name}: checksum mismatch (got {got}, expected {sha256}); deleted"
        )
    part.replace(target)


def _audio_members(archive: Path):
    """(name, bytes) for each audio file in a .zip or .tar archive."""
    suffixes = (".wav", ".flac")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.lower().endswith(suffixes):
                    yield name, z.read(name)
        return
    with tarfile.open(archive) as t:
        for member in t:
            if member.isfile() and member.name.lower().endswith(suffixes):
                yield member.name, t.extractfile(member).read()


def _unpack_code(archive: Path, target: Path) -> None:
    """Unpack a GitHub source tarball into `target`, without its top-level folder."""
    if target.exists():
        shutil.rmtree(target)
    with tarfile.open(archive) as t:
        t.extractall(target.parent / "downloads" / "code", filter="data")
        top = target.parent / "downloads" / "code" / t.getnames()[0].split("/")[0]
    top.replace(target)
    top.parent.rmdir()
    _done(target).write_text("")


def _unpack_audio(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, data in _audio_members(archive):
        try:
            to_16k_wav(data, target / (Path(name).stem + ".wav"))
            count += 1
        except RuntimeError as e:  # soundfile's error for a file it can't decode
            print(f"  skipped {name}: {e}", file=sys.stderr)
    if not count:
        raise ValueError(f"no readable audio in {archive}")
    _done(target).write_text(f"{count} clips\n")


def to_16k_wav(data: bytes, target: Path) -> None:
    """Write audio (any rate or channel count soundfile reads) as 16 kHz mono 16-bit WAV."""
    import numpy as np
    import soundfile
    from scipy.signal import resample_poly

    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, rate // g)
    soundfile.write(target, np.clip(audio, -1, 1), SAMPLE_RATE, subtype="PCM_16")


# openWakeWord's trainer imports `generate_samples` from piper_sample_generator_path, which
# piper-sample-generator 3 replaced with a package whose generator model is a required argument.
SHIM = '''"""Written by `prada voice train` for openWakeWord's trainer."""

import sys

sys.path.insert(0, {source!r})

from piper_sample_generator.__main__ import generate_samples as _generate  # noqa: E402


def generate_samples(*args, **kwargs):
    kwargs.setdefault("model", {model!r})
    return _generate(*args, **kwargs)
'''


# Runs openWakeWord's trainer with the phonemizer's classes allowed to unpickle.
RUNNER = '''"""Written by `prada voice train` for openWakeWord's trainer."""

import runpy

from prada.voice.train import allow_phonemizer_classes

allow_phonemizer_classes()
runpy.run_module("openwakeword.train", run_name="__main__", alter_sys=True)
'''


def training_config(phrase: str, name: str, samples: int, steps: int, work: Path) -> dict:
    """openWakeWord's training config (see its examples/custom_model.yml)."""
    root = assets_dir()
    background = background_dirs()
    return {
        "model_name": name,
        "target_phrase": [phrase],
        "custom_negative_phrases": [],
        "n_samples": samples,
        "n_samples_val": max(samples // 10, 100),
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        "piper_sample_generator_path": str(work),
        "output_dir": str(work),
        "rir_paths": [str(root / RIRS.name)],
        "background_paths": [str(p) for p in background],
        "background_paths_duplication_rate": [1] * len(background),
        "false_positive_validation_data_path": str(root / VALIDATION.name),
        "augmentation_rounds": 1,
        "feature_data_files": {"ACAV100M_sample": str(root / FEATURES.name)},
        "batch_n_per_class": {"ACAV100M_sample": 1024, "adversarial_negative": 50, "positive": 50},
        "model_type": "dnn",
        "layer_size": 32,
        "steps": steps,
        "max_negative_weight": 1500,
        "target_false_positives_per_hour": 0.2,
    }


def output_path(name: str, out: str | None) -> Path:
    """Where the model goes: `out` (a folder or an .onnx path), else `<data dir>/voice`."""
    from prada.voice.devices import models_dir

    target = Path(out).expanduser() if out else models_dir()
    return target if target.suffix == ".onnx" else target / f"{name}.onnx"


def train(
    phrase: str,
    name: str | None = None,
    samples: int = SAMPLES,
    steps: int = STEPS,
    out: str | None = None,
    run=subprocess.run,
) -> Path:
    """Train a model for `phrase` and return the .onnx path. Raises ValueError when the
    training data is missing and RuntimeError when openWakeWord's trainer fails."""
    import openwakeword.train  # noqa: F401  (fail early, with the extra's install hint)
    import piper  # noqa: F401

    name = slug(name or phrase)
    if samples < 100 or steps < 100:
        raise ValueError("--samples and --steps must be at least 100")
    if gone := missing():
        names = ", ".join(d.name for d in gone)
        raise ValueError(f"training data missing ({names}): run `prada voice train --setup`")
    _install_phonemizer()
    work = assets_dir() / "runs" / name
    work.mkdir(parents=True, exist_ok=True)
    _drop_partial_features(work / name)
    shim = SHIM.format(
        source=str(assets_dir() / SAMPLE_GENERATOR.name),
        model=str(assets_dir() / GENERATOR.name),
    )
    (work / "generate_samples.py").write_text(shim)
    config = work / "config.yaml"  # JSON is valid YAML
    config.write_text(json.dumps(training_config(phrase, name, samples, steps, work), indent=2))
    (work / "run_trainer.py").write_text(RUNNER)
    command = [sys.executable, str(work / "run_trainer.py"), "--training_config", str(config)]
    failed = RuntimeError("training failed (see above); run the same command to resume")
    if run([*command, "--generate_clips"], cwd=work).returncode != 0:
        raise failed
    # piper-sample-generator 3 writes at the voice's rate (22.05 kHz); the trainer wants 16.
    for folder in CLIP_FOLDERS:
        resample_folder(work / name / folder)
    model = work / f"{name}.onnx"
    model.unlink(missing_ok=True)
    if run([*command, "--augment_clips", "--train_model"], cwd=work).returncode != 0:
        if not model.is_file():
            raise failed
        # openwakeword 0.6.0 always tries a TFLite copy afterwards, which needs TensorFlow.
        print(
            "prada: (the TFLite error above doesn't matter: PRADA uses the .onnx)", file=sys.stderr
        )
    target = output_path(name, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model, target)
    return target


CLIP_FOLDERS = ("positive_train", "positive_test", "negative_train", "negative_test")


def resample_folder(folder: Path) -> None:
    """Rewrite the folder's WAVs that are not 16 kHz (reads only headers for the rest)."""
    import wave

    for path in sorted(folder.glob("*.wav")):
        with wave.open(str(path)) as w:
            if w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1:
                continue
        to_16k_wav(path.read_bytes(), path)


FEATURE_FILES = ("positive_features_train", "negative_features_train")
FEATURE_FILES += ("positive_features_test", "negative_features_test")


def _drop_partial_features(clips: Path) -> None:
    """The trainer skips augmentation once positive_features_train.npy exists, even if a run
    died before writing the rest; delete an incomplete set so it is computed again."""
    files = [clips / f"{f}.npy" for f in FEATURE_FILES]
    if not all(f.is_file() for f in files):
        for f in files:
            f.unlink(missing_ok=True)


def _install_phonemizer() -> None:
    """openWakeWord looks for DeepPhonemizer's model inside its own package and downloads it
    there when a word is not in its dictionary; put our copy there so training stays offline."""
    import openwakeword

    target = Path(openwakeword.__file__).parent / "resources" / PHONEMIZER.name
    if not target.is_file() or _sha256(target) != PHONEMIZER.sha256:  # e.g. fetched unchecked
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets_dir() / PHONEMIZER.name, target)
