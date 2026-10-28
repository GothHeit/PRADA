"""PRADA's voice: wake word -> speech-to-text -> the home node -> text-to-speech.

The loop is a client of the node's HTTP API, like `prada chat`, so it runs wherever the
microphone and speakers are. On a Windows PC that is the Windows side (WSL audio is
unreliable), reaching the node in WSL through localhost forwarding; on a future sentinel
node it is the Pi. Engines: openWakeWord (wake words), whisper.cpp (speech-to-text), Piper
(text-to-speech), all offline. They come with the optional `voice` extra and are imported
only by `prada.voice.devices`; under PRADA_FAKE_LLM the loop runs on text fakes instead.
"""

import os

# onnxruntime (openWakeWord, Piper) uploads usage telemetry to Microsoft on import on Linux
# unless told not to; on Windows it logs ETW events instead (devices._no_telemetry).
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
