"""PRADA's shared model client: Ollama on the node, or a module's canned fake offline.

Every chat function takes (messages, schema) and returns the model's JSON reply as a string.
"""

import os
import sys

MODEL = "qwen2.5:14b"
# Context window in tokens, set explicitly: Ollama's default (2048-4096) silently cuts the
# start of a long prompt, i.e. the persona and what she remembers. Keep the prompt well under:
# persona, memory (prada.memory.MEMORY_CHARS) and MAX_TURNS messages come to about 3000.
NUM_CTX = 8192

_fake_notice_shown = False


def chat(messages, schema, model=MODEL, keep_alive=None) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        messages=messages,
        format=schema,
        options={"temperature": 0, "num_ctx": NUM_CTX},
        keep_alive=keep_alive,
    )
    return response["message"]["content"]


def fake_enabled() -> bool:
    return bool(os.environ.get("PRADA_FAKE_LLM"))


def client(fake, keep_alive=None):
    """The chat function to use: `fake` when PRADA_FAKE_LLM is set, Ollama otherwise."""
    if fake_enabled():
        global _fake_notice_shown
        if not _fake_notice_shown:
            print("PRADA_FAKE_LLM is set: using the canned offline client", file=sys.stderr)
            _fake_notice_shown = True
        return fake
    if keep_alive is None:
        return chat
    return lambda messages, schema: chat(messages, schema, keep_alive=keep_alive)
