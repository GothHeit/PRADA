"""PRADA's shared model client: Ollama on the node, or a module's canned fake offline.

Every chat function takes (messages, schema) and returns the model's JSON reply as a string.
The ones `client` returns also take `on_text`, called with each piece of the reply as the
model writes it (a fake gives the whole reply as one piece).
"""

import contextvars
import os
import sys

from prada import config

# Set by whoever wants the token counts of the next calls (the node's flight recorder,
# prada.trace): a dict that `chat` fills with tokens_in / tokens_out as Ollama reports them.
USAGE: contextvars.ContextVar[dict | None] = contextvars.ContextVar("usage", default=None)

# Context window in tokens, set explicitly: Ollama's default (2048-4096) silently cuts the
# start of a long prompt, i.e. the persona and what she remembers. Keep the prompt well under:
# persona, memory (prada.memory.MEMORY_CHARS) and MAX_TURNS messages come to about 3000.
NUM_CTX = 8192

_fake_notice_shown = False


def chat(messages, schema, model=None, keep_alive=None, on_text=None, think=None) -> str:
    """`model` None = the configured brain (`[llm] model`)."""
    import ollama

    model = model or model_name()

    args = dict(
        model=model,
        messages=messages,
        format=schema,
        options={"temperature": 0, "num_ctx": NUM_CTX},
        keep_alive=keep_alive,
    )
    if think is not None:  # only sent when set: older models reject any think flag
        args["think"] = think
    if on_text is None:
        response = ollama.chat(**args)
        _count(response)
        return response["message"]["content"]
    pieces = []
    for chunk in ollama.chat(**args, stream=True):
        if piece := chunk["message"]["content"]:
            pieces.append(piece)
            on_text(piece)
        _count(chunk)  # the last chunk carries the counts
    return "".join(pieces)


def _count(response) -> None:
    usage = USAGE.get()
    if usage is None:
        return
    for key, name in (("prompt_eval_count", "tokens_in"), ("eval_count", "tokens_out")):
        try:
            value = response[key]
        except (KeyError, TypeError, AttributeError):
            value = None
        if value is not None:
            usage[name] = value


def model_name() -> str:
    """The configured brain model (`[llm] model` or PRADA_LLM_MODEL)."""
    return config.llm_settings().model


def fake_enabled() -> bool:
    return bool(os.environ.get("PRADA_FAKE_LLM"))


def client(fake, keep_alive=None, model=None, think=None):
    """The chat function to use: `fake` when PRADA_FAKE_LLM is set, Ollama otherwise.
    `model` and `think` override the configured ones (e.g. `prada eval --model`)."""
    if fake_enabled():
        global _fake_notice_shown
        if not _fake_notice_shown:
            print("PRADA_FAKE_LLM is set: using the canned offline client", file=sys.stderr)
            _fake_notice_shown = True

        def canned(messages, schema, on_text=None):
            text = fake(messages=messages, schema=schema)
            if on_text is not None:
                on_text(text)
            return text

        return canned
    settings = config.llm_settings()
    model = model or settings.model
    think = settings.think if think is None else think
    return lambda messages, schema, on_text=None: chat(
        messages, schema, model=model, keep_alive=keep_alive, on_text=on_text, think=think
    )
