# PRADA

**Personal Reasoning Autonomous Distributed Agent** — a local personal agent: a set of specialized modules that reason and act, designed as a distributed system.

Early construction. The first working module is an **offline language tutor**.

## How it fits together

PRADA is a **home node** that hosts modules. You talk to PRADA, and a **router** (one structured-output call) picks the module that should answer. Each module implements a small contract (`prada/module.py`): a name, a one-line description the router reads, and `handle(request) -> reply` with typed Pydantic messages. The node owns what modules share: the model client, one SQLite store (each module gets its own tables) and per-conversation state.

```bash
prada node        # run the home node on 127.0.0.1:7723 (stdlib HTTP + JSON, localhost only)
prada chat        # talk to PRADA; each message goes to the right module
prada tutor       # talk to the tutor directly
```

The node keeps the model loaded between requests, and the CLI is a thin client of it. If no node is running the CLI works in-process instead, so every command also works on its own. Write `/tutor ...` or `/wake` in `prada chat` to skip the router. The port comes from `--port`, `PRADA_NODE_PORT` or `[node] port` in the config file.

## Modules

### Tutor — offline language tutor

A conversational tutor for Japanese, French, and Russian that runs entirely on your machine — no cloud, no API keys. It checks your grammar, explains the rule in plain English, and keeps the conversation going in the language you're learning.

- **Local inference** via [Ollama](https://ollama.com) + `qwen2.5:14b`, at temperature 0 for determinism.
- **Structured output** (JSON Schema) validated with **Pydantic v2** — the model fills a typed `TutorTurn` contract instead of returning free text.
- A **two-layer design**: a silent *analysis* layer (verdict, error type, correction, English explanation) and a *surface* layer (`spoken`) — the single natural message the student actually hears. The scaffolding stays invisible so the exchange reads as a conversation.
- **Reasoning-first**: the schema emits a chain-of-thought field *before* the verdict, so classification follows the analysis.

## Quickstart

Prerequisite: [Ollama](https://ollama.com) running locally.

```bash
ollama pull qwen2.5:14b          # get the model

python -m venv tutor/.venv        # set up the module
source tutor/.venv/bin/activate
pip install -r tutor/requirements.txt

python -m tutor.cli --lang ru     # talk to the tutor (ru, fr or ja)
```

```
> Я читаю книга.
Almost — it should be "Я читаю книгу". Here книга is the direct object, so it
takes the accusative case (книга → книгу). Что ты любишь читать?
```

## Wake the PC

The `wake` module (say "wake up the PC" in `prada chat`) and `prada wake` send a Wake-on-LAN magic packet (standard library only) from any machine on your LAN. Enable Wake-on-LAN in the PC's BIOS and network adapter, and prefer a wired connection.

```bash
pip install -e .
prada wake --mac aa:bb:cc:dd:ee:ff --broadcast 192.168.1.255
```

Settings come from the command line, then `PRADA_WAKE_MAC` / `PRADA_WAKE_BROADCAST` / `PRADA_WAKE_PORT`, then `~/.config/prada/config.toml` (or `$PRADA_CONFIG`):

```toml
[wake]
mac = "aa:bb:cc:dd:ee:ff"
broadcast = "192.168.1.255"
```

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m pytest
echo "Я читаю книгу." | PRADA_FAKE_LLM=1 python -m tutor.cli
```

`PRADA_FAKE_LLM=1` swaps Ollama for canned clients (always in-process, never a running node), so tests and smoke runs never need a model.

## Eval

`prada eval [--lang ru]` runs the tutor over the labeled sentences in `tutor/evalset/` and prints verdict, error-category and correction accuracy per language. Run it against the real model before and after a prompt or model change; with `PRADA_FAKE_LLM=1` it only checks the plumbing.

## Roadmap

- Corrector → **interlocutor**: respond to *meaning*, not just grade the form
- **Voice** loop on the home node, then a wake word
- **Directed drills** for clean skill development
- **LoRA** fine-tuning on a measured eval set
- More modules: music, anime streaming, home devices

---

*Early-stage personal project.*
