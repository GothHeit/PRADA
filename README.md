# PRADA

**Personal Reasoning Autonomous Distributed Agent** — a local personal agent: a set of specialized modules that reason and act, designed as a distributed system.

Early construction. The first working module is an **offline language tutor**.

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

`prada wake` sends a Wake-on-LAN magic packet (standard library only) from any machine on your LAN. Enable Wake-on-LAN in the PC's BIOS and network adapter, and prefer a wired connection.

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

`PRADA_FAKE_LLM=1` swaps Ollama for a canned client, so tests and smoke runs never need a model.

## Roadmap

- Corrector → **interlocutor**: respond to *meaning*, not just grade the form
- **SQLite** learning log → per-category progress over time
- **Directed drills** for clean skill development
- **LoRA** fine-tuning on a measured eval set
- **Voice** I/O — voice is the north star

---

*Early-stage personal project.*
