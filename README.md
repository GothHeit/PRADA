# PRADA

**Personal Reasoning Autonomous Distributed Agent.** A JARVIS-inspired assistant for the home that runs entirely on your own hardware: no cloud, no accounts, no API keys. A home node is the brain; you talk to her by voice or in the terminal, and she picks the right module to answer. She never claims to have done something she didn't: actions are checked in code before she says they're done.

Early construction, built for one home first.

## What she can do

- **Talk** by voice: wake word, offline speech recognition and a local voice. Say the wake word again to interrupt her.
- **Teach languages**: a Japanese, French and Russian tutor that corrects your grammar and keeps the conversation going.
- **Remember**: notes you ask her to keep, and the conversation you're having.
- **Search the web when you ask**, through your own SearXNG, and tell you where the answer came from.
- **Wake the PC** over the network, and start herself when it boots.

## Quick start

You need Python 3.11+ and [Ollama](https://ollama.com).

```bash
ollama pull gemma4:12b

python -m venv .venv && source .venv/bin/activate
pip install -e .

prada node    # the home node, on 127.0.0.1:7723
prada chat    # talk to her in the terminal (works without a node too)
```

For voice, install the `voice` extra where your microphone is, download the models once, and listen:

```bash
pip install -e ".[voice]"
prada voice setup
prada voice
```

## Your settings

Settings live in `~/.config/prada/config.toml` (or `$PRADA_CONFIG`), and who she is (name, character, tone) lives in `persona.toml` next to it. Both stay on your machine, never in the repo. `prada persona init` writes a persona to edit; without one she uses a short neutral default.

## More

[docs/setup.md](docs/setup.md) covers the rest: starting at boot on Windows, sleep and Wake-on-LAN, voice settings, training your own wake word, web search, choosing the model, evals and development.

Next up: a conversation partner rather than a corrector, directed drills, and more modules (music, anime, home devices).
