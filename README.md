# PRADA

**Personal Reasoning Autonomous Distributed Agent** — a local personal agent: a set of specialized modules that reason and act, designed as a distributed system.

Early construction. The first working module is an **offline language tutor**.

## How it fits together

PRADA is a **home node** that hosts modules. You talk to PRADA, and a **router** (one structured-output call) picks the module that should answer. Each module implements a small contract (`prada/module.py`): a name, a one-line description the router reads, and `handle(request) -> reply` with typed Pydantic messages. The node owns what modules share: the model client, one SQLite store (each module gets its own tables) and per-conversation state, including the conversation itself, so every module speaks as the same PRADA.

Modules never act on the world directly. A module returns **typed actions** (Pydantic models) and lists the executors it may use; the node runs each action, checks the result (a saved note is read back, for example) and only then lets PRADA say it's done. If an action fails or isn't allowed, she says so instead. Her one hard rule, never claim something she didn't do, is enforced by the code, not just the prompt.

```bash
prada node        # run the home node on 127.0.0.1:7723 (stdlib HTTP + JSON, localhost only)
prada chat        # talk to PRADA; each message goes to the right module
prada tutor       # talk to the tutor directly
```

The node keeps the model loaded between requests, and the CLI is a thin client of it. If no node is running the CLI works in-process instead, so every command also works on its own. Write `/tutor ...` or `/wake` in `prada chat` to skip the router. The port comes from `--port`, `PRADA_NODE_PORT` or `[node] port` in the config file.

## Start at boot

On a Windows PC the brain (`prada node` and Ollama) lives in WSL, and a Task Scheduler task starts it when Windows boots, before anyone logs in (no autologin), and checks on it after every wake from sleep. Set up the repo in WSL first (a `.venv` with `pip install -e .`), then, from an elevated PowerShell:

```powershell
cd \\wsl$\Ubuntu\home\you\projetos\PRADA\windows
powershell -ExecutionPolicy Bypass -File .\install-autostart.ps1   # register the task and start it now
powershell -ExecutionPolicy Bypass -File .\check-node.ps1          # is the node up on 127.0.0.1:7723?
powershell -ExecutionPolicy Bypass -File .\uninstall-autostart.ps1 # remove the task and stop the node
```

To run without you logged in, the task stores your Windows password: the install script asks for it once (for a Microsoft account, the account password, not the PIN). If you change that password, run the install script again. `-S4U` registers the task without a password, but WSL may not start under that logon; `check-node.ps1` shows the task's last result if it doesn't. Running the script again also replaces the older start-at-login task.

Run from a `\\wsl$` path, the install script works out the distro and repo path itself; otherwise pass `-Distro Ubuntu -RepoPath /home/you/projetos/PRADA`. `-Port` changes the port. The task runs `scripts/start-node.sh`, which starts Ollama if it isn't already running and then the node; logs go to `~/.local/share/prada/node.log` and `ollama.log` in WSL. It starts 30 s after boot and runs again 15 s after each wake: a node that survived sleep is left alone, one that died is started again. Inside WSL, `prada health [--wait 30]` does the same check as `check-node.ps1`. Windows reaches the node through WSL's localhost forwarding (on by default). Microphone and speakers run on the Windows side, talking to this node (see [Voice](#voice)); that part still starts at login.

### Sleep and Wake-on-LAN

The PC can sleep when idle and be woken over the network (by `prada wake` from another machine, later by a Raspberry Pi). `setup-sleep-wake.ps1` sets the Windows side, from an elevated PowerShell (`-DryRun` only shows the changes):

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-sleep-wake.ps1 -SleepMinutes 30
```

It makes the power plan sleep after that many idle minutes on AC and never hibernate on its own (most network cards can't wake from hibernation), and lets each wired card wake the PC for a magic packet only, not for ordinary traffic. It prints the old timeouts first and ends with the card's MAC address, which goes in the waker's `[wake] mac`. To undo, run `powercfg /change standby-timeout-ac <old minutes>` (and `hibernate-timeout-ac`), and untick the wake options on the card's Power Management tab in Device Manager. Wake-on-LAN needs Ethernet; Wi-Fi rarely supports it.

Nothing is changed in the BIOS/UEFI. If the PC doesn't wake, check there (names vary by board): turn on "Wake on LAN" / "Power On By PCI-E" / "Resume by PCI-E device", and turn off "ErP" / "EuP Ready" / "Deep Sleep", which cut power to the card. `powercfg /a` (printed by the script) shows whether the PC uses classic sleep (S3) or Modern Standby (S0); with Modern Standby, waking depends on the card and driver. Windows Fast Startup only matters for waking a PC that was shut down, not one that sleeps.

## Voice

Say a wake word, then speak: PRADA hears you with [openWakeWord](https://github.com/dscripka/openWakeWord), transcribes with [whisper.cpp](https://github.com/ggml-org/whisper.cpp) and answers with a [Piper](https://github.com/OHF-Voice/piper1-gpl) voice, all offline. `prada voice` is a client of the node, like `prada chat`, so it runs where the microphone and speakers are. On the Windows PC that is Windows itself (audio inside WSL is unreliable), reaching the node in WSL at `127.0.0.1:7723` through localhost forwarding.

In a Windows terminal, with Python 3.11+ (the venv lives on the Windows side, outside WSL):

```powershell
py -m venv $HOME\prada-voice
$HOME\prada-voice\Scripts\pip install "\\wsl$\Ubuntu\home\you\projetos\PRADA[voice]"
$HOME\prada-voice\Scripts\prada voice setup     # download the models once (the only network use)
$HOME\prada-voice\Scripts\prada voice devices   # list microphones and speakers
$HOME\prada-voice\Scripts\prada voice           # listen; Ctrl+C to stop
```

Settings go in `[voice]` of the config file (on Windows, `%USERPROFILE%\.config\prada\config.toml`); every key is optional:

```toml
[voice]
wake_words = ["hey_jarvis", "~/prada/wake_up_prada.onnx"]  # pretrained names or your own .onnx models
threshold = 0.5             # wake word score that counts as heard
whisper_model = "small"     # whisper.cpp model name (multilingual) or path to a ggml .bin
language = "auto"           # or "en", "ru", ...
piper_voice = "en_US-amy-medium"
input_device = "Microphone (USB Audio)"   # name or index from `prada voice devices`
silence = 0.8               # seconds of quiet that end what you're saying
follow_up = 6.0             # seconds she keeps listening after a reply, no wake word needed (0 = off)

[voice.voices]              # sentences in these languages get their own voice
ru = "ru_RU-irina-medium"
fr = "fr_FR-siwis-medium"
```

Write Windows paths in single quotes (`'C:\Users\you\wake_up_prada.onnx'`): in double quotes TOML reads `\U` as an escape. Any number of wake words listen at once. The pretrained ones are `alexa`, `hey_jarvis`, `hey_mycroft` and `hey_rhasspy`. "Wake Up Prada" needs its own model (see below). Models live in `<data dir>/voice` (`%USERPROFILE%\.local\share\prada\voice` on Windows); a bare name in `wake_words` is a model there if `<name>.onnx` exists, else a pretrained one. With `PRADA_FAKE_LLM=1`, `prada voice` reads requests from stdin and prints the replies, so it runs without audio or models.

### Your own wake words

`prada voice train "wake up prada"` trains a wake word model on your own machine with openWakeWord's recipe: [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) says the phrase, and phrases that sound like it, in hundreds of synthetic voices, the clips get room echo and background noise, and a small model learns to tell them from 2,000 hours of everyday audio. It runs on Linux (WSL on the Windows PC), with Python 3.11 (openwakeword needs `tflite-runtime` on Linux, which stops at 3.11), in its own venv since PyTorch takes several GB:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv fetches Python 3.11 if the distro lacks it
uv venv -p 3.11 ~/.venvs/prada-train
VIRTUAL_ENV=~/.venvs/prada-train uv pip install -e ".[wakeword]"
~/.venvs/prada-train/bin/prada voice train --setup             # download ~17 GB of training data, once
~/.venvs/prada-train/bin/prada voice train "wake up prada" --out /mnt/c/Users/you/.local/share/prada/voice
```

After `--setup` training is offline. With an NVIDIA GPU (PyTorch uses it from WSL) a run takes roughly one to two hours; on CPU only, plan for a night. Stop it any time and run the same command to continue. `--out` writes the model straight into the Windows voice models folder, so `wake_words = ["wake_up_prada", "hey_jarvis"]` finds it. Train more phrases the same way, one model each. `--samples` (default 25000) and `--steps` (default 50000) trade time for accuracy; WAV recordings of your own room (16 kHz) in `~/.local/share/prada/wakeword/background/<any folder>/` teach it what to ignore. If it wakes too easily or too rarely, raise or lower `threshold`.

## Modules

### Tutor — offline language tutor

A conversational tutor for Japanese, French, and Russian that runs entirely on your machine — no cloud, no API keys. It checks your grammar, explains the rule in plain English, and keeps the conversation going in the language you're learning.

- **Local inference** via [Ollama](https://ollama.com) + `qwen2.5:14b`, at temperature 0 for determinism.
- **Structured output** (JSON Schema) validated with **Pydantic v2** — the model fills a typed `TutorTurn` contract instead of returning free text.
- A **two-layer design**: a silent *analysis* layer (verdict, error type, correction, English explanation) and a *surface* layer (`spoken`) — the single natural message the student actually hears. The scaffolding stays invisible so the exchange reads as a conversation.
- **Reasoning-first**: the schema emits a chain-of-thought field *before* the verdict, so classification follows the analysis.

### Notes — things to remember

Say "note that the keys are in the drawer" (or `/notes ...` in `prada chat`) and PRADA saves it to the shared store, reads it back to make sure it's there, and only then confirms. "What are my notes?" reads the last ten back.

## Quickstart

Prerequisite: [Ollama](https://ollama.com) running locally.

```bash
ollama pull qwen2.5:14b          # get the model

python -m venv .venv              # set up PRADA
source .venv/bin/activate
pip install -e .

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
- A trained **"Wake Up Prada"** wake word, and voice that starts at login
- **Directed drills** for clean skill development
- **LoRA** fine-tuning on a measured eval set
- More modules: music, anime streaming, home devices

---

*Early-stage personal project.*
