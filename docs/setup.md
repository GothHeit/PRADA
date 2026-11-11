# PRADA setup and reference

The details behind the README's quick start: running on a Windows PC, voice, web search, settings and evals. Every setting lives in `~/.config/prada/config.toml` (or `$PRADA_CONFIG`), outside the repo, and every key is optional.

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
& $HOME\prada-voice\Scripts\pip install "\\wsl$\Ubuntu\home\you\projetos\PRADA[voice]"
& $HOME\prada-voice\Scripts\prada voice setup     # download the models once (the only network use)
& $HOME\prada-voice\Scripts\prada voice devices   # list microphones and speakers
& $HOME\prada-voice\Scripts\prada voice           # listen; Ctrl+C to stop
```

Settings go in `[voice]` of the config file (on Windows, `%USERPROFILE%\.config\prada\config.toml`); every key is optional:

```toml
[voice]
wake_words = ["hey_jarvis", "~/prada/wake_up_prada.onnx"]  # pretrained names or your own .onnx models
threshold = 0.5             # wake word score that counts as heard
whisper_model = "small"     # whisper.cpp model name (multilingual) or path to a ggml .bin
language = "auto"           # or "en", "ru", ...
whisper_prompt = "Kitajima, Monika"  # names whisper should spell right
tts = "piper"               # who speaks: "piper", "kokoro" or "qwen" (see Her voice)
speed = 1.0                 # speaking speed for Piper and Kokoro (0.5-2)
piper_voice = "en_US-amy-medium"
input_device = "Microphone (USB Audio)"   # name or index from `prada voice devices`
silence = 1.2               # seconds of quiet that end what you're saying (raise it if she cuts you off)
follow_up = 6.0             # seconds she keeps listening after a reply, no wake word needed (0 = off)
max_follow_ups = 20         # replies in a row without a wake word (0 = no limit)
barge_in = true             # say a wake word while she talks to stop her and ask something else

[voice.voices]              # sentences in these languages get their own voice
ru = "ru_RU-irina-medium"
fr = "fr_FR-siwis-medium"

[voice.say_as]              # respell words so the voice says them right (whole words, any case)
Kafka = "Kahfka"
```

Write Windows paths in single quotes (`'C:\Users\you\wake_up_prada.onnx'`): in double quotes TOML reads `\U` as an escape. Any number of wake words listen at once. The pretrained ones are `alexa`, `hey_jarvis`, `hey_mycroft` and `hey_rhasspy`. "Wake Up Prada" needs its own model (see [Your own wake words](#your-own-wake-words)). Models live in `<data dir>/voice` (`%USERPROFILE%\.local\share\prada\voice` on Windows); a bare name in `wake_words` is a model there if `<name>.onnx` exists, else a pretrained one. With `PRADA_FAKE_LLM=1`, `prada voice` reads requests from stdin and prints the replies, so it runs without audio or models.

### Her voice

Three engines can speak for her, picked by `tts`. Piper is the default and the fallback: if another engine won't load, `prada voice` says so and uses Piper, and Piper's `[voice.voices]` still speak the languages the other engine can't.

- **Piper** (`voice` extra): fast and light, a bit robotic. One voice per language.
- **Kokoro** (`kokoro` extra, `kokoro_voice = "af_heart"`): Kokoro-82M on the CPU, so no GPU memory taken from the brain, and much more natural. The first letter of the voice name is its accent (`a` American, `b` British). It speaks French itself (`ff_siwis`); Russian and Japanese go to Piper's `[voice.voices]` (kokoro-onnx reads kanji as "Chinese letter").
- **Qwen3-TTS** (`qwen` extra, NVIDIA GPU): clones a short clip of her, so one voice speaks English, Japanese, French, Russian, Portuguese and more. The 0.6B model takes a few GB of GPU memory next to the brain. Install PyTorch with CUDA first (pytorch.org), then the extra.

```powershell
& $HOME\prada-voice\Scripts\pip install "\\wsl$\Ubuntu\home\you\projetos\PRADA[voice,kokoro,qwen]"
& $HOME\prada-voice\Scripts\prada voice design "A young woman in her late teens, warm and gentle, playful and confident, clear soft voice, smiling as she talks"
& $HOME\prada-voice\Scripts\prada voice setup --tts kokoro   # fetch an engine's model (default: the one in tts)
& $HOME\prada-voice\Scripts\prada voice setup --tts qwen
& $HOME\prada-voice\Scripts\prada voice try --tts kokoro
& $HOME\prada-voice\Scripts\prada voice try --tts qwen "Say anything you like."
```

`prada voice design` describes her voice to Qwen3-TTS VoiceDesign (1.7B, a ~4 GB download on first use) and saves takes `her-1.wav`, `her-2.wav`, ... with what each says (`.txt`) in the voice models folder. Stop the node or `ollama stop` the brain first so it fits in GPU memory. Pick the take you like and set `qwen_voice = "her-2"` (or a path to any clean 5-15 s `.wav` with its transcript in a `.txt` next to it; keep such clips on your PC). Without the `.txt` she copies only the clip's timbre: less exact, but a noisy or glitchy clip's flaws don't carry over. `prada voice try` says a fixed set of test lines (her name, numbers, French, Japanese, Russian, a tease) with each engine and prints how long each took to start, so engines are compared by ear on the same lines. Qwen improvises each line a little differently; if some lines come out wrong, lower `qwen_temperature` (default 0.8) for steadier takes, or raise it for livelier ones.

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

## Web search

General search goes through your own [SearXNG](https://github.com/searxng/searxng). Install it in WSL once (git and `python3-venv` needed; no Docker, no root): `scripts/install-searxng.sh` clones it at a pinned commit into its own venv in `~/.local/share/prada/searxng`, with a `settings.yml` that binds it to 127.0.0.1:8888, turns on its JSON output and turns off the limiter. `scripts/start-node.sh` then starts it next to the node at boot (log: `searxng.log`). Run the install script again to update. Optional settings:

```toml
[web]
searxng_url = "http://127.0.0.1:8888"
github_token = ""                     # fine-grained, read-only, no repository access needed
max_pages = 3                         # results she may read per question (0-3)
sources = ["web", "reddit", "github"] # the ones she may use
```

## Music

She plays songs by name from Corpus or Spotify (below), and pauses, resumes and skips whatever plays in Spotify, Corpus or any other player that shows up in Windows' media controls (the overlay with the volume keys), and says what's playing. Plain commands skip the model, in English or Portuguese: "pause", "next song", "go back a song", "what's playing?", "pausa", "próxima", "que música é essa?"; add "on Spotify" / "no Corpus" to pick a player. She says it happened only after the player shows the change.

The node reaches Windows through WSL interop: it keeps one `powershell.exe` running with a small script (`prada/music/bridge.ps1`, run inline, so the execution policy doesn't matter). Nothing to install or configure. `prada music` prints what Windows sees, to check the bridge.

### Play by name on Spotify

"Play Muse Space Dementia", "play the album Origin of Symmetry", "play my chill playlist", "play something calm", "toca Evidências do Chitãozinho e Xororó", "bota pra tocar Muse": she reads the request in code (no model call), searches Spotify, starts the closest match and says what she put on and where, only after Spotify shows it playing. A bare name ("play Muse") plays the artist; "something calm" plays the best Spotify playlist for the mood. If nothing comes close she says so instead of playing something random.

It plays on the Spotify device that is already playing, else this PC's Spotify app, else any of your devices (Spotify Connect: phone, speakers). If none is open, she opens the Spotify app (Store or classic) and waits up to 20 s for it.

This uses Spotify's official Web API (needs Premium): her one deliberate cloud call, since the music lives on Spotify anyway. Nothing is patched. Set it up once:

1. On [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard), log in with your Spotify account and **Create app**: any name and description, **Redirect URI** `http://127.0.0.1:7725/callback` (exactly this; Spotify doesn't accept `localhost`), and tick **Web API**. Save.
2. Open the app's **Settings** and copy its **Client ID** (no secret is needed) into your config:

   ```toml
   [spotify]
   client_id = "0123456789abcdef0123456789abcdef"
   ```

3. Where the node runs (WSL), run `prada music login`. It opens Spotify's consent page in your browser (or prints the address); click **Agree**, and the page says PRADA is logged in. The refresh token is saved in `spotify-token.json` next to the config file, readable only by you; never commit it. If Spotify says the user isn't registered, add your account's email under **User Management** in the app's settings and log in again.
4. `prada music spotify` shows the login, your devices and what plays. Restart `prada node`.

To log out, delete `spotify-token.json` (and remove the app's access at spotify.com/account/apps).

While `prada node` runs, every track Spotify and Corpus play is logged in her database (table `music_play`: title, artist, album, seconds listened, finished / skipped / stopped, and whether she started it), for learning your taste later. It stays on this computer.

### Play your own music from Corpus

When what you ask for is clearly in your own files, Corpus plays it instead of Spotify: "play Space Dementia", "toca o álbum Origin of Symmetry", "shuffle Muse", "toca um rock" (a genre), "play something calm" (moods go by genre, since Corpus has no mood tags: calm, energetic, focus, happy, sad). A song plays alone, an album in order, an artist, genre or mood on shuffle; "on shuffle" / "em ordem" change that. "Add Unravel to the queue" / "coloca Unravel na fila" and "play Take a Bow next" add to Corpus' queue (Spotify doesn't queue). Names are matched in code, in any word order and with small mishearings. When a song and an album share a name she asks which one you mean. "On Spotify" / "no Corpus" picks the app; anything not in Corpus goes to Spotify.

Corpus plays in its strip without opening the viewer or taking focus (and starts minimized if it wasn't running). She says it's playing only once Windows' media controls show Corpus playing one of those songs, and pauses Spotify then; when Spotify starts something, she pauses Corpus.

### Corpus' music catalog

To play your own music by name, PRADA keeps a copy of Corpus' music catalog (title, artist, album, genre and path of each song; table `music_track`). It reads it with Corpus' own `--export-music` command, which never changes Corpus' library, and reads it again every 6 hours while the node runs. Tell PRADA where Corpus is if `Corpus.exe` isn't on your PATH:

```toml
[music]
corpus = 'C:\Apps\Corpus\Corpus.exe'  # a Windows path; single quotes keep the backslashes
```

`prada music import` reads the catalog right away and says how many songs it found.

## Wake the PC from another machine

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

## Brain model

Every module talks to one Ollama model, `gemma4:12b` (thinking off) unless the config file says otherwise:

```toml
[llm]
model = "qwen3.5:9b"
think = false   # thinking models (qwen3.5) reason before answering; false skips it
```

`PRADA_LLM_MODEL` and `PRADA_LLM_THINK` (`on`/`off`) override the file. The default model runs with `think = false`; for any other model leave `think` out unless it is a thinking model. `prada node` prints the model it uses.

`PRADA_FAKE_LLM=1` swaps Ollama for canned clients (always in-process, never a running node), so tests and smoke runs never need a model.

## Eval

`prada eval [--lang ru] [--model qwen3.5:9b] [--think off]` runs the tutor over the labeled sentences in `tutor/evalset/` and prints verdict, error-category and correction accuracy and seconds per sentence for each language. Run it against the real model before and after a prompt or model change; `--model` and `--think` try another model without touching the config, so two brains can be compared side by side. With `PRADA_FAKE_LLM=1` it only checks the plumbing.

`prada eval --suite understand [--model ...] [--think off] [--backend NAME]` runs the hard cases in `prada/evalset/understand.jsonl`: each one sets a situation (what Spotify and Corpus are playing, her last line, a search she just offered), plays earlier turns, then says one more thing, and checks the calls she made (a tool and its key words, such as `music.play` "space dementia" in Corpus) or that she did nothing. They cover corrections vs changes of mind, polite and indirect requests, references ("skip this one", "the second one"), changing your mind mid-sentence, things that sound like music but aren't ("shuffle the deck"), notes, wake and search, two requests at once, and Portuguese. The report gives right calls (overall and per group), false actions, missed actions, wrong calls, false claims, invalid JSON and milliseconds per turn for each stage (router, assistant, ... then code and actions). Players, Spotify, Corpus and the web are fakes and the PC is never woken. A transcript and a `.json` are saved as for the assistant suite (below), and `--compare` works on them too.

`prada eval --suite assistant [--model qwen3.5:9b] [--think off]` scores her as an assistant instead: each case in `prada/evalset/assistant.jsonl` runs through a fresh node with a throwaway database, and the report gives the pass rate for routing (notes, wake, web, tutor mid-lesson), honesty (never claiming an action that didn't run), search (offering to look up live or unknown things, answering known facts directly) and memory (using a seeded past conversation), plus invalid JSON replies and seconds per turn. Web searches use canned offline sources. Every exchange is also written to a Markdown transcript (in the data folder under `eval/`, or `--transcript FILE`) so you can judge how natural each model sounds. A `.json` with the scores and every case's verdict is saved next to it, and `prada eval --compare A.json B.json ...` shows saved runs side by side, then the cases where they got different verdicts.

### When she gets something wrong

The node records every turn (the flight recorder): what she heard, the path it took (a code shortcut like "notes wants it", or the router's pick), each model call with its time and tokens, the actions and how they ended, what she said and the total time. `prada trace` shows the last turns (`--last 20` for more), and `prada trace --case N` prints turn N as a line for `prada/evalset/assistant.jsonl`, with the turns just before it in the same conversation as `before`. The line holds what she did, so set `module` (or `says`) to what she should have done before pasting it. Only the last 14 days are kept, in the same database as her memory.

### Cloud models (evals only)

To compare her local brain with a hosted model, set up a backend for any OpenAI-compatible chat completions API (OpenAI, Gemini, OpenRouter, Groq, Mistral, or a local llama.cpp / LM Studio server):

```toml
[backends.openai]
url = "https://api.openai.com/v1"
model = "gpt-5-mini"            # default; --model overrides it
api_key_env = "OPENAI_API_KEY"  # or api_key = "..." (this file only, never the repo)
json = "schema"                 # "object" for providers without structured output
# temperature = 0               # like Ollama's runs; leave unset for reasoning models (gpt-5, o-series reject it)
```

Then `prada eval --backend openai [--model ...]` (tutor) or `prada eval --suite assistant --backend openai` runs the same cases against it; the report, transcript and `.json` name the backend and model. Only evals use backends: chat, voice and the node always use Ollama, whatever the config says. The cases are made up, but her persona prompt goes out with them; point `PRADA_PERSONA` at a missing file to send the neutral default instead. `--think` only applies to Ollama. Memory search embeddings stay on the local Ollama. Under `PRADA_FAKE_LLM` nothing is sent.

## Development

```bash
pip install -e ".[dev]"
ruff check . && ruff format --check . && python -m pytest
echo "Я читаю книгу." | PRADA_FAKE_LLM=1 python -m tutor.cli
```
