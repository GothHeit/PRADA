#!/bin/sh
# Start the PRADA home node inside WSL; the Windows boot task runs this.
# If systemd manages Ollama (the official installer's ollama.service, with its own model store),
# leave it to systemd: it may still be starting, and a second `ollama serve` as this user
# would look in ~/.ollama and not find the models. Otherwise start Ollama if nothing answers on
# its port. Then run the node in the foreground,
# so the task (and the WSL distro) stays up for as long as the node does.
# Extra arguments go to `prada node` (e.g. --port 7723). Logs: $PRADA_DATA_DIR or
# ~/.local/share/prada, files node.log and ollama.log.
set -eu
cd "$(dirname "$0")/.."

py="${PRADA_PYTHON:-}"
if [ -z "$py" ]; then
    for candidate in .venv/bin/python tutor/.venv/bin/python; do
        if [ -x "$candidate" ]; then
            py="$candidate"
            break
        fi
    done
fi
py="${py:-python3}"

logs="${PRADA_DATA_DIR:-$HOME/.local/share/prada}"
mkdir -p "$logs"

ollama_up() {
    "$py" -c 'import urllib.request as u
u.build_opener(u.ProxyHandler({})).open("http://127.0.0.1:11434", timeout=1)' 2>/dev/null
}

if systemctl is-enabled --quiet ollama 2>/dev/null; then
    :
elif command -v ollama >/dev/null 2>&1 && ! ollama_up; then
    nohup ollama serve >>"$logs/ollama.log" 2>&1 &
fi

echo "$(date '+%F %T') starting node with $py" >>"$logs/node.log"
exec "$py" -m prada node "$@" >>"$logs/node.log" 2>&1
