#!/bin/sh
# Start the PRADA home node inside WSL; the Windows boot task runs this.
# If systemd manages Ollama (the official installer's ollama.service, with its own model store),
# leave it to systemd: it may still be starting, and a second `ollama serve` as this user
# would look in ~/.ollama and not find the models. Otherwise start Ollama if nothing answers on
# its port. If SearXNG is installed (scripts/install-searxng.sh), start it on 127.0.0.1:8888
# unless it already answers. Then run the node in the foreground,
# so the task (and the WSL distro) stays up for as long as the node does.
# Extra arguments go to `prada node` (e.g. --port 7723). Logs: the data dir ($PRADA_DATA_DIR,
# else $XDG_DATA_HOME/prada, else ~/.local/share/prada), files node.log, ollama.log and
# searxng.log; a log over 5 MB is moved to <name>.1 at start.
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

logs="${PRADA_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/prada}"
mkdir -p "$logs"
for log in node ollama searxng; do
    if [ -f "$logs/$log.log" ] && [ "$(wc -c <"$logs/$log.log")" -gt 5000000 ]; then
        mv -f "$logs/$log.log" "$logs/$log.log.1"
    fi
done

answers() {
    "$py" -c 'import sys, urllib.request as u
u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=1)' "$1" 2>/dev/null
}
ollama_up() { answers http://127.0.0.1:11434; }

if systemctl is-enabled --quiet ollama 2>/dev/null; then
    :
elif command -v ollama >/dev/null 2>&1 && ! ollama_up; then
    nohup ollama serve >>"$logs/ollama.log" 2>&1 &
fi

searxng="${PRADA_SEARXNG_DIR:-$logs/searxng}"
if [ -x "$searxng/venv/bin/python" ] && ! answers http://127.0.0.1:8888/healthz; then
    SEARXNG_SETTINGS_PATH="$searxng/settings.yml" nohup "$searxng/venv/bin/python" -m searx.webapp \
        >>"$logs/searxng.log" 2>&1 &
fi

echo "$(date '+%F %T') starting node with $py" >>"$logs/node.log"
exec "$py" -m prada node "$@" >>"$logs/node.log" 2>&1
