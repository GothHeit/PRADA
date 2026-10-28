#!/bin/sh
# Install (or update) SearXNG, the metasearch server PRADA's web search uses, as a plain
# Python app in its own venv: no Docker, no root, nothing system-wide. Run it again to update
# to the commit pinned below. start-node.sh starts it on 127.0.0.1:8888 next to the node.
# Needs git and python3 with venv (Ubuntu: sudo apt install git python3-venv python3-dev).
# Lives in $PRADA_SEARXNG_DIR, else <data dir>/searxng: src/, venv/, settings.yml.
set -eu

COMMIT=c360868e5b7c01e463a362a53518190d5a3b1097  # master on 2026-10-28
data="${PRADA_DATA_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/prada}"
dir="${PRADA_SEARXNG_DIR:-$data/searxng}"
mkdir -p "$dir"

if [ ! -d "$dir/src/.git" ]; then
    git clone --filter=blob:none https://github.com/searxng/searxng "$dir/src"
fi
git -C "$dir/src" fetch --quiet origin
git -C "$dir/src" -c advice.detachedHead=false checkout --quiet "$COMMIT"

[ -x "$dir/venv/bin/python" ] || python3 -m venv "$dir/venv"
pip="$dir/venv/bin/pip"
"$pip" install --quiet -U pip setuptools wheel pyyaml msgspec typing-extensions pybind11
(cd "$dir/src" && "$pip" install --quiet --use-pep517 --no-build-isolation -e .)

if [ ! -f "$dir/settings.yml" ]; then
    key="$("$dir/venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))')"
    cat >"$dir/settings.yml" <<YAML
# SearXNG for PRADA: only the node talks to it, on 127.0.0.1. Everything not set here comes
# from SearXNG's own defaults (searx/settings.yml).
use_default_settings: true

general:
  instance_name: "PRADA search"

search:
  safe_search: 1
  formats: [html, json]   # PRADA reads the JSON

server:
  bind_address: "127.0.0.1"
  port: 8888
  secret_key: "$key"
  limiter: false          # no bots to keep out: nothing but the node can reach it
  image_proxy: false
  public_instance: false

valkey:
  url: false              # the limiter is off, so no Valkey server is needed
YAML
    chmod 600 "$dir/settings.yml"
fi

echo "SearXNG $COMMIT is installed in $dir."
echo "Start it with scripts/start-node.sh (or by hand:"
echo "  SEARXNG_SETTINGS_PATH=$dir/settings.yml $dir/venv/bin/python -m searx.webapp)"
