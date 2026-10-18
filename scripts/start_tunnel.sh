#!/usr/bin/env bash
# Публичный HTTPS-адрес через Cloudflare Quick Tunnel (аккаунт не нужен).
# Адрес вида https://<случайное-имя>.trycloudflare.com меняется при каждом запуске туннеля.
set -euo pipefail
CF="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
PORT="${PORT:-8000}"
exec "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:${PORT}"
