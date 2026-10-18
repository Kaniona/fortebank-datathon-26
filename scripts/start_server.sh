#!/usr/bin/env bash
# Запуск сервиса. Слушает только localhost: наружу публикуется через туннель (scripts/start_tunnel.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
export YOLO_AUTOINSTALL=False HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # никаких обращений в сеть при работе
exec python -m uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1
