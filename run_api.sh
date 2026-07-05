#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" && source .venv/bin/activate && exec python scripts/api_server.py --port 8001 --device "${DEVICE:-cuda:0}"