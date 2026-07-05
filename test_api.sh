#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" && source .venv/bin/activate && exec python test_api.py