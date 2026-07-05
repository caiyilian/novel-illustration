#!/bin/bash
# 启动插图生成 API 服务
# Usage: bash scripts/start_api.sh [port] [device]

DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$DIR/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PORT=${1:-8000}
DEVICE=${2:-cuda:2}

python "$DIR/scripts/api_server.py" --port "$PORT" --device "$DEVICE"