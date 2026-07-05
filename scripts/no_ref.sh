#!/bin/bash
# 无参考图模式 — 纯 SDXL 文生图
# Usage: bash scripts/no_ref.sh "<prompt>" <output> [device]

DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$DIR/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

PROMPT=${1:?用法: $0 "<prompt>" [输出文件] [device]}
OUTPUT=${2:-output.png}
DEVICE=${3:-cuda:2}

python "$DIR/scripts/generate_illustration.py" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --device "$DEVICE"