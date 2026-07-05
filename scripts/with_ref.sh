#!/bin/bash
# 有参考图模式 — 使用 PuLID v1.1 + SDXL 保持角色长相
# Usage: bash scripts/with_ref.sh <ref_image> "<prompt>" <output> [device]

DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$DIR/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 如需要 HTTP 代理（首次运行以下载 EVA-CLIP 等模型），取消注释并填写代理 IP
# export http_proxy=http://<代理IP>:7890
# export https_proxy=http://<代理IP>:7890

REF=${1:?用法: $0 <参考图路径> "<prompt>" [输出文件] [device]}
PROMPT=${2:?用法: $0 <参考图路径> "<prompt>" [输出文件] [device]}
OUTPUT=${3:-output.png}
DEVICE=${4:-cuda:2}

python "$DIR/scripts/generate_illustration.py" \
    --ref "$REF" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --device "$DEVICE"