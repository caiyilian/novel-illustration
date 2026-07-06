#!/bin/bash
# 一键下载 InstantID 所需文件（含断点续传）
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

unset HF_HUB_OFFLINE
export http_proxy=http://172.31.102.132:7890
export https_proxy=http://172.31.102.132:7890

echo "=== 1. 克隆 InstantID 仓库 ==="
if [ ! -d "InstantID" ]; then
    git clone https://github.com/instantX-research/InstantID.git
else
    echo "InstantID 已存在，跳过"
fi

echo "=== 2. 下载模型权重 ==="
mkdir -p InstantID/checkpoints/ControlNetModel

# config.json（小文件）
if [ ! -f "InstantID/checkpoints/ControlNetModel/config.json" ]; then
    echo "下载 config.json..."
    curl -L --noproxy '*' -o InstantID/checkpoints/ControlNetModel/config.json \
        "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/config.json"
fi

# ip-adapter.bin
if [ ! -f "InstantID/checkpoints/ip-adapter.bin" ]; then
    echo "下载 ip-adapter.bin (约 1.6GB, 可能需要几分钟)..."
    curl -L -C - -o InstantID/checkpoints/ip-adapter.bin \
        "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin"
else
    echo "ip-adapter.bin 已存在"
fi

# ControlNet model（大文件，带续传）
echo "下载 ControlNet 模型 (约 2.3GB, 可能较慢, 支持断点续传)..."
curl -L -C - --retry 5 --retry-delay 10 -o InstantID/checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors \
    "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors"

echo "=== 3. 检查 antelopev2 ==="
if [ -d "models/antelopev2" ]; then
    echo "antelopev2 已存在"
    mkdir -p InstantID/models
    ln -sf "$DIR/models/antelopev2" "$DIR/InstantID/models/antelopev2"
else
    echo "错误: models/antelopev2 不存在"
    exit 1
fi

echo ""
echo "=== 下载完成 ==="
ls -lh InstantID/checkpoints/ControlNetModel/
ls -lh InstantID/checkpoints/ip-adapter.bin