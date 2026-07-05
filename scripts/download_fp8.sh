#!/bin/bash
# 下载脚本：从 ModelScope 下载 fp8 模型
# 修改 URL 和输出路径即可

set -e

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

# === 修改这里的下载链接和目标文件名 ===

# 1. fp8 主模型 (~12GB)
URL1="https://www.modelscope.cn/models/XLabs-AI/flux-dev-fp8/resolve/main/flux-dev-fp8.safetensors"
OUT1="$MODELS_DIR/flux-dev-fp8.safetensors"

# 2. 量化配置文件 (~3KB)
URL2="https://www.modelscope.cn/models/XLabs-AI/flux-dev-fp8/resolve/main/flux_dev_quantization_map.json"
OUT2="$MODELS_DIR/flux_dev_quantization_map.json"

# === 下载 ===

echo "Downloading fp8 model..."
wget -c -O "$OUT1" "$URL1"
echo "Saved to $OUT1"

echo "Downloading quantization map..."
wget -c -O "$OUT2" "$URL2"
echo "Saved to $OUT2"

echo "Done!"
ls -lh "$OUT1" "$OUT2"