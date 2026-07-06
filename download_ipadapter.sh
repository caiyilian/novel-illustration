#!/bin/bash
# 下载 IP-Adapter-FaceID 所需文件
# 大文件需手动下载，脚本会给出命令
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

unset HF_HUB_OFFLINE
export http_proxy=http://172.31.102.132:7890
export https_proxy=http://172.31.102.132:7890

echo "=== 1. 克隆 IP-Adapter 仓库 ==="
if [ ! -d "IP-Adapter" ]; then
    git clone https://github.com/tencent-ailab/IP-Adapter.git
else
    echo "IP-Adapter 已存在，跳过"
fi

echo "=== 2. 下载模型权重 ==="
mkdir -p IP-Adapter/models

echo ""
echo "请手动下载以下大文件，用下面的命令（你在终端跑就行）："
echo ""

FILES=(
    "ip-adapter-faceid_sdxl.bin"
    "ip-adapter-faceid_sdxl_lora.safetensors"
)

for f in "${FILES[@]}"; do
    if [ -f "IP-Adapter/models/$f" ]; then
        echo "  ✅ $f 已存在"
    else
        echo "  curl -L -C - -o IP-Adapter/models/$f \"https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/$f\""
    fi
done

echo ""
echo "下载完成后，目录结构应为："
echo "  IP-Adapter/"
echo "  ├── ip_adapter/          # 代码（已克隆）"
echo "  ├── models/"
echo "  │   ├── ip-adapter-faceid_sdxl.bin"
echo "  │   └── ip-adapter-faceid_sdxl_lora.safetensors"
echo "  └── ..."