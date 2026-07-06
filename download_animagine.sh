#!/bin/bash
# 从 ModelScope 下载 Animagine XL 3.1（动漫微调 SDXL 模型，~7GB）
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

echo "=== 从 ModelScope 下载 Animagine XL 3.1 ==="
echo ""

python -c "
from modelscope import snapshot_download
print('正在下载（~7GB，不需要代理）...')
snapshot_download('cagliostrolab/animagine-xl-3.1', local_dir='models/animagine-xl-3.1')
print('下载完成')
"