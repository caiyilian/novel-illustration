#!/bin/bash
# 从 ModelScope 下载 CogVideoX-5B（支持 I2V，非商用）
# 注意：5B 模型可能超 24GB 显存，先下载试试
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

echo "=== 下载 CogVideoX-5B（支持 I2V 图生视频）==="

python -c "
from modelscope import snapshot_download
print('正在下载...')
snapshot_download('ZhipuAI/CogVideoX-5b', local_dir='models/CogVideoX-5b')
print('下载完成')
"