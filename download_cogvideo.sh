#!/bin/bash
# 从 ModelScope 下载 CogVideoX-2B（Apache 2.0，~4GB）
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

echo "=== 下载 CogVideoX-2B（Apache 2.0，~4GB）==="

python -c "
from modelscope import snapshot_download
print('正在下载...')
snapshot_download('ZhipuAI/CogVideoX-2b', local_dir='models/CogVideoX-2b')
print('下载完成')
"