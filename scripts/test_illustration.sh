#!/bin/bash
# 测试脚本：验证三种用例
# 1. 有参考图 → 角色长相一致
# 2. 无参考图 → 正常生成
# 3. 同一参考图 + 不同 prompt → 角色长相一致

source "$(dirname "$0")/.venv/bin/activate"

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export http_proxy=http://172.31.102.132:7890
export https_proxy=http://172.31.102.132:7890

DEVICE=${1:-cuda:2}
BASE_DIR="$(dirname "$0")"
REF_IMG="$BASE_DIR/PuLID/example_inputs/liuyifei.png"

echo "=== Test 1: 有参考图 ==="
python "$BASE_DIR/generate_illustration.py" \
    --ref "$REF_IMG" \
    --prompt "portrait, anime style, a girl with long hair, cinematic lighting" \
    --output "$BASE_DIR/test_result_1.png" \
    --device "$DEVICE"

echo "=== Test 2: 无参考图 ==="
python "$BASE_DIR/generate_illustration.py" \
    --prompt "a beautiful anime girl in a garden, soft light" \
    --output "$BASE_DIR/test_result_2.png" \
    --device "$DEVICE"

echo "=== Test 3: 同一参考图 + 不同 prompt ==="
python "$BASE_DIR/generate_illustration.py" \
    --ref "$REF_IMG" \
    --prompt "1girl, warrior armor, holding a sword, epic background" \
    --output "$BASE_DIR/test_result_3.png" \
    --device "$DEVICE"

echo "=== All tests done ==="
ls -lh "$BASE_DIR"/test_result_*.png