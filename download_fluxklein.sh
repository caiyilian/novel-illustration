#!/bin/bash
# 一键下载 FLUX.2-klein-4B（用 curl 避免 HEAD 请求被拦截）
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

export http_proxy=http://172.31.102.132:7890
export https_proxy=http://172.31.102.132:7890

BASE="https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/resolve/main"
OUT="models/FLUX.2-klein-4B"

mkdir -p "$OUT"/{scheduler,text_encoder,tokenizer,transformer,vae}

dl() { echo "  ↓ $1"; curl -L -C - -# -o "$OUT/$1" "$BASE/$1" 2>&1 | tail -1; echo "  ✅ $1"; }

echo "=== 下载 FLUX.2-klein-4B ==="
echo ""

echo "--- 配置文件 ---"
dl "model_index.json"
dl "scheduler/scheduler_config.json"

echo "--- tokenizer ---"
dl "tokenizer/tokenizer_config.json"
dl "tokenizer/tokenizer.json"
dl "tokenizer/vocab.json"
dl "tokenizer/merges.txt"
dl "tokenizer/added_tokens.json"
dl "tokenizer/special_tokens_map.json"
dl "tokenizer/chat_template.jinja"

echo "--- text_encoder (Qwen3, ~1.5GB) ---"
dl "text_encoder/config.json"
dl "text_encoder/generation_config.json"
dl "text_encoder/model.safetensors.index.json"
dl "text_encoder/model-00001-of-00002.safetensors"
dl "text_encoder/model-00002-of-00002.safetensors"

echo "--- transformer ---"
dl "transformer/config.json"
dl "transformer/diffusion_pytorch_model.safetensors"

echo "--- VAE ---"
dl "vae/config.json"
dl "vae/diffusion_pytorch_model.safetensors"

echo "--- 主模型 (~8GB, 带断点续传) ---"
dl "flux-2-klein-4b.safetensors"

echo ""
if [ -f "$OUT/flux-2-klein-4b.safetensors" ]; then
    echo "✅ 全部下载完成"
    ls -lh "$OUT/flux-2-klein-4b.safetensors"
else
    echo "❌ 主模型未完成，重新运行脚本可续传"
fi