# Novel Voice Cast — Illustration Generator

为小说生成动漫风格插图的零样本身份保持生成工具。

基于 **SDXL base 1.0** + **PuLID v1.1**，提供 HTTP API 供调用。

## 功能

- **有参考图模式**：上传角色参考图，生成新场景时保持角色长相
- **无参考图模式**：纯文本 prompt 生成
- **API 服务**：HTTP 接口，方便集成到其他项目

## 硬件要求

- **GPU**: NVIDIA RTX 4090 (24GB) 或同等及以上
- **存储**: 至少 30GB 可用空间
- **系统**: Linux

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境（Python 3.10）
uv venv -p 3.10 .venv
source .venv/bin/activate

# 安装 PyTorch 2.6 (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 安装依赖
pip install diffusers transformers accelerate safetensors \
    opencv-python insightface onnxruntime-gpu gradio \
    basicsr==1.3.1 facexlib fastapi uvicorn
```

### 2. 下载模型权重

将以下权重下载后放入 `models/` 目录：

| 权重 | 下载源 | 存放位置 |
|------|--------|---------|
| SDXL base 1.0 | HuggingFace `stabilityai/stable-diffusion-xl-base-1.0` 或 ModelScope | `models/sdxl-base-1.0/` |
| PuLID v1.1 | HuggingFace `guozinan/PuLID` → `pulid_v1.1.safetensors` | `models/pulid/pulid_v1.1.safetensors` |
| InsightFace antelopev2 | HuggingFace `DIAMONIK7777/antelopev2` | `models/antelopev2/` |

### 3. 克隆 PuLID 仓库

```bash
git clone https://github.com/ToTheBeginning/PuLID.git
```

### 4. 生成图片

```bash
# 有参考图
bash scripts/with_ref.sh 参考图.png "your prompt" output.png cuda:0

# 无参考图
bash scripts/no_ref.sh "your prompt" output.png cuda:0
```

### 5. 启动 API 服务

```bash
nohup bash scripts/start_api.sh 8000 cuda:0 > api.log 2>&1 &
```

## API 文档

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/generate` | POST | 生成图片 |

详细参数见 `docs/插图生成API接入文档.md`。

## 项目结构

```
├── scripts/           # Python 脚本和 shell 脚本
│   ├── generate_illustration.py  # 主生成脚本
│   ├── api_server.py             # API 服务
│   ├── with_ref.sh               # 有参考图启动
│   ├── no_ref.sh                 # 无参考图启动
│   └── start_api.sh              # API 服务启动
├── docs/              # 文档
├── models/            # 模型权重（需自行下载）
├── PuLID/             # PuLID 仓库
└── .venv/             # Python 环境
```

## 注意事项

- **首次运行需要联网**以下载 EVA02-CLIP 和 facexlib 模型权重，缓存后即可离线
- **角色一致性有限**：SDXL + PuLID v1.1 能保持大体特征（发色、服色），但五官细节可能有变化
- **不支持并发请求**：单 GPU 同时只能处理一张图