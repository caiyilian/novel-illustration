# CogVideoX 视频生成 — 部署调研

## 概述

CogVideoX 是智谱 AI 开源的视频生成模型，支持文生视频（T2V）、图生视频（I2V）、视频续写（V2V）。

> 非商用场景下三个版本均可使用。

---

## 模型对比

| 模型 | 参数量 | 最低显存 | 推荐显存 | 许可 | 视频长度 | 最大分辨率 | 支持任务 |
|------|--------|---------|---------|------|---------|-----------|---------|
| CogVideoX-2B | 2B | 3.6GB (INT8) | 4GB (FP16) | **Apache 2.0** | 6秒 | 720×480 | T2V / V2V |
| CogVideoX-5B | 5B | 5GB (BF16) | 10GB | 非商用 | 6秒 | 720×480 | T2V / I2V / V2V |
| CogVideoX1.5-5B | 5B | 10GB (BF16) | 15GB | 非商用 | 10秒 | 1360×768 | T2V / I2V / V2V |

## 硬件适配

| GPU | 2B (FP16) | 2B (INT8) | 5B (BF16) | 5B (INT8) | 1.5-5B (BF16) |
|-----|-----------|-----------|-----------|-----------|--------------|
| RTX 4090 24GB | ✅ 实测~20.3GB | ✅ | ✅ | ✅ | ✅ |
| RTX 3060 12GB | ❌ 实测~20.3GB | ✅ | ❌ | ✅ | ❌ |

## 实测数据（CogVideoX-2B，RTX 4090 24GB，FP16）

| 任务 | 峰值显存 | 速度 | 帧数 | 分辨率 |
|------|---------|------|------|--------|
| **T2V**（文生视频） | **20752 MiB** (~20.3GB) | 1.68s/it（25步≈49s） | 49帧 | 720×480 |
| **V2V**（视频续写） | **20712 MiB** (~20.2GB) | 1.68s/it（15步≈35s） | 49帧 | 720×480 |
| **I2V**（图生视频） | — | — | — | — |

## 注意事项

- **2B 版不支持 I2V**，需要 5B 版
- 2B 版在 FP16 下实际峰值约 **20.3GB**，接近 24GB 上限，3060 12GB 无法运行
- 使用 INT8 量化可降至 ~3.6GB，但需安装 torchao

## 依赖

- Python 3.10 - 3.12
- diffusers + transformers + accelerate + torch
- 可选：torchao（INT8 量化）

## 下载方式

| 模型 | HuggingFace | ModelScope |
|------|------------|------------|
| CogVideoX-2B | THUDM/CogVideoX-2b | ZhipuAI/CogVideoX-2b |
| CogVideoX-5B | THUDM/CogVideoX-5b | ZhipuAI/CogVideoX-5b |
| CogVideoX1.5-5B | THUDM/CogVideoX1.5-5B | ZhipuAI/CogVideoX1.5-5B |

## 限制

- 仅支持英文 prompt（需用 LLM 翻译优化）
- 最长 10 秒视频
- 5B 模型非商用许可（对我们无影响）
- 推理速度较慢（A100 上 ~90 秒/段）