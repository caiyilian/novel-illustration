# 插图生成 — 后续方案部署 TODO

现状：✅ **SDXL + PuLID v1.1** 已部署运行（峰值 ~14.7GB，~3-4s/张）

以下方案按优先级排列，均可在 RTX 4090 24GB 上运行。

---

## ✅ 1. InstantID (SDXL)

| 项目 | 内容 |
|------|------|
| 基座 | SDXL |
| 估测显存 | ~16GB |
| 官方仓库 | https://github.com/instantX-research/InstantID |
| 状态 | ✅ **已部署** |
| 脚本 | `scripts/generate_instantid.py` |
| 模型位置 | `InstantID/checkpoints/` |
| 峰值显存 | 15404 MiB |
| 速度 | ~5.73 it/s（30步） |
| 实测结论 | ❌ **脸崩，质量不如 PuLID v1.1**。InstantID 基于 ControlNet + IP-Adapter，人脸关键点约束过强，导致脸部变形/失真。PuLID 的 attention injection 方式对原始模型行为干扰更小，身份保持更自然。 |

---

## ☐ 2. IP-Adapter-FaceID (SDXL)

| 项目 | 内容 |
|------|------|
| 基座 | SDXL |
| 估测显存 | ~10GB |
| 官方仓库 | https://huggingface.co/h94/IP-Adapter-FaceID |
| 状态 | ⏳ 待部署 |

---

## ☐ 3. FLUX.2 klein (4B)

| 项目 | 内容 |
|------|------|
| 基座 | FLUX.2 |
| 估测显存 | ~13GB |
| 官方仓库 | Black Forest Labs (待确认) |
| 状态 | ⏳ 待部署 |

---

## ☐ 4. SDXL + 动漫微调模型

| 项目 | 内容 |
|------|------|
| 基座 | SDXL |
| 估测显存 | ~8-10GB |
| 可选模型 | Illustrious / RouWei-0.6 / Anima |
| 状态 | ⏳ 待部署 |

---

## 显存对比

```
PuLID v1.1 (SDXL)    ████████████████████░░  ~14.7GB  ✅ 已部署
InstantID (SDXL)      ████████████████████░░  ~16GB    📝 待试
IP-Adapter-FaceID     ██████████████░░░░░░░░  ~10GB    📝 待试
FLUX.2 klein          ██████████████████░░░░  ~13GB    📝 待试
SDXL+动漫微调         ████████████░░░░░░░░░░  ~8-10GB  📝 待试
总容量                ████████████████████████  24GB
```