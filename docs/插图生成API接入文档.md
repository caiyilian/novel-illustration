# Novel Voice Cast — 插图生成 API 接入文档

## 概述

Linux 服务器（RTX 4090 24GB）提供一个 HTTP API，用于为小说场景生成动漫风格插图。

| 功能 | 说明 |
|------|------|
| 有参考图 | 上传角色参考图 → 保持长相生成新场景图 |
| 无参考图 | 纯文本 prompt 生成 |
| 基座模型 | SDXL base 1.0（2.6B 参数） |
| 身份保持 | PuLID v1.1（zero-shot，不训练） |

---

## 启动服务

```bash
cd /sda/Public/wyl/novel-illustration
nohup bash start_api.sh <端口号> <设备> > api.log 2>&1 &

# 示例（后台启动，端口 8001，GPU 2号卡）
nohup bash start_api.sh 8001 cuda:2 > api.log 2>&1 &
```

等待约 30 秒加载模型，看到 `Application startup complete` 即就绪。

---

## 停止服务

```bash
# 查 PID
ps aux | grep api_server

# 杀进程
kill <PID>
```

---

## API 接口

### 1. 健康检查

```
GET /health
```

返回：`{"status": "ok", "device": "cuda:2"}`

### 2. 生成图片

```
POST /generate
Content-Type: multipart/form-data
```

**参数：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `prompt` | string | 是 | — | 场景描述 |
| `ref_image` | file | 否 | — | 角色参考图（PNG/JPG），不传则无参考图模式 |
| `method` | string | 否 | `pulid` | 生成方法：`pulid`（身份保持，推荐）、`instantid`（身份保持，脸易崩）、`fluxklein`（纯文生图，无身份保持，质量好） |
| `neg_prompt` | string | 否 | 见下方 | 负面提示词 |
| `seed` | int | 否 | -1（随机） | 随机种子 |
| `steps` | int | 否 | 25(PuLID)/30(InstantID)/4(fluxklein) | 推理步数 |
| `cfg` | float | 否 | 7.0(PuLID)/5.0(InstantID)/1.0(fluxklein) | CFG 强度 |
| `id_scale` | float | 否 | 0.8 | PuLID 身份保持强度（0.5-1.5） |
| `num_zero` | int | 否 | 20 | PuLID 身份可编辑性（10-30） |
| `ip_scale` | float | 否 | 0.8 | InstantID IP-Adapter 强度 |
| `cn_scale` | float | 否 | 0.8 | InstantID ControlNet 强度 |
| `height` | int | 否 | 1152 | 图片高度 |
| `width` | int | 否 | 896 | 图片宽度 |

**生成方法说明：**

| method | 能力 | 支持参考图 |
|--------|------|-----------|
| `pulid`（默认） | 身份保持 ✅ 质量最好 | ✅ 支持 |
| `instantid` | 身份保持 ⚠️ 脸易崩 | ✅ 支持 |
| `fluxklein` | 纯文生图 ✅ 画质好 | ❌ 不支持 |

**返回：** PNG 图片二进制（直接写入文件即可）

**默认负面提示词：**
```
flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,
artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands,
low resolution, partially rendered objects, deformed or partially rendered eyes,
deformed, deformed eyeballs, cross-eyed, blurry
```

---

## Windows 端调用示例

### 安装依赖

```bash
pip install requests Pillow
```

### 无参考图模式

```python
import requests

BASE = 'http://<服务器IP>:8001'

resp = requests.post(f'{BASE}/generate', data={
    'prompt': '1girl, anime style, long blue hair, purple eyes, wearing a white dress, '
              'standing in a garden with cherry blossoms, soft sunlight, masterpiece',
    'steps': 20,
    'cfg': 7.0,
})

with open('output.png', 'wb') as f:
    f.write(resp.content)
print('Generated:', 'output.png')
```

### 有参考图模式（PuLID，默认）

```python
import requests

BASE = 'http://<服务器IP>:8001'

with open('holo.png', 'rb') as f:
    resp = requests.post(f'{BASE}/generate', data={
        'prompt': 'portrait of a cute wolf girl with brown hair and wolf ears, '
                  'medieval traveler outfit, smiling, warm tavern interior, '
                  'anime style, masterpiece',
        'method': 'pulid',  # puLID（推荐，质量更好）
        'id_scale': 0.8,
        'num_zero': 20,
        'steps': 25,
    }, files={
        'ref_image': f,
    })

with open('tavern_scene.png', 'wb') as f:
    f.write(resp.content)
print('Generated:', 'tavern_scene.png')
```

### 有参考图模式（InstantID，备选）

```python
resp = requests.post(f'{BASE}/generate', data={
    'prompt': 'portrait of a cute wolf girl, tavern',
    'method': 'instantid',  # InstantID（人脸易崩，不推荐）
    'ip_scale': 0.8,
    'cn_scale': 0.8,
    'steps': 30,
}, files={
    'ref_image': open('holo.png', 'rb'),
})
```

### 批量生成（多场景、同一角色）

```python
import requests
from pathlib import Path

BASE = 'http://<服务器IP>:8001'

scenes = [
    'walking through a wheat field at sunset',
    'sitting by a campfire at night, starry sky',
    'standing on a medieval village market street',
]

with open('holo.png', 'rb') as ref_f:
    ref_data = ref_f.read()

for i, scene in enumerate(scenes):
    prompt = f'portrait of a cute wolf girl with brown hair and wolf ears, {scene}, anime style, masterpiece'
    resp = requests.post(f'{BASE}/generate', data={
        'prompt': prompt,
        'id_scale': 0.8,
        'num_zero': 20,
        'seed': 42 + i,  # 每张不同种子
    }, files={
        'ref_image': ('holo.png', ref_data, 'image/png'),
    })
    Path(f'scene_{i+1}.png').write_bytes(resp.content)
    print(f'Scene {i+1} done')
```

---

## prompt 写作建议

SDXL 对提示词风格敏感，以下原则可提升效果：

| 原则 | 说明 | 示例 |
|------|------|------|
| 正面描述 | 描述画面内容，不要写否定 | ✅ `sunlight` ❌ `no darkness` |
| 质量词 | 加 `masterpiece`, `high quality`, `best quality` | 放在 prompt 末尾 |
| 风格词 | 指定风格 | `anime style`, `illustration`, `digital painting` |
| 角色描述 | 发色、发型、服装、表情 | `brown hair, wolf ears, medieval traveler outfit, smiling` |
| 场景描述 | 环境、光线、氛围 | `warm tavern interior, fireplace, wooden tables` |
| 负面词 | 用 `neg_prompt` 参数传，别写在 prompt 里 | 使用默认值即可 |

**常用风格词：** `anime style`, `illustration`, `digital painting`, `artstation`, `studio ghibli`, `makoto shinkai`

不建议加 `1girl`, `solo` 等 Danbooru 标签（SDXL 原生理解这些，但 anime 标签可能让风格更偏写实）。

---

## 限制

- **角色一致性有限：** PuLID v1.1 on SDXL 能保持大体特征（发色、服色、体态），但五官细节（眼型、脸型）可能有变化。要提高一致性可调大 `id_scale`（如 1.5）和调小 `num_zero`（如 10）
- **动漫参考图识别较弱：** InsightFace 的人脸检测模型训练于真实人脸，对动漫/插画风格参考图可能检测失败。代码已加回落机制（用检测框直接裁剪），但效果不如真实照片
- **每张图生成时间：** 25 步约 3-4 秒（不含模型加载）
- **不支持并发请求：** 一个 GPU 同时只能处理一张图

---

## 模型与文件位置

| 组件 | 位置 |
|------|------|
| SDXL base 1.0 | `models/sdxl-base-1.0/` |
| PuLID v1.1 | `models/pulid/pulid_v1.1.safetensors` |
| InsightFace antelopev2 | `models/antelopev2/` |
| API 服务脚本 | `api_server.py` |
| 启动脚本 | `start_api.sh` |
| 主生成脚本 | `generate_illustration.py` |
| Python 环境 | `.venv/` |

首次启动需要联网（走 HTTP 代理）以下载 EVA02-CLIP 和 facexlib 模型权重，缓存后即可离线运行。