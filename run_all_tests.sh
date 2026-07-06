#!/bin/bash
# 综合测试脚本：测试 5 个模型 × 所有提示词
# 一个模型测完再换下一个，避免显存爆炸
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

source .venv/bin/activate

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export http_proxy=http://172.31.102.132:7890
export https_proxy=http://172.31.102.132:7890

OUT="$DIR/test_outputs/report"
mkdir -p "$OUT"/{noref,withref}

REF_IMG="$DIR/ref_images/holo.png"
SEED=42
DEVICE="cuda:2"

# ========== 提示词 ==========

# 无参考图（3个）
N1='1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'
N2='a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'
N3='a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'

# 有参考图（2个，holo.png）
R1='portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style, masterpiece, high quality'
R2='a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'

NEG='flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, low resolution, partially rendered objects, deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed, blurry'

echo "==========================================="
echo "  综合测试：5个模型 × 所有提示词"
echo "==========================================="
echo ""

# ========== 1. FLUX.2-klein-4B（纯文生图，无参考图） ==========
echo "--- [1/5] FLUX.2-klein-4B (无参考图) ---"
cat > /tmp/test_flux.py << 'PY'
import torch, os, time
os.environ['HF_HUB_OFFLINE'] = '1'
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained('models/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
prompts = [
    ('prompt1', '1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
    ('prompt3', 'a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'),
]
for name, p in prompts:
    start = time.time()
    img = pipe(prompt=p, height=896, width=1152, guidance_scale=1.0, num_inference_steps=4, generator=torch.manual_seed(42)).images[0]
    img.save(f'test_outputs/report/noref/fluxklein_{name}.png')
    print(f'  {name}: {time.time()-start:.1f}s')
PY
timeout 300 python /tmp/test_flux.py

# ========== 2. Animagine XL 3.1（纯文生图，无参考图） ==========
echo ""
echo "--- [2/5] Animagine XL 3.1 (无参考图) ---"
cat > /tmp/test_animagine.py << 'PY'
import torch, time, os
os.environ['HF_HUB_OFFLINE'] = '1'
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained('models/animagine-xl-3.1', torch_dtype=torch.float16, use_safetensors=True).to('cuda:2')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
prompts = [
    ('prompt1', '1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
    ('prompt3', 'a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'),
]
for name, p in prompts:
    start = time.time()
    img = pipe(prompt=p, negative_prompt='nsfw, lowres, bad anatomy, worst quality, low quality, blurry', height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(42)).images[0]
    img.save(f'test_outputs/report/noref/animagine_{name}.png')
    print(f'  {name}: {time.time()-start:.1f}s')
PY
timeout 300 python /tmp/test_animagine.py

# ========== 3. PuLID v1.1（SDXL base，支持参考图） ==========
echo ""
echo "--- [3/5] PuLID v1.1 (SDXL, 有参考图+无参考图) ---"
cat > /tmp/test_pulid.py << 'PY'
import torch, time, sys, os, numpy as np
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'PuLID')
import eva_clip.factory as eva_factory
def _patch(cp, ml='cpu', mk='model|module|state_dict', io=False, sl=[]):
    ck = torch.load(cp, map_location=ml, weights_only=False)
    for m in mk.split('|'):
        if isinstance(ck, dict) and m in ck: sd = ck[m]; break
    else: sd = ck
    if next(iter(sd.items()))[0].startswith('module'): sd = {k[7:]: v for k,v in sd.items()}
    for k in sl:
        if k in list(sd.keys()): del sd[k]
    if os.getenv('RoPE') == '1':
        for k in list(sd.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k: del sd[k]
    return sd
eva_factory.load_state_dict = _patch

from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long
from generate_illustration import PuLIDGenerator, SDXL_LOCAL_PATH, DEFAULT_NEGATIVE_PROMPT

gen = PuLIDGenerator(sdxl_repo=SDXL_LOCAL_PATH, device='cuda:2')
attention.NUM_ZERO = 20; attention.ORTHO = False; attention.ORTHO_v2 = True

# 有参考图
ref = np.array(Image.open('ref_images/holo.png').convert('RGB'))
ref = resize_numpy_image_long(ref, 1024)
uncond, id_emb = gen.get_id_embedding([ref])

ref_prompts = [
    ('prompt1', 'portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
]
for name, p in ref_prompts:
    start = time.time()
    img = gen.generate(p, (1, 1152, 896), DEFAULT_NEGATIVE_PROMPT, id_emb, uncond, 0.8, 7.0, 25, 42)
    img.save(f'test_outputs/report/withref/pulid_{name}.png')
    print(f'  ref_{name}: {time.time()-start:.1f}s')

# 无参考图
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained(str(SDXL_LOCAL_PATH), torch_dtype=torch.float16, variant='fp16').to('cuda:2')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
noref_prompts = [
    ('prompt1', '1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
    ('prompt3', 'a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'),
]
for name, p in noref_prompts:
    start = time.time()
    img = pipe(prompt=p, negative_prompt=DEFAULT_NEGATIVE_PROMPT, height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(42)).images[0]
    img.save(f'test_outputs/report/noref/pulid_{name}.png')
    print(f'  noref_{name}: {time.time()-start:.1f}s')
PY
timeout 600 python /tmp/test_pulid.py

# ========== 4. InstantID（SDXL base，支持参考图） ==========
echo ""
echo "--- [4/5] InstantID (SDXL, 有参考图) ---"
cat > /tmp/test_instantid.py << 'PY'
import torch, time, sys, os
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'InstantID')
from generate_instantid import InstantIDGenerator
gen = InstantIDGenerator(device='cuda:2')
ref_pil = Image.open('ref_images/holo.png').convert('RGB')
ref_prompts = [
    ('prompt1', 'portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
]
for name, p in ref_prompts:
    start = time.time()
    img = gen.generate(p, '', ref_pil, (1152, 896), 30, 5.0, 0.8, 0.8, 42)
    img.save(f'test_outputs/report/withref/instantid_{name}.png')
    print(f'  ref_{name}: {time.time()-start:.1f}s')
PY
timeout 300 python /tmp/test_instantid.py

# ========== 5. PuLID + Animagine 基座（支持参考图） ==========
echo ""
echo "--- [5/5] PuLID + Animagine 基座 (有参考图+无参考图) ---"
cat > /tmp/test_pulid_animagine.py << 'PY'
import torch, time, sys, os, numpy as np
from PIL import Image
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'PuLID')
import eva_clip.factory as eva_factory
def _patch(cp, ml='cpu', mk='model|module|state_dict', io=False, sl=[]):
    ck = torch.load(cp, map_location=ml, weights_only=False)
    for m in mk.split('|'):
        if isinstance(ck, dict) and m in ck: sd = ck[m]; break
    else: sd = ck
    if next(iter(sd.items()))[0].startswith('module'): sd = {k[7:]: v for k,v in sd.items()}
    for k in sl:
        if k in list(sd.keys()): del sd[k]
    if os.getenv('RoPE') == '1':
        for k in list(sd.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k: del sd[k]
    return sd
eva_factory.load_state_dict = _patch

from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long
from generate_illustration import PuLIDGenerator, DEFAULT_NEGATIVE_PROMPT

gen = PuLIDGenerator(sdxl_repo='models/animagine-xl-3.1', device='cuda:2')
attention.NUM_ZERO = 20; attention.ORTHO = False; attention.ORTHO_v2 = True

# 有参考图
ref = np.array(Image.open('ref_images/holo.png').convert('RGB'))
ref = resize_numpy_image_long(ref, 1024)
uncond, id_emb = gen.get_id_embedding([ref])

ref_prompts = [
    ('prompt1', 'portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
]
for name, p in ref_prompts:
    start = time.time()
    img = gen.generate(p, (1, 1152, 896), DEFAULT_NEGATIVE_PROMPT, id_emb, uncond, 0.8, 7.0, 25, 42)
    img.save(f'test_outputs/report/withref/pulid_animagine_{name}.png')
    print(f'  ref_{name}: {time.time()-start:.1f}s')

# 无参考图
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained('models/animagine-xl-3.1', torch_dtype=torch.float16, use_safetensors=True).to('cuda:2')
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
noref_prompts = [
    ('prompt1', '1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'),
    ('prompt2', 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'),
    ('prompt3', 'a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'),
]
for name, p in noref_prompts:
    start = time.time()
    img = pipe(prompt=p, negative_prompt='nsfw, lowres, bad anatomy, worst quality, low quality, blurry', height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(42)).images[0]
    img.save(f'test_outputs/report/noref/pulid_animagine_{name}.png')
    print(f'  noref_{name}: {time.time()-start:.1f}s')
PY
timeout 600 python /tmp/test_pulid_animagine.py

echo ""
echo "==========================================="
echo "  生成完毕，文件列表："
echo "==========================================="
ls -lh "$OUT/noref/"
echo "---"
ls -lh "$OUT/withref/"

# 生成拼接图
echo ""
echo "--- 生成可视化拼接图 ---"
python -c "
import os
from PIL import Image
import numpy as np

out = 'test_outputs/report'

# 无参考图：5 models × 3 prompts
noref_models = ['pulid', 'animagine', 'pulid_animagine', 'fluxklein']
noref_labels = ['SDXL (base)', 'Animagine', 'PuLID+Animagine', 'FLUX.2-klein']
noref_prompts = ['prompt1', 'prompt2', 'prompt3']
noref_names = ['portrait', 'dance', 'boy_on_rooftop']

# 实际存在的模型
existing = []
existing_labels = []
for m, lbl in zip(noref_models, noref_labels):
    if os.path.exists(f'{out}/noref/{m}_prompt1.png'):
        existing.append(m)
        existing_labels.append(lbl)

if existing:
    img_size = (224, 288)
    rows = len(noref_prompts)
    cols = len(existing)
    canvas = Image.new('RGB', (cols * (img_size[0] + 10) + 10, rows * (img_size[1] + 40) + 40), (255,255,255))
    
    for r, (p, name) in enumerate(zip(noref_prompts, noref_names)):
        for c, (m, lbl) in enumerate(zip(existing, existing_labels)):
            path = f'{out}/noref/{m}_{p}.png'
            if os.path.exists(path):
                img = Image.open(path).resize(img_size, Image.LANCZOS)
                x = 10 + c * (img_size[0] + 10)
                y = 40 + r * (img_size[1] + 40)
                canvas.paste(img, (x, y))
                # 列名（第一行上面）
                if r == 0:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(canvas)
                    draw.text((x + 20, 5), lbl, fill=(0,0,0))
                # 行名（最左边）
                if c == 0:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(canvas)
                    draw.text((5, y + img_size[1]//2 - 5), name, fill=(0,0,0))
    canvas.save(f'{out}/noref_collage.png')
    print(f'  无参考图拼接图: {out}/noref_collage.png ({cols}x{rows})')

# 有参考图：3 models × 2 prompts
withref_models = ['pulid', 'instantid', 'pulid_animagine']
withref_labels = ['PuLID (SDXL)', 'InstantID', 'PuLID+Animagine']
withref_prompts = ['prompt1', 'prompt2']
withref_names = ['portrait', 'dance']

existing_w = []
existing_w_labels = []
for m, lbl in zip(withref_models, withref_labels):
    if os.path.exists(f'{out}/withref/{m}_prompt1.png'):
        existing_w.append(m)
        existing_w_labels.append(lbl)

if existing_w:
    img_size = (224, 288)
    rows = len(withref_prompts)
    cols = len(existing_w)
    canvas = Image.new('RGB', (cols * (img_size[0] + 10) + 10, rows * (img_size[1] + 40) + 40), (255,255,255))
    
    for r, (p, name) in enumerate(zip(withref_prompts, withref_names)):
        for c, (m, lbl) in enumerate(zip(existing_w, existing_w_labels)):
            path = f'{out}/withref/{m}_{p}.png'
            if os.path.exists(path):
                img = Image.open(path).resize(img_size, Image.LANCZOS)
                x = 10 + c * (img_size[0] + 10)
                y = 40 + r * (img_size[1] + 40)
                canvas.paste(img, (x, y))
                if r == 0:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(canvas)
                    draw.text((x + 20, 5), lbl, fill=(0,0,0))
                if c == 0:
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(canvas)
                    draw.text((5, y + img_size[1]//2 - 5), name, fill=(0,0,0))
    canvas.save(f'{out}/withref_collage.png')
    print(f'  有参考图拼接图: {out}/withref_collage.png ({cols}x{rows})')
" 2>&1

echo ""
echo "全部完成，报告目录: $OUT"