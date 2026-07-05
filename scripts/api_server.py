"""
Novel Voice Cast — 插图生成 API 服务
用法: python api_server.py [--port 8000] [--device cuda:2]
"""

import argparse
import io
import gc
import os
import sys
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'PuLID'))

os.environ['HF_HUB_OFFLINE'] = '1'

import eva_clip.factory as eva_factory
_orig_lsd = eva_factory.load_state_dict
def _patched_lsd(cp, ml='cpu', mk='model|module|state_dict', io=False, sl=[]):
    ck = torch.load(cp, map_location=ml, weights_only=False)
    for m in mk.split('|'):
        if isinstance(ck, dict) and m in ck: sd = ck[m]; break
    else: sd = ck
    if next(iter(sd.items()))[0].startswith('module'): sd = {k[7:]: v for k, v in sd.items()}
    for k in sl:
        if k in list(sd.keys()): del sd[k]
    if os.getenv('RoPE') == '1':
        for k in list(sd.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k: del sd[k]
    return sd
eva_factory.load_state_dict = _patched_lsd

from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long
from generate_illustration import PuLIDGenerator, SDXL_LOCAL_PATH, DEFAULT_NEGATIVE_PROMPT

app = FastAPI(title='Novel Voice Cast - Illustration Generator')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

generator = None
device = 'cuda'


@app.on_event('startup')
def load_models():
    global generator
    print(f'Loading PuLID + SDXL on {device}...')
    generator = PuLIDGenerator(device=device)
    print('Ready.')


@app.post('/generate')
async def generate(
    prompt: str = Form(...),
    ref_image: UploadFile = File(None),
    neg_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    seed: int = Form(-1),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    id_scale: float = Form(0.8),
    height: int = Form(1152),
    width: int = Form(896),
    num_zero: int = Form(20),
):
    seed = seed if seed != -1 else torch.Generator(device='cpu').seed()
    torch.set_grad_enabled(False)

    if ref_image is not None:
        attention.NUM_ZERO = num_zero
        attention.ORTHO = False
        attention.ORTHO_v2 = True

        img_data = await ref_image.read()
        id_image = np.array(Image.open(io.BytesIO(img_data)).convert('RGB'))
        id_image = resize_numpy_image_long(id_image, 1024)
        uncond_id_embedding, id_embedding = generator.get_id_embedding([id_image])

        img = generator.generate(
            prompt, (1, height, width), neg_prompt,
            id_embedding, uncond_id_embedding,
            id_scale, cfg, steps, seed,
        )
    else:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(SDXL_LOCAL_PATH), torch_dtype=torch.float16, variant='fp16'
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        img = pipe(
            prompt=prompt, negative_prompt=neg_prompt,
            height=height, width=width,
            num_inference_steps=steps, guidance_scale=cfg,
            generator=torch.manual_seed(seed),
        ).images[0]

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return Response(content=buf.getvalue(), media_type='image/png')


@app.get('/health')
def health():
    return {'status': 'ok', 'device': device}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()
    device = args.device
    uvicorn.run(app, host=args.host, port=args.port)