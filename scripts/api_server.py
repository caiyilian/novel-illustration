"""
Novel Voice Cast — 插图生成 API 服务
支持方法: pulid（默认）, instantid
用法: python api_server.py [--port 8000] [--device cuda:2]
"""

import argparse
import io
import gc
import os
import sys
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
sys.path.insert(0, str(BASE_DIR / 'InstantID'))

os.environ['HF_HUB_OFFLINE'] = '1'

# --- eva_clip patch (for PuLID) ---
import eva_clip.factory as eva_factory
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

pulid = None
device = 'cuda'


@app.on_event('startup')
def load_models():
    global pulid
    print(f'Loading PuLID + SDXL on {device}...')
    pulid = PuLIDGenerator(device=device)
    print('Ready. Methods: pulid, instantid, animagine, pulid_animagine, fluxklein')


@app.post('/generate')
async def generate(
    prompt: str = Form(...),
    ref_image: UploadFile = File(None),
    neg_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    method: str = Form('pulid'),
    seed: int = Form(-1),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    id_scale: float = Form(0.8),
    height: int = Form(1152),
    width: int = Form(896),
    num_zero: int = Form(20),
    ip_scale: float = Form(0.8),
    cn_scale: float = Form(0.8),
):
    seed = seed if seed != -1 else torch.Generator(device='cpu').seed()
    torch.set_grad_enabled(False)

    if method == 'fluxklein':
        from diffusers import Flux2KleinPipeline
        pipe = Flux2KleinPipeline.from_pretrained(
            str(BASE_DIR / 'models' / 'FLUX.2-klein-4B'),
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()
        img = pipe(
            prompt=prompt,
            height=height, width=width,
            guidance_scale=1.0,
            num_inference_steps=min(steps, 4),
            generator=torch.manual_seed(seed),
        ).images[0]
    elif method == 'animagine':
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(BASE_DIR / 'models' / 'animagine-xl-3.1'),
            torch_dtype=torch.float16, use_safetensors=True,
        ).to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        img = pipe(
            prompt=prompt, negative_prompt=neg_prompt,
            height=height, width=width,
            num_inference_steps=steps, guidance_scale=cfg,
            generator=torch.manual_seed(seed),
        ).images[0]
    elif method == 'pulid_animagine':
        model_path = str(BASE_DIR / 'models' / 'animagine-xl-3.1')
        if ref_image is not None:
            from pulid import attention_processor as attention
            from pulid.utils import resize_numpy_image_long
            from generate_illustration import PuLIDGenerator, DEFAULT_NEGATIVE_PROMPT
            from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
            from pulid.encoders_transformer import IDFormer
            from pulid.utils import is_torch2_available
            if is_torch2_available():
                from pulid.attention_processor import AttnProcessor2_0 as AP, IDAttnProcessor2_0 as IDAP
            else:
                from pulid.attention_processor import AttnProcessor as AP, IDAttnProcessor as IDAP
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16, use_safetensors=True
            ).to(device)
            pipe.watermark = None
            procs = {}
            for name, _ in pipe.unet.attn_processors.items():
                ca = None if name.endswith('attn1.processor') else pipe.unet.config.cross_attention_dim
                if name.startswith('mid_block'): hs = pipe.unet.config.block_out_channels[-1]
                elif name.startswith('up_blocks'): hs = list(reversed(pipe.unet.config.block_out_channels))[int(name[len('up_blocks.')])]
                elif name.startswith('down_blocks'): hs = pipe.unet.config.block_out_channels[int(name[len('down_blocks.')])]
                procs[name] = IDAP(hidden_size=hs, cross_attention_dim=ca).to(device) if ca is not None else AP()
            pipe.unet.set_attn_processor(procs)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            gen = PuLIDGenerator(device=device)
            gen.pipe = pipe
            gen.id_adapter_attn_layers = torch.nn.ModuleList(pipe.unet.attn_processors.values())
            attention.NUM_ZERO = num_zero; attention.ORTHO = False; attention.ORTHO_v2 = True
            img_data = await ref_image.read()
            id_image = np.array(Image.open(io.BytesIO(img_data)).convert('RGB'))
            id_image = resize_numpy_image_long(id_image, 1024)
            uncond, id_emb = gen.get_id_embedding([id_image])
            img = gen.generate(prompt, (1, height, width), neg_prompt, id_emb, uncond, id_scale, cfg, steps, seed)
            del gen, pipe; gc.collect(); torch.cuda.empty_cache()
        else:
            from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path, torch_dtype=torch.float16, use_safetensors=True,
            ).to(device)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            img = pipe(
                prompt=prompt, negative_prompt=neg_prompt,
                height=height, width=width,
                num_inference_steps=steps, guidance_scale=cfg,
                generator=torch.manual_seed(seed),
            ).images[0]
    elif ref_image is not None:
        if method == 'instantid':
            from generate_instantid import InstantIDGenerator
            gen = InstantIDGenerator(device=device)
            img_data = await ref_image.read()
            ref_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
            img = gen.generate(
                prompt, neg_prompt, ref_pil,
                (height, width), steps, cfg,
                ip_scale, cn_scale, seed,
            )
            del gen; gc.collect(); torch.cuda.empty_cache()
        else:
            attention.NUM_ZERO = num_zero
            attention.ORTHO = False
            attention.ORTHO_v2 = True
            img_data = await ref_image.read()
            id_image = np.array(Image.open(io.BytesIO(img_data)).convert('RGB'))
            id_image = resize_numpy_image_long(id_image, 1024)
            uncond_id_embedding, id_embedding = pulid.get_id_embedding([id_image])
            img = pulid.generate(
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


@app.post('/generate_video')
async def generate_video(
    prompt: str = Form(...),
    task: str = Form('t2v'),
    seed: int = Form(-1),
    steps: int = Form(25),
    cfg: float = Form(6.0),
    fps: int = Form(8),
):
    seed = seed if seed != -1 else torch.Generator(device='cpu').seed()
    torch.set_grad_enabled(False)

    from diffusers import CogVideoXPipeline
    from diffusers.utils import export_to_video
    import tempfile

    pipe = CogVideoXPipeline.from_pretrained(
        str(BASE_DIR / 'models' / 'CogVideoX-2b'),
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=torch.manual_seed(seed),
    ).frames[0]

    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    export_to_video(video, tmp.name, fps=fps)

    with open(tmp.name, 'rb') as f:
        content = f.read()
    os.unlink(tmp.name)

    return Response(content=content, media_type='video/mp4')


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