import argparse
import gc
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'InstantID'))

os.environ['HF_HUB_OFFLINE'] = '1'

from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLPipeline
from diffusers.models import ControlNetModel
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

SDXL_LOCAL_PATH = os.path.join(BASE_DIR, 'models', 'sdxl-base-1.0')
CONTROLNET_PATH = os.path.join(BASE_DIR, 'InstantID', 'checkpoints', 'ControlNetModel')
IP_ADAPTER_PATH = os.path.join(BASE_DIR, 'InstantID', 'checkpoints', 'ip-adapter.bin')
ANTELOPE_PATH = os.path.join(BASE_DIR, 'models', 'antelopev2')

DEFAULT_NEGATIVE_PROMPT = (
    '(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, '
    'illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured'
)


class InstantIDGenerator:
    def __init__(self, device='cuda'):
        self.device = device

        self.app = FaceAnalysis(
            name='antelopev2', root=str(BASE_DIR),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        controlnet = ControlNetModel.from_pretrained(
            CONTROLNET_PATH, torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            SDXL_LOCAL_PATH,
            controlnet=controlnet,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.pipe.load_ip_adapter_instantid(IP_ADAPTER_PATH)
        self.pipe.watermark = None

        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, prompt, neg_prompt, ref_image_pil, size=(1152, 896),
                 steps=30, guidance_scale=5.0, ip_adapter_scale=0.8,
                 controlnet_conditioning_scale=0.8, seed=42):

        face_info = self.app.get(cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            raise RuntimeError('No face detected in reference image')
        face_info = sorted(
            face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1])
        )[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(ref_image_pil, face_info['kps'])

        img = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image_embeds=face_emb,
            image=face_kps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            ip_adapter_scale=ip_adapter_scale,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=size[0],
            width=size[1],
            generator=torch.manual_seed(seed),
        ).images[0]
        return img


def main():
    parser = argparse.ArgumentParser(description='Generate with InstantID')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg', type=float, default=5.0)
    parser.add_argument('--ip_scale', type=float, default=0.8)
    parser.add_argument('--cn_scale', type=float, default=0.8)
    parser.add_argument('--height', type=int, default=1152)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--neg_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    seed = args.seed if args.seed != -1 else torch.Generator(device='cpu').seed()

    if args.ref is not None:
        gen = InstantIDGenerator(device=args.device)
        ref_image_pil = Image.open(args.ref).convert('RGB')
        img = gen.generate(
            args.prompt, args.neg_prompt, ref_image_pil,
            (args.height, args.width), args.steps, args.cfg,
            args.ip_scale, args.cn_scale, seed,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_LOCAL_PATH, torch_dtype=torch.float16, variant='fp16'
        ).to(args.device)
        img = pipe(
            prompt=args.prompt, negative_prompt=args.neg_prompt,
            height=args.height, width=args.width,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            generator=torch.manual_seed(seed),
        ).images[0]

    img.save(args.output)
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()