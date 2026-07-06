import argparse
import gc
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'IP-Adapter'))

os.environ['HF_HUB_OFFLINE'] = '1'

from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

SDXL_LOCAL_PATH = os.path.join(BASE_DIR, 'models', 'sdxl-base-1.0')
IP_CKPT = os.path.join(BASE_DIR, 'IP-Adapter', 'models', 'ip-adapter-faceid_sdxl.bin')
LORA_CKPT = os.path.join(BASE_DIR, 'IP-Adapter', 'models', 'ip-adapter-faceid_sdxl_lora.safetensors')
ANTELOPE_DIR = os.path.join(BASE_DIR, 'models', 'antelopev2')

DEFAULT_NEGATIVE_PROMPT = (
    'monochrome, lowres, bad anatomy, worst quality, low quality, blurry'
)


class IPAdapterFaceIDGenerator:
    def __init__(self, device='cuda'):
        self.device = device

        self.app = FaceAnalysis(
            name='antelopev2', root=str(BASE_DIR),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
            beta_schedule='scaled_linear', clip_sample=False,
            set_alpha_to_one=False, steps_offset=1,
        )

        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_LOCAL_PATH, torch_dtype=torch.float16,
            scheduler=noise_scheduler, add_watermarker=False,
        )

        pipe.load_lora_weights(LORA_CKPT, weight_name=os.path.basename(LORA_CKPT))
        pipe.fuse_lora()

        self.ip_model = IPAdapterFaceIDXL(pipe, IP_CKPT, device, num_tokens=4)

        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, prompt, neg_prompt, ref_image_pil, scale=1.0,
                 steps=30, guidance_scale=7.5, seed=42, height=1152, width=896):

        face_info = self.app.get(cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            raise RuntimeError('No face detected in reference image')
        face_info = sorted(
            face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1])
        )[-1]
        faceid_embeds = torch.from_numpy(face_info['embedding']).unsqueeze(0).to(self.device)

        images = self.ip_model.generate(
            prompt=prompt,
            negative_prompt=neg_prompt,
            faceid_embeds=faceid_embeds,
            scale=scale,
            num_samples=1,
            seed=seed,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
        )
        return images[0]


def main():
    parser = argparse.ArgumentParser(description='Generate with IP-Adapter-FaceID')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg', type=float, default=7.5)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--height', type=int, default=1152)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--neg_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    seed = args.seed if args.seed != -1 else torch.Generator(device='cpu').seed()

    if args.ref is not None:
        gen = IPAdapterFaceIDGenerator(device=args.device)
        ref_pil = Image.open(args.ref).convert('RGB')
        img = gen.generate(
            args.prompt, args.neg_prompt, ref_pil,
            args.scale, args.steps, args.cfg, seed,
            args.height, args.width,
        )
    else:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_LOCAL_PATH, torch_dtype=torch.float16, variant='fp16'
        ).to(args.device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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