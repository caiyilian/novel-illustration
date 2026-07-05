import argparse
import gc
import os
import sys

import numpy as np
import torch
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'PuLID'))

os.environ['HF_HUB_OFFLINE'] = '1'

# Patch eva_clip's torch.load for PyTorch 2.6 compat
import eva_clip.factory as eva_factory
_orig_lsd = eva_factory.load_state_dict
def _patched_lsd(checkpoint_path, map_location='cpu', model_key='model|module|state_dict', is_openai=False, skip_list=[]):
    ck = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    for mk in model_key.split('|'):
        if isinstance(ck, dict) and mk in ck: sd = ck[mk]; break
    else: sd = ck
    if next(iter(sd.items()))[0].startswith('module'): sd = {k[7:]: v for k, v in sd.items()}
    for k in skip_list:
        if k in list(sd.keys()): del sd[k]
    if os.getenv('RoPE') == '1':
        for k in list(sd.keys()):
            if 'freqs_cos' in k or 'freqs_sin' in k: del sd[k]
    return sd
eva_factory.load_state_dict = _patched_lsd

from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long

DEFAULT_NEGATIVE_PROMPT = (
    'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality,'
    'artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, '
    'low resolution, partially rendered objects,  deformed or partially rendered eyes, '
    'deformed, deformed eyeballs, cross-eyed,blurry'
)

SDXL_LOCAL_PATH = os.path.join(BASE_DIR, 'models', 'sdxl-base-1.0')
PULID_CKPT = os.path.join(BASE_DIR, 'models', 'pulid', 'pulid_v1.1.safetensors')
ANTELOPE_DIR = os.path.join(BASE_DIR, 'models', 'antelopev2')


class PuLIDGenerator:
    def __init__(self, device='cuda'):
        self.device = device
        self.debug_img_list = []

        from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline
        from safetensors.torch import load_file
        from pulid.encoders_transformer import IDFormer
        from pulid.utils import is_torch2_available
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        from eva_clip import create_model_and_transforms
        from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        import insightface
        from insightface.app import FaceAnalysis

        if is_torch2_available():
            from pulid.attention_processor import AttnProcessor2_0 as AttnProcessor
            from pulid.attention_processor import IDAttnProcessor2_0 as IDAttnProcessor
        else:
            from pulid.attention_processor import AttnProcessor, IDAttnProcessor

        # Load SDXL
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_LOCAL_PATH, torch_dtype=torch.float16, variant='fp16'
        ).to(self.device)
        self.pipe.watermark = None

        # Hack UNet attention layers for ID embeddings
        id_adapter_attn_procs = {}
        for name, _ in self.pipe.unet.attn_processors.items():
            ca_dim = None if name.endswith('attn1.processor') else self.pipe.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hs = self.pipe.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                hs = list(reversed(self.pipe.unet.config.block_out_channels))[int(name[len('up_blocks.')])]
            elif name.startswith('down_blocks'):
                hs = self.pipe.unet.config.block_out_channels[int(name[len('down_blocks.')])]
            if ca_dim is not None:
                id_adapter_attn_procs[name] = IDAttnProcessor(hidden_size=hs, cross_attention_dim=ca_dim).to(self.device)
            else:
                id_adapter_attn_procs[name] = AttnProcessor()
        self.pipe.unet.set_attn_processor(id_adapter_attn_procs)
        self.id_adapter_attn_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)

        # ID adapter
        self.id_adapter = IDFormer().to(self.device)

        # Face helper
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model='retinaface_resnet50', save_ext='png', device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)

        # EVA-CLIP
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device)
        eva_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_mean, (list, tuple)): eva_mean = (eva_mean,) * 3
        if not isinstance(eva_std, (list, tuple)): eva_std = (eva_std,) * 3
        self.eva_transform_mean = eva_mean
        self.eva_transform_std = eva_std
        self.openai_mean = OPENAI_DATASET_MEAN
        self.openai_std = OPENAI_DATASET_STD

        # InsightFace
        self.app = FaceAnalysis(
            name='antelopev2', root=BASE_DIR,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model(
            os.path.join(ANTELOPE_DIR, 'glintr100.onnx')
        )
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        self._load_pretrain()

    def _load_pretrain(self):
        from safetensors.torch import load_file
        state_dict = load_file(PULID_CKPT)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            state_dict_dict[module][k[len(module) + 1:]] = v
        for module in state_dict_dict:
            print(f'loading {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return x.repeat(1, 3, 1, 1)

    def get_id_embedding(self, image_list):
        import cv2
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import normalize, resize

        id_cond_list = []
        id_vit_hidden_list = []

        for image in image_list:
            self.face_helper.clean_all()
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            face_info = self.app.get(image_bgr)
            if len(face_info) > 0:
                face_info = sorted(face_info, key=lambda x: (x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]
                id_ante_embedding = face_info['embedding']
                self.debug_img_list.append(image[
                    int(face_info['bbox'][1]):int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]):int(face_info['bbox'][2]),
                ])
            else:
                id_ante_embedding = None

            self.face_helper.read_image(image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=True)
            self.face_helper.align_warp_face()
            if len(self.face_helper.cropped_faces) > 0:
                align_face = self.face_helper.cropped_faces[0]
            elif len(face_info) > 0:
                # facexlib 对齐失败，用 insightface 检测框裁剪
                x1, y1, x2, y2 = [int(v) for v in face_info['bbox']]
                # 向外扩 20% 以获得完整面部
                h, w = image.shape[:2]
                margin_x = int((x2 - x1) * 0.2)
                margin_y = int((y2 - y1) * 0.2)
                x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
                x2, y2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
                crop = image[y1:y2, x1:x2]
                crop = cv2.resize(crop, (512, 512))
                align_face = crop[:, :, ::-1]  # RGB to BGR for facexlib compat
            else:
                raise RuntimeError('facexlib align face fail')

            if id_ante_embedding is None:
                print('insightface failed, extracting from align face')
                id_ante_embedding = self.handler_ante.get_feat(align_face)

            id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device)
            if id_ante_embedding.ndim == 1:
                id_ante_embedding = id_ante_embedding.unsqueeze(0)

            input_t = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
            input_t = input_t.to(self.device)
            parsing_out = self.face_helper.face_parse(
                normalize(input_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            )[0]
            parsing_out = parsing_out.argmax(dim=1, keepdim=True)
            bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
            bg = sum(parsing_out == i for i in bg_label).bool()
            white_image = torch.ones_like(input_t)
            face_features_image = torch.where(bg, white_image, self.to_gray(input_t))
            self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

            face_features_image = resize(
                face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC
            )
            face_features_image = normalize(
                face_features_image, self.eva_transform_mean, self.eva_transform_std
            )
            id_cond_vit, id_vit_hidden = self.clip_vision_model(
                face_features_image, return_all_features=False, return_hidden=True, shuffle=False
            )
            id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
            id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

            id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
            id_cond_list.append(id_cond)
            id_vit_hidden_list.append(id_vit_hidden)

        id_uncond = torch.zeros_like(id_cond_list[0])
        id_vit_hidden_uncond = [torch.zeros_like(id_vit_hidden_list[0][i]) for i in range(len(id_vit_hidden_list[0]))]

        id_cond = torch.stack(id_cond_list, dim=1)
        id_vit_hidden = id_vit_hidden_list[0]
        for i in range(1, len(image_list)):
            for j, x in enumerate(id_vit_hidden_list[i]):
                id_vit_hidden[j] = torch.cat([id_vit_hidden[j], x], dim=1)
        id_embedding = self.id_adapter(id_cond, id_vit_hidden)
        uncond_id_embedding = self.id_adapter(id_uncond, id_vit_hidden_uncond)

        return uncond_id_embedding, id_embedding

    def generate(self, prompt, size, neg_prompt='', id_embedding=None, uncond_id_embedding=None,
                 id_scale=0.8, guidance_scale=7.0, steps=25, seed=42):
        cross_attention_kwargs = {'id_embedding': id_embedding, 'id_scale': id_scale}

        img = self.pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            height=size[1],
            width=size[2],
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
            cross_attention_kwargs=cross_attention_kwargs,
        ).images[0]
        return img


def main():
    parser = argparse.ArgumentParser(description='Generate illustration with PuLID v1.1 + SDXL')
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--ref', type=str, default=None)
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--cfg', type=float, default=7.0)
    parser.add_argument('--id_scale', type=float, default=0.8)
    parser.add_argument('--height', type=int, default=1152)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--neg_prompt', type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument('--num_zero', type=int, default=20)
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    seed = args.seed if args.seed != -1 else torch.Generator(device='cpu').seed()

    if args.ref is not None:
        attention.NUM_ZERO = args.num_zero
        attention.ORTHO = False
        attention.ORTHO_v2 = True

        generator = PuLIDGenerator(device=args.device)
        id_image = np.array(Image.open(args.ref).convert('RGB'))
        id_image = resize_numpy_image_long(id_image, 1024)
        uncond_id_embedding, id_embedding = generator.get_id_embedding([id_image])

        img = generator.generate(
            args.prompt, (1, args.height, args.width),
            args.neg_prompt, id_embedding, uncond_id_embedding,
            args.id_scale, args.cfg, args.steps, seed,
        )
    else:
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            SDXL_LOCAL_PATH, torch_dtype=torch.float16, variant='fp16'
        ).to(args.device)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        img = pipeline(
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.manual_seed(seed),
        ).images[0]

    img.save(args.output)
    print(f'Done: {args.output}')


if __name__ == '__main__':
    main()