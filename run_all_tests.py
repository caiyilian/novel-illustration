import os, sys, time, gc, numpy as np
from PIL import Image
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'PuLID')
sys.path.insert(0, 'InstantID')

DEVICE = 'cuda:2'
SEED = 42
OUT = 'test_outputs/report'
os.makedirs(f'{OUT}/noref', exist_ok=True)
os.makedirs(f'{OUT}/withref', exist_ok=True)

# ========== 提示词 ==========
N1 = '1girl, anime style, long blue hair, purple eyes, wearing a white dress, standing in a garden with cherry blossoms, soft sunlight, masterpiece, high quality'
N2 = 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'
N3 = 'a young boy with black hair and red eyes, wearing a school uniform, standing on a rooftop at sunset, looking at the city, anime style, masterpiece'
NOREF_PROMPTS = [('prompt1', N1), ('prompt2', N2), ('prompt3', N3)]

R1 = 'portrait of a cute wolf girl with brown hair and wolf ears, medieval traveler outfit, warm tavern interior, anime style, masterpiece, high quality'
R2 = 'a cute wolf girl with brown hair and wolf ears, wearing a medieval traveler outfit, dancing with a tall dark-haired man in a tavern, holding hands and spinning, warm candlelight, anime style, masterpiece, high quality'
REF_PROMPTS = [('prompt1', R1), ('prompt2', R2)]

NEG = 'flaws in the eyes, flaws in the face, flaws, lowres, non-HDRi, low quality, worst quality, artifacts noise, text, watermark, glitch, deformed, mutated, ugly, disfigured, hands, low resolution, partially rendered objects, deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed, blurry'
ANIME_NEG = 'nsfw, lowres, bad anatomy, worst quality, low quality, blurry'

def timeit(fn):
    start = time.time()
    result = fn()
    return result, time.time() - start

# ========== 1. FLUX.2-klein-4B ==========
print('--- [1/5] FLUX.2-klein-4B ---')
from diffusers import Flux2KleinPipeline
pipe = Flux2KleinPipeline.from_pretrained('models/FLUX.2-klein-4B', torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
for name, p in NOREF_PROMPTS:
    img, t = timeit(lambda: pipe(prompt=p, height=896, width=1152, guidance_scale=1.0, num_inference_steps=4, generator=torch.manual_seed(SEED)).images[0])
    img.save(f'{OUT}/noref/fluxklein_{name}.png')
    print(f'  noref_{name}: {t:.1f}s')
del pipe; gc.collect(); torch.cuda.empty_cache()

# ========== 2. Animagine XL 3.1 ==========
print('--- [2/5] Animagine XL 3.1 ---')
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
pipe = StableDiffusionXLPipeline.from_pretrained('models/animagine-xl-3.1', torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
for name, p in NOREF_PROMPTS:
    img, t = timeit(lambda p=p: pipe(prompt=p, negative_prompt=ANIME_NEG, height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(SEED)).images[0])
    img.save(f'{OUT}/noref/animagine_{name}.png')
    print(f'  noref_{name}: {t:.1f}s')
del pipe; gc.collect(); torch.cuda.empty_cache()

# ========== 3. SDXL base (纯文生图) ==========
print('--- [3/5] SDXL base (no-ref) ---')
pipe = StableDiffusionXLPipeline.from_pretrained('models/sdxl-base-1.0', torch_dtype=torch.float16, variant='fp16').to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
for name, p in NOREF_PROMPTS:
    img, t = timeit(lambda p=p: pipe(prompt=p, negative_prompt=NEG, height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(SEED)).images[0])
    img.save(f'{OUT}/noref/sdxl_{name}.png')
    print(f'  noref_{name}: {t:.1f}s')
del pipe; gc.collect(); torch.cuda.empty_cache()

# ========== 4. PuLID + SDXL base ==========
print('--- [4/5] PuLID v1.1 (SDXL) ---')
import eva_clip.factory as eva_factory
_orig_lsd = eva_factory.load_state_dict
def _patched_lsd(cp, ml='cpu', mk='model|module|state_dict', io=False, sl=[]):
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
eva_factory.load_state_dict = _patched_lsd

from pulid import attention_processor as attention
from pulid.utils import resize_numpy_image_long
import generate_illustration as gi
# Monkey-patch for Animagine (no variant='fp16')
_original_init = gi.PuLIDGenerator.__init__
def _patched_init(self, device='cuda'):
    _original_init(self, device)

gen = gi.PuLIDGenerator(device=DEVICE)
attention.NUM_ZERO = 20; attention.ORTHO = False; attention.ORTHO_v2 = True
ref = np.array(Image.open('ref_images/holo.png').convert('RGB'))
ref = resize_numpy_image_long(ref, 1024)
uncond, id_emb = gen.get_id_embedding([ref])
for name, p in REF_PROMPTS:
    img, t = timeit(lambda p=p: gen.generate(p, (1, 1152, 896), DEFAULT_NEGATIVE_PROMPT, id_emb, uncond, 0.8, 7.0, 25, SEED))
    img.save(f'{OUT}/withref/pulid_{name}.png')
    print(f'  ref_{name}: {t:.1f}s')
del gen; gc.collect(); torch.cuda.empty_cache()

# ========== 5. PuLID + Animagine 基座 ==========
print('--- [5/5] PuLID + Animagine base ---')
import generate_illustration as gi
class AnimaginePuLIDGenerator(PuLIDGenerator):
    def __init__(self, device='cuda'):
        self.device = device
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        self.pipe = gi.StableDiffusionXLPipeline.from_pretrained('models/animagine-xl-3.1', torch_dtype=torch.float16, use_safetensors=True).to(device)
        self.pipe.watermark = None
        self.hack_unet_attn_layers(self.pipe.unet)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.id_adapter = gi.IDFormer().to(device)
        self.face_helper = gi.FaceRestoreHelper(upscale_factor=1, face_size=512, crop_ratio=(1,1), det_model='retinaface_resnet50', save_ext='png', device=device)
        self.face_helper.face_parse = None
        self.face_helper.face_parse = gi.init_parsing_model(model_name='bisenet', device=device)
        model, _, _ = gi.create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual; self.clip_vision_model = model.to(device)
        from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        self.eva_transform_mean = OPENAI_DATASET_MEAN; self.eva_transform_std = OPENAI_DATASET_STD
        from insightface.app import FaceAnalysis; import insightface
        self.app = FaceAnalysis(name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
        self.handler_ante.prepare(ctx_id=0)
        gc.collect(); torch.cuda.empty_cache()
        self._load_pretrain()
        self.debug_img_list = []

gen = AnimaginePuLIDGenerator(device=DEVICE)
attention.NUM_ZERO = 20; attention.ORTHO = False; attention.ORTHO_v2 = True
ref = np.array(Image.open('ref_images/holo.png').convert('RGB'))
ref = resize_numpy_image_long(ref, 1024)
uncond, id_emb = gen.get_id_embedding([ref])
for name, p in REF_PROMPTS:
    img, t = timeit(lambda p=p: gen.generate(p, (1, 1152, 896), DEFAULT_NEGATIVE_PROMPT, id_emb, uncond, 0.8, 7.0, 25, SEED))
    img.save(f'{OUT}/withref/pulid_animagine_{name}.png')
    print(f'  ref_{name}: {t:.1f}s')

# ========== 无参考图 ==========
pipe = StableDiffusionXLPipeline.from_pretrained('models/animagine-xl-3.1', torch_dtype=torch.float16, use_safetensors=True).to(DEVICE)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
for name, p in NOREF_PROMPTS:
    img, t = timeit(lambda p=p: pipe(prompt=p, negative_prompt=ANIME_NEG, height=1152, width=896, num_inference_steps=25, guidance_scale=7.0, generator=torch.manual_seed(SEED)).images[0])
    img.save(f'{OUT}/noref/pulid_animagine_{name}.png')
    print(f'  noref_{name}: {t:.1f}s')

print('\nAll done.')