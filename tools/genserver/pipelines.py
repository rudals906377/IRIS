"""생성 모드의 그리기: Stable Diffusion 인페인팅 + IP-Adapter(참고 사진 따라 그리기).

- 마스크 안만 새로 그리고, 나머지는 원본 픽셀을 그대로 둔다(얼굴·배경 보존).
- 참고 사진이 있으면 IP-Adapter로 그 스타일(색·질감·형태)을 따르고, 글 설명(prompt)으로 보강한다.
- 모델은 처음 한 번 Hugging Face에서 내려받는다(약 2GB + IP-Adapter 1.7GB).
- GPU가 없으면 CPU로도 돌지만 한 장에 몇 분이 걸린다(시험용).
- --dry-run 이면 모델 없이 마스크를 색으로 표시한 그림을 돌려준다(연결 시험용).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from typing import Callable

import numpy as np
from PIL import Image

from masks import feather

# 모델 묶음: 이름 → (인페인팅 모델, IP-Adapter 폴더, 가중치, 이미지 인코더 폴더, 기본 생성 크기)
#  sd15: 가볍고 빠름(GPU 4GB~). sdxl: 더 크고 세밀함(GPU 10GB~, 메모리가 모자라면 자동으로 일부를 CPU에 둔다)
PRESETS = {
    "sd15": ("stable-diffusion-v1-5/stable-diffusion-inpainting", "models", "ip-adapter_sd15.bin", "image_encoder", 768),
    "sdxl": ("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.safetensors", "models/image_encoder", 1024),
}
SDXL_VAE = "madebyollin/sdxl-vae-fp16-fix"  # fp16에서 검은 그림이 나오는 문제를 고친 VAE
IP_REPO = "h94/IP-Adapter"
MODEL_ENV = os.environ.get("IRIS_GEN_MODEL", "auto")  # auto | sd15 | sdxl | (Hugging Face 모델 id)
MAX_SIDE_ENV = os.environ.get("IRIS_GEN_MAX_SIDE")
MODEL_ID = PRESETS["sd15"][0]  # 실제 값은 load()에서 정해진다(health 표시용 초기값)


def pick_preset(name: str, log=print) -> tuple[str, tuple]:
    """모델 이름(auto/sd15/sdxl/HF id) → (preset 이름, preset 값). auto는 GPU 메모리로 정한다."""
    if name == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                mem = torch.cuda.get_device_properties(0).total_memory / 2**30
                name = "sdxl" if mem >= 9.5 else "sd15"
                log(f"GPU 메모리 {mem:.1f}GB → 모델 {name}")
            else:
                name = "sd15"
        except Exception:  # noqa: BLE001
            name = "sd15"
    if name in PRESETS:
        return name, PRESETS[name]
    # 직접 지정한 HF id: 이름에 xl 이 있으면 SDXL 계열로 본다
    base = PRESETS["sdxl" if "xl" in name.lower() else "sd15"]
    return name, (name, *base[1:])

# 분야별 글 설명 틀. {desc}에는 스타일 AI가 읽은 속성(영어)이나 사용자가 적은 설명이 들어간다
PROMPTS = {
    "hair": (
        "photo of the same person with {desc}, natural realistic hair, consistent lighting, high detail, sharp focus",
        "bald, deformed face, different person, blurry, lowres, extra heads, text, watermark, cartoon",
    ),
    "nail": (
        "close-up photo of a hand with {desc} manicure, realistic glossy gel nails, clean cuticles, high detail",
        "deformed fingers, extra fingers, missing fingers, blurry, lowres, text, watermark",
    ),
    "tattoo": (
        "realistic {desc} tattoo on skin, crisp ink lines, natural skin texture, photo",
        "blurry, smudged, text, watermark, extra limbs, deformed, cartoon",
    ),
}
# 분야별 기본 세기: 웹이 보내는 그림에는 실시간 합성(타투 도안·네일 색)이 이미 올라가 있으므로
# 타투·네일은 그것을 살리면서 피부에 녹이는 정도만(낮은 strength), 헤어는 모양을 새로 그린다(높은 strength)
DEFAULT_STRENGTH = {"hair": 0.95, "nail": 0.65, "tattoo": 0.45}
DEFAULT_GUIDANCE = {"hair": 7.0, "nail": 7.5, "tattoo": 8.0}
DEFAULT_DESC = {"hair": "the hairstyle from the reference photo", "nail": "the nail art from the reference photo", "tattoo": "the tattoo from the reference photo"}


@dataclass
class GenRequest:
    category: str
    image: Image.Image
    mask: np.ndarray  # 0~255 uint8, 이미지 크기
    reference: Image.Image | None = None
    desc: str | None = None
    negative: str | None = None
    steps: int = 30
    strength: float | None = None
    guidance: float | None = None
    ip_scale: float = 0.6
    seed: int | None = None
    # 헤어: 생성 뒤 머리 영역의 색을 참고 사진 머리색에 맞춘다(모델이 색을 틀리거나 투톤으로 그리는 것을 막음)
    color_lock: bool = True
    # 색 맞추기에 쓸 함수(결과, 마스크, 참고 사진, 로그) → 결과. masks.Masker 를 아는 server 가 넣어 준다
    # (타입 표기가 없으면 dataclass 필드로 안 잡혀 생성 요청이 통째로 실패한다)
    color_fn: Callable | None = None


class Generator:
    def __init__(self, dry_run: bool = False, device: str | None = None, model_name: str = MODEL_ENV) -> None:
        self.dry_run = dry_run
        self.pipe = None
        self.device = device
        self.loaded_model: str | None = None
        self.model_name = model_name
        self.max_side = int(MAX_SIDE_ENV) if MAX_SIDE_ENV else 768
        self.is_xl = False

    def load(self, log=print) -> None:
        if self.dry_run or self.pipe is not None:
            return
        global MODEL_ID
        import torch

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        dtype = torch.float16 if dev == "cuda" else torch.float32
        preset_name, (model_id, ip_sub, ip_weight, ip_enc, max_side) = pick_preset(self.model_name, log)
        self.max_side = int(MAX_SIDE_ENV) if MAX_SIDE_ENV else max_side
        self.is_xl = preset_name == "sdxl" or "xl" in model_id.lower()
        log(f"모델 불러오는 중: {model_id} ({dev}, 생성 크기 {self.max_side})")
        if self.is_xl:
            from diffusers import AutoencoderKL, StableDiffusionXLInpaintPipeline

            kw = dict(torch_dtype=dtype)
            if dev == "cuda":
                kw["variant"] = "fp16"
                kw["vae"] = AutoencoderKL.from_pretrained(SDXL_VAE, torch_dtype=dtype)
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model_id, **kw)
        else:
            from diffusers import StableDiffusionInpaintPipeline

            pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=dtype, safety_checker=None)
        log("IP-Adapter 불러오는 중")
        pipe.load_ip_adapter(IP_REPO, subfolder=ip_sub, weight_name=ip_weight, image_encoder_folder=ip_enc)
        # 주의: enable_attention_slicing / xformers 는 IP-Adapter 가 바꿔 둔 어텐션 처리기를 덮어써서
        # "'tuple' object has no attribute 'shape'" 로 생성이 실패한다(RTX 3080에서 확인). 켜지 않는다.
        offload = False
        if dev == "cuda" and self.is_xl:
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            offload = mem < 12.5  # SDXL + IP-Adapter Plus 는 12GB 이하에서 다 올리면 모자란다
        if offload:
            log("GPU 메모리가 넉넉하지 않아 모델 일부를 CPU에 두고 번갈아 올린다(조금 느림)")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(dev)
        self.pipe = pipe
        self.loaded_model = model_id
        MODEL_ID = model_id
        log("준비 완료")

    def prompt_for(self, req: GenRequest) -> tuple[str, str]:
        pos, neg = PROMPTS.get(req.category, PROMPTS["hair"])
        desc = (req.desc or "").strip() or DEFAULT_DESC.get(req.category, "")
        return pos.format(desc=desc), (req.negative or neg)

    def generate(self, req: GenRequest, log=print) -> dict:
        t0 = time.time()
        img = req.image.convert("RGB")
        W, H = img.size
        mask = req.mask
        if self.dry_run:
            out = _dry_run(img, mask, req.reference)
            return {"image": out, "elapsed_ms": int((time.time() - t0) * 1000), "prompt": self.prompt_for(req)[0], "model": "dry-run"}

        self.load(log)
        import torch

        prompt, negative = self.prompt_for(req)
        # 마스크 주변만 잘라서 그린다: 손톱·타투처럼 작은 영역도 세밀하게 나오고, 나머지는 손대지 않는다
        x0, y0, x1, y1 = _crop_box(mask, W, H, margin=0.45, min_size=int(min(W, H) * 0.5))
        crop = img.crop((x0, y0, x1, y1))
        crop_mask = Image.fromarray(mask).crop((x0, y0, x1, y1))
        cw, ch = crop.size
        # 생성 크기: 긴 변 MAX_SIDE 이하, 8의 배수
        s = min(1.0, self.max_side / max(cw, ch))
        gw, gh = (max(64, int(cw * s)) // 8) * 8, (max(64, int(ch * s)) // 8) * 8
        small = crop.resize((gw, gh), Image.LANCZOS)
        mask_img = crop_mask.resize((gw, gh), Image.BILINEAR)
        gen = torch.Generator(device="cpu").manual_seed(req.seed if req.seed is not None else int(time.time()) % 100000)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=small,
            mask_image=mask_img,
            width=gw,
            height=gh,
            num_inference_steps=req.steps,
            strength=req.strength if req.strength is not None else DEFAULT_STRENGTH.get(req.category, 0.9),
            guidance_scale=req.guidance if req.guidance is not None else DEFAULT_GUIDANCE.get(req.category, 7.0),
            generator=gen,
        )
        if req.reference is not None:
            self.pipe.set_ip_adapter_scale(req.ip_scale)
            kwargs["ip_adapter_image"] = req.reference.convert("RGB").resize((224, 224), Image.LANCZOS)
        else:
            self.pipe.set_ip_adapter_scale(0.0)
            # IP-Adapter가 켜져 있으면 입력이 필요하므로 빈 그림을 준다
            kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (128, 128, 128))
        log(f"생성 시작 {gw}x{gh} (잘라낸 {cw}x{ch} / 전체 {W}x{H}), {req.steps}단계: {prompt}")
        gen_crop = self.pipe(**kwargs).images[0].resize((cw, ch), Image.LANCZOS)
        result = img.copy()
        result.paste(gen_crop, (x0, y0))
        # 마스크 밖은 원본 그대로(가장자리는 부드럽게)
        a = feather(mask, max(3, int(min(W, H) * 0.01)))[..., None]
        merged = (np.asarray(result).astype(np.float32) * a + np.asarray(img).astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(merged)
        if req.category == "hair" and req.color_lock and req.reference is not None and req.color_fn is not None:
            try:
                out_img = req.color_fn(out_img, mask, req.reference, log)
            except Exception as e:  # noqa: BLE001
                log(f"색 맞추기 건너뜀: {e}")
        return {"image": out_img, "elapsed_ms": int((time.time() - t0) * 1000), "prompt": prompt, "model": MODEL_ID, "crop": [x0, y0, x1, y1]}


def _crop_box(mask: np.ndarray, W: int, H: int, margin: float, min_size: int) -> tuple[int, int, int, int]:
    """마스크를 감싸는 상자를 margin만큼 넓히고, 너무 작으면 min_size까지 키운다(이미지 안으로 제한)."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, W, H
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
    size = max(int(max(bw, bh) * (1 + 2 * margin)), min_size)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    x1, y1 = x0 + size, y0 + size
    # 이미지 밖으로 나가면 안으로 민다
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > W:
        x0 -= x1 - W
        x1 = W
    if y1 > H:
        y0 -= y1 - H
        y1 = H
    return max(0, x0), max(0, y0), min(W, x1), min(H, y1)


def _dry_run(img: Image.Image, mask: np.ndarray, reference: Image.Image | None) -> Image.Image:
    """모델 없이: 마스크를 보라색으로 표시하고 참고 사진을 구석에 붙인다."""
    a = (mask.astype(np.float32) / 255.0)[..., None]
    base = np.asarray(img).astype(np.float32)
    tint = np.array([180, 80, 220], np.float32)
    out = (base * (1 - 0.45 * a) + tint * 0.45 * a).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(out)
    if reference is not None:
        th = reference.convert("RGB").copy()
        th.thumbnail((img.width // 4, img.height // 4))
        pil.paste(th, (img.width - th.width - 8, 8))
    return pil


def lock_hair_color(result: Image.Image, mask: np.ndarray, reference: Image.Image, hair_of, log=print, strength: float = 0.85) -> Image.Image:
    """생성 결과의 머리색을 참고 사진 머리색에 맞춘다(Lab 색공간에서 평균·분산 맞추기, 머리 영역만).

    hair_of(img) → 0~1 머리카락 확률(이미지 크기). 결과 쪽은 생성 마스크 안의 머리카락만 손댄다.
    밝기(L)는 평균만 반쯤 맞추고 결의 밝기 차이는 그대로 둬서 윤기·올이 살아 있게 한다.
    """
    import cv2

    res = np.asarray(result.convert("RGB"))
    ref = np.asarray(reference.convert("RGB"))
    hr = hair_of(result)
    hf = hair_of(reference)
    m_res = (hr > 0.5) & (mask > 64)
    m_ref = hf > 0.6
    if m_res.sum() < 200 or m_ref.sum() < 200:
        log(f"색 맞추기 건너뜀: 머리 영역이 작음(결과 {int(m_res.sum())}px, 참고 {int(m_ref.sum())}px)")
        return result
    lab_r = cv2.cvtColor(res, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_f = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)
    pr = lab_r[m_res]
    pf = lab_f[m_ref]
    mu_r, sd_r = pr.mean(0), pr.std(0) + 1e-3
    mu_f, sd_f = pf.mean(0), pf.std(0) + 1e-3
    # 분포 맞추기(분위수 대응): 결과 머리의 어두운 쪽은 참고 머리의 어두운 쪽으로, 밝은 쪽은 밝은 쪽으로.
    # 평균·분산만 맞추면 투톤(반은 검정)처럼 분포가 두 덩어리일 때 어두운 쪽이 남는다
    qs = np.linspace(0, 1, 65)
    out = lab_r.copy()
    for ch in range(3):
        src_q = np.quantile(pr[:, ch], qs)
        dst_q = np.quantile(pf[:, ch], qs)
        mapped = np.interp(lab_r[..., ch], src_q, dst_q)
        # 밝기는 결(윤기)을 조금 살리려고 90%만 옮기고, 색은 전부 옮긴다
        k = 0.9 if ch == 0 else 1.0
        out[..., ch] = lab_r[..., ch] + (mapped - lab_r[..., ch]) * k
    out = out.clip(0, 255)
    # 머리카락 확률 × 생성 마스크를 가중치로 부드럽게 섞는다
    w = (np.clip((hr - 0.3) / 0.4, 0, 1) * np.clip(mask.astype(np.float32) / 255.0, 0, 1))[..., None] * strength
    mixed = out * w + lab_r * (1 - w)
    rgb = cv2.cvtColor(mixed.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    log(f"색 맞추기: 결과 머리 Lab 평균 {mu_r.round(1).tolist()} → 참고 {mu_f.round(1).tolist()}")
    return Image.fromarray(rgb)
