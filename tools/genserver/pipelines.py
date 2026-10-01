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

import numpy as np
from PIL import Image

from masks import feather

MODEL_ID = os.environ.get("IRIS_GEN_MODEL", "stable-diffusion-v1-5/stable-diffusion-inpainting")
IP_ADAPTER = ("h94/IP-Adapter", "models", "ip-adapter_sd15.bin")
MAX_SIDE = int(os.environ.get("IRIS_GEN_MAX_SIDE", "768"))

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
    strength: float = 0.95
    guidance: float = 7.0
    ip_scale: float = 0.6
    seed: int | None = None


class Generator:
    def __init__(self, dry_run: bool = False, device: str | None = None) -> None:
        self.dry_run = dry_run
        self.pipe = None
        self.device = device
        self.loaded_model: str | None = None

    def load(self, log=print) -> None:
        if self.dry_run or self.pipe is not None:
            return
        import torch
        from diffusers import StableDiffusionInpaintPipeline

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = dev
        dtype = torch.float16 if dev == "cuda" else torch.float32
        log(f"모델 불러오는 중: {MODEL_ID} ({dev})")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, safety_checker=None)
        log("IP-Adapter 불러오는 중")
        pipe.load_ip_adapter(IP_ADAPTER[0], subfolder=IP_ADAPTER[1], weight_name=IP_ADAPTER[2])
        pipe = pipe.to(dev)
        if dev == "cuda":
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            pipe.enable_attention_slicing()
        self.pipe = pipe
        self.loaded_model = MODEL_ID
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
        # 생성 크기: 긴 변 MAX_SIDE 이하, 8의 배수
        s = min(1.0, MAX_SIDE / max(W, H))
        gw, gh = (max(64, int(W * s)) // 8) * 8, (max(64, int(H * s)) // 8) * 8
        small = img.resize((gw, gh), Image.LANCZOS)
        mask_img = Image.fromarray(mask).resize((gw, gh), Image.BILINEAR)
        gen = torch.Generator(device="cpu").manual_seed(req.seed if req.seed is not None else int(time.time()) % 100000)
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=small,
            mask_image=mask_img,
            width=gw,
            height=gh,
            num_inference_steps=req.steps,
            strength=req.strength,
            guidance_scale=req.guidance,
            generator=gen,
        )
        if req.reference is not None:
            self.pipe.set_ip_adapter_scale(req.ip_scale)
            kwargs["ip_adapter_image"] = req.reference.convert("RGB").resize((224, 224), Image.LANCZOS)
        else:
            self.pipe.set_ip_adapter_scale(0.0)
            # IP-Adapter가 켜져 있으면 입력이 필요하므로 빈 그림을 준다
            kwargs["ip_adapter_image"] = Image.new("RGB", (224, 224), (128, 128, 128))
        log(f"생성 시작 {gw}x{gh}, {req.steps}단계: {prompt}")
        result = self.pipe(**kwargs).images[0].resize((W, H), Image.LANCZOS)
        # 마스크 밖은 원본 그대로(가장자리는 부드럽게)
        a = feather(mask, max(3, int(min(W, H) * 0.01)))[..., None]
        merged = (np.asarray(result).astype(np.float32) * a + np.asarray(img).astype(np.float32) * (1 - a)).clip(0, 255).astype(np.uint8)
        return {"image": Image.fromarray(merged), "elapsed_ms": int((time.time() - t0) * 1000), "prompt": prompt, "model": MODEL_ID}


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
