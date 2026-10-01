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
        # 마스크 주변만 잘라서 그린다: 손톱·타투처럼 작은 영역도 세밀하게 나오고, 나머지는 손대지 않는다
        x0, y0, x1, y1 = _crop_box(mask, W, H, margin=0.45, min_size=int(min(W, H) * 0.5))
        crop = img.crop((x0, y0, x1, y1))
        crop_mask = Image.fromarray(mask).crop((x0, y0, x1, y1))
        cw, ch = crop.size
        # 생성 크기: 긴 변 MAX_SIDE 이하, 8의 배수
        s = min(1.0, MAX_SIDE / max(cw, ch))
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
        return {"image": Image.fromarray(merged), "elapsed_ms": int((time.time() - t0) * 1000), "prompt": prompt, "model": MODEL_ID, "crop": [x0, y0, x1, y1]}


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
