"""AnyDoor(물체 합성) 서버: IRIS 생성 서버가 부르는 작은 HTTP 래퍼.

AnyDoor 저장소(https://github.com/ali-vilab/AnyDoor, MIT) 루트에 이 파일을 복사해 그 안의 가상환경으로 실행한다.
  python anydoor_server.py --port 8766 [--no-save-memory]
요청:  POST /compose  { ref_image, ref_mask, tar_image, tar_mask (dataURL), guidance=5.0, steps=30, seed? }
       참고 사진의 마스크(흰색=물체)에 있는 물체를 대상 사진의 마스크 자리에 합성해 돌려준다.
응답:  { image (dataURL JPEG), elapsed_ms }

공식 run_inference.py 의 process_pairs / crop_back / inference_single_image 를 그대로 옮기되,
모델을 한 번만 올리고 save_memory(저메모리 모드: 확산 단계마다 모델 일부를 CPU로)를 켤 수 있게 했다.
10GB 카드(RTX 3080)는 저메모리 모드가 기본이다.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import random
import sys
import time

import cv2
import einops
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import OmegaConf
from PIL import Image
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

from cldm.ddim_hacked import DDIMSampler  # noqa: E402
from cldm.hack import disable_verbosity, enable_sliced_attention  # noqa: E402
from cldm.model import create_model, load_state_dict  # noqa: E402
from datasets.data_utils import (  # noqa: E402
    box2squre,
    box_in_box,
    expand_bbox,
    expand_image_mask,
    get_bbox_from_mask,
    pad_to_square,
    sobel,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

app = FastAPI(title="AnyDoor 합성 서버")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {"model": None, "sampler": None, "save_memory": True, "ckpt": "", "log": []}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE["log"].append(line)
    del STATE["log"][:-100]


def load_model(save_memory: bool) -> None:
    disable_verbosity()
    if save_memory:
        enable_sliced_attention()
    config = OmegaConf.load("./configs/inference.yaml")
    ckpt = config.pretrained_model
    if not os.path.exists(ckpt):
        raise SystemExit(f"가중치가 없습니다: {ckpt} (configs/inference.yaml 의 pretrained_model 확인)")
    log(f"모델 불러오는 중: {ckpt} (저메모리 모드 {'켬' if save_memory else '끔'})")
    model = create_model(config.config_file).cpu()
    model.load_state_dict(load_state_dict(ckpt, location="cpu"))
    if save_memory:
        # 저메모리: 전체를 GPU에 두지 않고, low_vram_shift 가 단계마다 필요한 부분(UNet+ControlNet / DINOv2+VAE)만 올린다.
        # 단, 모델 본체에 직접 달린 버퍼(betas, alphas_cumprod 등)는 GPU에 있어야 한다: DDIM 샘플러가 잡음을
        # betas 가 있는 장치에 만들기 때문(공식 코드는 model.cuda() 뒤에 shift 하지만 그러면 한순간 10GB 를 넘는다).
        model = model.cpu()
        model._buffers = {k: (v.cuda() if v is not None else None) for k, v in model._buffers.items()}
        for prm in model.parameters(recurse=False):
            prm.data = prm.data.cuda()
        if torch.is_tensor(getattr(model, "logvar", None)):
            model.logvar = model.logvar.cuda()
        model.low_vram_shift(is_diffusing=False)
    else:
        model = model.cuda()
    STATE["model"] = model
    STATE["sampler"] = DDIMSampler(model)
    STATE["save_memory"] = save_memory
    STATE["ckpt"] = ckpt
    log("준비 완료")


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    """공식 run_inference.process_pairs 와 같음(무작위 증강만 제거)."""
    ref_box_yyxx = get_bbox_from_mask(ref_mask)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1 - ref_mask_3)
    y1, y2, x1, x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2, x1:x2, :]
    ref_mask = ref_mask[y1:y2, x1:x2]
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=1.2)
    ref_mask_3 = np.stack([ref_mask, ref_mask, ref_mask], -1)
    masked_ref_image = pad_to_square(masked_ref_image, pad_value=255, random=False)
    masked_ref_image = cv2.resize(masked_ref_image, (224, 224)).astype(np.uint8)
    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value=0, random=False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224, 224)).astype(np.uint8)
    ref_mask = ref_mask_3[:, :, 0]
    masked_ref_image_aug = masked_ref_image
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose / 255)

    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1, 1.2])
    tar_box_yyxx_crop = expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx_crop
    cropped_target_image = tar_image[y1:y2, x1:x2, :]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1, y2, x1, x2 = tar_box_yyxx
    ref_image_collage = cv2.resize(ref_image_collage, (x2 - x1, y2 - y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2 - x1, y2 - y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)
    collage = cropped_target_image.copy()
    collage[y1:y2, x1:x2, :] = ref_image_collage
    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2, x1:x2, :] = 1.0
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value=0, random=False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value=0, random=False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value=-1, random=False).astype(np.uint8)
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512, 512)).astype(np.float32)
    collage = cv2.resize(collage, (512, 512)).astype(np.float32)
    collage_mask = (cv2.resize(collage_mask, (512, 512)).astype(np.float32) > 0.5).astype(np.float32)
    masked_ref_image_aug = masked_ref_image_aug / 255
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0
    collage = np.concatenate([collage, collage_mask[:, :, :1]], -1)
    return dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array(tar_box_yyxx_crop))


def crop_back(pred, tar_image, extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1, y2, x1, x2 = tar_box_yyxx_crop
    pred = cv2.resize(pred, (W2, H2))
    m = 5
    gen_image = tar_image.copy()
    if W1 == H1:
        gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
        return gen_image
    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:, pad1:-pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1:-pad2, :, :]
    gen_image[y1 + m : y2 - m, x1 + m : x2 - m, :] = pred[m:-m, m:-m]
    return gen_image


@torch.no_grad()
def compose(ref_image, ref_mask, tar_image, tar_mask, guidance_scale=5.0, ddim_steps=30, seed=None):
    model = STATE["model"]
    sampler = STATE["sampler"]
    save_memory = STATE["save_memory"]
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    if seed is None:
        seed = random.randint(0, 65535)
    torch.manual_seed(seed)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)
    ref = item["ref"]
    hint = item["hint"]
    control = torch.from_numpy(hint.copy()).float().cuda()
    control = einops.rearrange(control[None], "b h w c -> b c h w").clone()
    clip_input = torch.from_numpy(ref.copy()).float().cuda()
    clip_input = einops.rearrange(clip_input[None], "b h w c -> b c h w").clone()
    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(clip_input)]}
    un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1, 3, 224, 224))])]}
    shape = (4, 64, 64)
    if save_memory:
        model.low_vram_shift(is_diffusing=True)
    model.control_scales = [1.0] * 13
    samples, _ = sampler.sample(ddim_steps, 1, shape, cond, verbose=False, eta=0.0, unconditional_guidance_scale=guidance_scale, unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy()
    pred = np.clip(x_samples[0], 0, 255)[1:, :, :]
    return crop_back(pred, tar_image, item["extra_sizes"], item["tar_box_yyxx_crop"])


def decode(data_url: str, mode: str = "RGB") -> np.ndarray:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url))).convert(mode))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"그림을 읽지 못했습니다: {e}") from e


def encode(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=93)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class ComposeBody(BaseModel):
    ref_image: str
    ref_mask: str
    tar_image: str
    tar_mask: str
    guidance: float = 5.0
    steps: int = 30
    seed: int | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "engine": "anydoor", "model_loaded": STATE["model"] is not None, "ckpt": STATE["ckpt"], "save_memory": STATE["save_memory"], "device": "cuda" if torch.cuda.is_available() else "cpu", "log": STATE["log"][-5:]}


@app.post("/compose")
def compose_api(body: ComposeBody) -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "모델이 아직 준비되지 않았습니다")
    t0 = time.time()
    ref = decode(body.ref_image)
    ref_mask = (decode(body.ref_mask, "L") > 127).astype(np.uint8)
    tar = decode(body.tar_image)
    tar_mask = (decode(body.tar_mask, "L") > 127).astype(np.uint8)
    if ref_mask.shape[:2] != ref.shape[:2] or tar_mask.shape[:2] != tar.shape[:2]:
        raise HTTPException(400, "마스크 크기가 그림과 다릅니다")
    if ref_mask.sum() < 16 or tar_mask.sum() < 16:
        raise HTTPException(422, "마스크가 비어 있습니다")
    try:
        out = compose(ref, ref_mask, tar, tar_mask, body.guidance, max(1, min(100, body.steps)), body.seed)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        log(f"GPU 메모리 부족: {e}")
        raise HTTPException(500, "GPU 메모리 부족 — --no-save-memory 를 빼고(저메모리 모드) 다시 실행하세요") from e
    except Exception as e:  # noqa: BLE001
        log(f"합성 실패: {e}")
        raise HTTPException(500, f"합성 실패: {e}") from e
    ms = int((time.time() - t0) * 1000)
    log(f"합성 완료 {ms}ms (단계 {body.steps})")
    return {"image": encode(out), "elapsed_ms": ms}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="AnyDoor 합성 서버")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-save-memory", action="store_true", help="GPU 메모리가 16GB 이상이면 전체를 GPU에 올려 더 빠르게")
    a = ap.parse_args()
    load_model(save_memory=not a.no_save_memory)
    print(f"AnyDoor 서버: http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
