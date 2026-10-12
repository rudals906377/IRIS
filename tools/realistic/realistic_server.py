"""사진처럼 그리기 서버: 고른 사진의 **헤어 모양·색**(HairFastGAN)과 **메이크업**(Stable-Makeup)을 내 사진에 옮긴다.

IRIS 생성 서버(8765)가 부르는 작은 HTTP 래퍼. 포트 8767.
  python realistic_server.py [--port 8767] [--hair-dir ../HairFastGAN] [--makeup-dir ../Stable-Makeup] [--no-makeup] [--no-hair]

요청
  POST /hair    { face, shape, color? (dataURL) }  → { image (원래 크기, 머리를 바꾼 사진), valid (덮인 곳 마스크), aligned (1024), elapsed_ms }
                 color 를 비우면 shape 사진의 머리 모양과 색을 함께 옮긴다
  POST /makeup  { face, makeup (dataURL), steps=30, guidance=1.6, seed? } → { image (원래 크기), valid (얼굴 상자 마스크), elapsed_ms }
  GET  /health  → { ok, hair, makeup, device, log }

두 모델 모두 **얼굴이 정면에 가깝고 크게 보이는 사진**에서 잘 된다. 얼굴을 찾아 정렬·자르고 결과를 원래 자리에 되돌려 주며,
어디를 어떻게 섞을지(머리카락·얼굴 피부 마스크, 색감 맞추기)는 IRIS 생성 서버가 한다.
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="IRIS 사진처럼 그리기 서버")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {"hair": None, "makeup": None, "device": "cuda" if torch.cuda.is_available() else "cpu", "log": [], "hair_dir": "", "makeup_dir": ""}


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE["log"].append(line)
    del STATE["log"][:-100]


def decode(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"그림을 읽지 못했습니다: {e}") from e


def encode(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=93)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


# ---------------- HairFastGAN ----------------
class HairModel:
    """HairFastGAN(AIRI-Institute, MIT): 얼굴 사진 + 모양 사진 + 색 사진 → 1024 정렬 얼굴 → 원래 사진 자리에 되돌림."""

    def __init__(self, repo: str, device: str) -> None:
        sys.path.insert(0, repo)
        sys.path.insert(0, HERE)
        os.chdir(repo)  # 저장소 코드가 'pretrained_models/...' 상대 경로를 쓴다
        import dlib  # noqa: PLC0415
        from hair_swap import HairFast, get_parser  # noqa: PLC0415
        from utils.shape_predictor import align_face  # noqa: PLC0415

        args = get_parser().parse_args(["--device", device, "--batch_size", "1"])  # 메모리 절약(3장을 한 번에 넣으면 14GB 를 넘겼다)
        self.model = HairFast(args)
        self.device = device
        self.predictor = dlib.shape_predictor(os.path.join(repo, "pretrained_models", "ShapeAdaptor", "shape_predictor_68_face_landmarks.dat"))
        self._align_face = align_face

    def swap(self, face: Image.Image, shape: Image.Image, color: Image.Image, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, Image.Image]:
        """원래 사진 크기로 되돌린 결과(RGB 배열), 덮인 곳 마스크, 정렬 공간의 결과(1024)."""
        import torchvision.transforms.functional as TF  # noqa: PLC0415

        import face_align  # noqa: PLC0415

        lm = face_align.landmarks68(face, self.predictor)
        if lm is None:
            raise ValueError("내 사진에서 얼굴을 찾지 못했습니다. 정면에 가깝게, 얼굴이 크게 보이게 해 주세요")
        face_al, quad = face_align.align(face, lm)
        # 참고 사진(모양·색)은 저장소의 정렬 함수로(되붙일 필요가 없다). 같은 사진이면 한 번만
        try:
            if shape is color:
                (shape_t,) = self._align_face([TF.to_tensor(shape)], self.predictor)
                color_t = shape_t
            else:
                shape_t, color_t = self._align_face([TF.to_tensor(shape), TF.to_tensor(color)], self.predictor)
        except (ValueError, IndexError) as e:
            raise ValueError(f"고른 사진에서 얼굴을 찾지 못했습니다: {e}") from e
        face_t = TF.to_tensor(face_al)
        out = self.model.swap(face_t, shape_t, color_t, align=False, seed=seed)
        out = out.squeeze(0) if out.ndim == 4 else out
        res1024 = TF.to_pil_image(out.detach().float().clamp(0, 1).cpu())
        warped, valid = face_align.paste_back(res1024, quad, face.size)
        return warped, valid, res1024


# ---------------- Stable-Makeup ----------------
class MakeupModel:
    """Stable-Makeup(Apache-2.0): 얼굴 사진 + 화장 참고 사진 → 화장을 옮긴 512 얼굴."""

    def __init__(self, repo: str, device: str) -> None:
        sys.path.insert(0, repo)
        sys.path.insert(0, HERE)
        cwd = os.getcwd()
        os.chdir(repo)  # 저장소 안의 수정된 diffusers 를 쓴다
        try:
            from diffusers import ControlNetModel, DDIMScheduler, UNet2DConditionModel  # noqa: PLC0415
            from detail_encoder.encoder_plus import detail_encoder  # noqa: PLC0415
            from pipeline_sd15 import StableDiffusionControlNetPipeline  # noqa: PLC0415
            from makeup_draw import StructureDrawer  # noqa: PLC0415

            self.device = device
            dtype = torch.float16 if device == "cuda" else torch.float32
            model_id = os.path.join(repo, "models", "sd15")
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            id_enc = ControlNetModel.from_unet(unet)
            pose_enc = ControlNetModel.from_unet(unet)
            self.encoder = detail_encoder(unet, os.path.join(repo, "models", "image_encoder_l"), device, dtype=torch.float32)
            w = os.path.join(repo, "models", "stablemakeup")
            self.encoder.load_state_dict(torch.load(os.path.join(w, "pytorch_model.bin"), map_location="cpu"), strict=False)
            id_enc.load_state_dict(torch.load(os.path.join(w, "pytorch_model_1.bin"), map_location="cpu"), strict=False)
            pose_enc.load_state_dict(torch.load(os.path.join(w, "pytorch_model_2.bin"), map_location="cpu"), strict=False)
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(model_id, safety_checker=None, unet=unet, controlnet=[id_enc, pose_enc], torch_dtype=torch.float32)
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
            if device == "cuda":
                # 10GB 카드: fp16 + 필요한 부분만 GPU 에(저메모리)
                self.pipe.to(dtype=dtype)
                self.encoder.to(dtype=dtype)
                try:
                    self.pipe.enable_model_cpu_offload()
                except Exception:  # noqa: BLE001
                    self.pipe.to("cuda")
                self.encoder.to("cuda")
            else:
                self.pipe.to("cpu")
            self.drawer = StructureDrawer(device)
        finally:
            os.chdir(cwd)

    def face_box(self, img: Image.Image) -> tuple[int, int, int, int] | None:
        """가장 큰 얼굴을 1.6배로 감싼 정사각형(x0, y0, x1, y1). 없으면 None."""
        import cv2  # noqa: PLC0415

        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        _, boxes, _, _ = self.drawer.detector.detect_align(bgr)
        boxes = boxes.cpu().numpy()
        if len(boxes) == 0:
            return None
        x, y, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cx, cy, side = (x + x1) / 2, (y + y1) / 2 + (y1 - y) * 0.05, max(x1 - x, y1 - y) * 1.6
        half = side / 2
        return int(round(cx - half)), int(round(cy - half)), int(round(cx + half)), int(round(cy + half))

    def transfer(self, face: Image.Image, makeup: Image.Image, steps: int = 30, guidance: float = 1.6, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """원래 사진 크기의 결과(RGB 배열)와 얼굴 상자 마스크. 얼굴이 없으면 ValueError."""
        box = self.face_box(face)
        if box is None:
            raise ValueError("내 사진에서 얼굴을 찾지 못했습니다")
        x0, y0, x1, y1 = box
        # 상자가 사진 밖으로 나가면 검게 채워 자른다(결과는 사진 안쪽만 되붙임)
        canvas = Image.new("RGB", (x1 - x0, y1 - y0), (0, 0, 0))
        canvas.paste(face, (-x0, -y0))
        side = canvas.width
        face512 = canvas.resize((512, 512), Image.LANCZOS)
        mbox = self.face_box(makeup)
        mk = makeup
        if mbox is not None:
            mc = Image.new("RGB", (mbox[2] - mbox[0], mbox[3] - mbox[1]), (0, 0, 0))
            mc.paste(makeup, (-mbox[0], -mbox[1]))
            mk = mc
        mk512 = mk.resize((512, 512), Image.LANCZOS)
        pose = self.drawer.draw(face512, 512)
        out = self.encoder.generate(id_image=[face512, pose], makeup_image=mk512, pipe=self.pipe, guidance_scale=guidance, num_inference_steps=steps, seed=seed)
        out = out.resize((side, side), Image.LANCZOS)
        W, H = face.size
        full = np.asarray(face.convert("RGB")).copy()
        valid = np.zeros((H, W), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if sx1 > sx0 and sy1 > sy0:
            full[sy0:sy1, sx0:sx1] = np.asarray(out)[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0]
            valid[sy0:sy1, sx0:sx1] = 255
        return full, valid


# ---------------- HTTP ----------------
class HairBody(BaseModel):
    face: str
    shape: str
    color: str | None = None
    seed: int | None = None


class MakeupBody(BaseModel):
    face: str
    makeup: str
    steps: int = 30
    guidance: float = 1.6
    seed: int | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "hair": STATE["hair"] is not None, "makeup": STATE["makeup"] is not None, "device": STATE["device"], "log": STATE["log"][-5:]}


@app.post("/hair")
def hair(body: HairBody) -> dict:
    if STATE["hair"] is None:
        raise HTTPException(503, "헤어 모델이 꺼져 있습니다(--no-hair 로 켰거나 설치 전)")
    t0 = time.time()
    face, shape = decode(body.face), decode(body.shape)
    color = decode(body.color) if body.color else shape
    try:
        with torch.inference_mode():
            warped, valid, res1024 = STATE["hair"].swap(face, shape, color, seed=body.seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log(f"헤어 실패: {e}")
        raise HTTPException(500, f"헤어 생성 실패: {e}") from e
    ms = int((time.time() - t0) * 1000)
    log(f"헤어 완료 {ms}ms")
    return {"image": encode(Image.fromarray(warped)), "valid": encode(Image.fromarray(valid), "PNG"), "aligned": encode(res1024), "elapsed_ms": ms}


@app.post("/makeup")
def makeup(body: MakeupBody) -> dict:
    if STATE["makeup"] is None:
        raise HTTPException(503, "메이크업 모델이 꺼져 있습니다(--no-makeup 으로 켰거나 설치 전)")
    t0 = time.time()
    try:
        with torch.inference_mode():
            full, valid = STATE["makeup"].transfer(decode(body.face), decode(body.makeup), steps=max(10, min(50, body.steps)), guidance=body.guidance, seed=body.seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:  # noqa: BLE001
        log(f"메이크업 실패: {e}")
        raise HTTPException(500, f"메이크업 생성 실패: {e}") from e
    ms = int((time.time() - t0) * 1000)
    log(f"메이크업 완료 {ms}ms")
    return {"image": encode(Image.fromarray(full)), "valid": encode(Image.fromarray(valid), "PNG"), "elapsed_ms": ms}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--hair-dir", default=os.path.join(HERE, "HairFastGAN"))
    p.add_argument("--makeup-dir", default=os.path.join(HERE, "Stable-Makeup"))
    p.add_argument("--no-hair", action="store_true")
    p.add_argument("--no-makeup", action="store_true")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    device = a.device or STATE["device"]
    STATE["device"] = device
    if device == "cpu":
        sys.path.insert(0, HERE)
        import cpu_patch  # noqa: F401  CPU 검증용: 코드에 박힌 cuda 호출을 CPU 로
    cwd = os.getcwd()
    if not a.no_hair and os.path.isdir(a.hair_dir):
        log("헤어 모델(HairFastGAN) 불러오는 중...")
        STATE["hair"] = HairModel(os.path.abspath(a.hair_dir), device)
        os.chdir(cwd)
        log("헤어 준비 완료")
    if not a.no_makeup and os.path.isdir(a.makeup_dir):
        log("메이크업 모델(Stable-Makeup) 불러오는 중...")
        STATE["makeup"] = MakeupModel(os.path.abspath(a.makeup_dir), device)
        os.chdir(cwd)
        log("메이크업 준비 완료")
    if STATE["hair"] is None and STATE["makeup"] is None:
        raise SystemExit("켤 모델이 없습니다. 먼저  python setup.py")
    log(f"준비 완료 (장치 {device}) → http://{a.host}:{a.port}/health")
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
