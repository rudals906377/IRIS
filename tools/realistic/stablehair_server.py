"""사진처럼 그리기 — 헤어 2(Stable-Hair, 디퓨전): 고른 사진의 머리 모양·색을 내 얼굴에 옮긴다. 포트 8768.

HairFastGAN(8767 의 /hair)보다 느리지만(GPU 10~20초) 머리카락 결과 경계가 더 자연스럽다.
Stable-Makeup 과 각자 고친 diffusers 를 저장소 안에 두고 쓰므로 한 프로세스에 함께 올릴 수 없어 따로 띄운다(run.py 가 같이 켠다).

  python stablehair_server.py [--port 8768] [--repo Stable-Hair] [--sd15 Stable-Makeup/models/sd15]

요청(8767 의 /hair 와 같은 모양)
  POST /hair  { face, shape (dataURL), steps=30, seed? } → { image(원래 크기), valid(덮인 곳), aligned(1024), bald(512, 머리를 지운 내 얼굴), elapsed_ms }
  GET  /health → { ok, hair, device, log }

순서(공식 infer_full.py 와 같음): 내 얼굴 정렬(FFHQ 방식) → 512 → ① 머리 지우기(Bald Converter) → ② 고른 사진의 머리를 입히기
(Hair Extractor + Latent IdentityNet) → 1024 → 원래 사진 자리에 되돌림. 어디를 섞을지는 IRIS 생성 서버가 정한다.
GPU 메모리를 아끼려고 단계마다 필요한 모델만 GPU 에 올렸다 내린다(같은 PC 에서 8767 서버도 함께 돈다).
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
app = FastAPI(title="IRIS 헤어 2(Stable-Hair) 서버")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE: dict = {"model": None, "device": "cuda" if torch.cuda.is_available() else "cpu", "log": []}


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


def find_predictor(extra: str | None = None) -> str | None:
    """dlib 68점 얼굴 점 모델: HairFastGAN 설치본을 같이 쓰고, 없으면 Stable-Hair/models 에 받아 둔 것"""
    for p in (
        extra or "",
        os.path.join(HERE, "HairFastGAN", "pretrained_models", "ShapeAdaptor", "shape_predictor_68_face_landmarks.dat"),
        os.path.join(HERE, "Stable-Hair", "models", "shape_predictor_68_face_landmarks.dat"),
    ):
        if os.path.exists(p):
            return p
    return None


class StableHairModel:
    SIZE = 512

    def __init__(self, repo: str, sd15: str, device: str, keep_on_gpu: bool | None = None, dtype: torch.dtype | None = None, predictor: str | None = None) -> None:
        sys.path.insert(0, repo)
        sys.path.insert(0, HERE)
        os.chdir(repo)  # 저장소 안의 고친 diffusers 를 쓴다
        import dlib  # noqa: PLC0415
        from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler  # noqa: PLC0415
        from ref_encoder.adapter import adapter_injection, set_scale  # noqa: PLC0415
        from ref_encoder.latent_controlnet import ControlNetModel  # noqa: PLC0415
        from ref_encoder import reference_control  # noqa: PLC0415
        from ref_encoder.reference_unet import ref_unet  # noqa: PLC0415
        from transformers import CLIPTextModel, CLIPTokenizer  # noqa: PLC0415
        from utils.pipeline import StableHairPipeline  # noqa: PLC0415
        from utils.pipeline_cn import StableDiffusionControlNetPipeline  # noqa: PLC0415

        self.device = device
        self.dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
        if keep_on_gpu is None:
            keep_on_gpu = device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 20 * 2**30
        self.keep = keep_on_gpu or device != "cuda"
        # 참고 머리 특징을 넘길 때 fp16 으로 고정되어 있다 → 모델 정밀도를 따르게(GPU fp16 이면 그대로)
        _update = reference_control.ReferenceAttentionControl.update
        reference_control.ReferenceAttentionControl.update = lambda ctl, writer, dtype=None: _update(ctl, writer, dtype=self.dtype)
        w = lambda *p: torch.load(os.path.join(repo, "models", *p), map_location="cpu")  # noqa: E731
        dt = self.dtype

        vae = AutoencoderKL.from_pretrained(sd15, subfolder="vae", torch_dtype=dt)
        te = CLIPTextModel.from_pretrained(sd15, subfolder="text_encoder", torch_dtype=dt)
        tok = CLIPTokenizer.from_pretrained(sd15, subfolder="tokenizer")
        # ① 머리 지우기: 기본 UNet + Bald Converter(ControlNet)
        # 처음부터 쓸 정밀도로 불러 메모리 최고점을 줄인다(fp32 로 세 벌이면 13GB)
        unet_base = UNet2DConditionModel.from_pretrained(sd15, subfolder="unet", torch_dtype=dt)
        bald = ControlNetModel.from_unet(unet_base)
        bald.load_state_dict(w("stage1", "pytorch_model.bin"), strict=False)
        controlnet = ControlNetModel.from_unet(unet_base)
        controlnet.load_state_dict(w("stage2", "pytorch_model_2.bin"), strict=False)
        # ② 머리 입히기: 어댑터를 끼운 UNet + 참고 머리 인코더(UNet 모양)
        unet_hair = UNet2DConditionModel.from_pretrained(sd15, subfolder="unet", torch_dtype=dt)
        encoder = ref_unet.from_pretrained(sd15, subfolder="unet", torch_dtype=dt)
        encoder.load_state_dict(w("stage2", "pytorch_model.bin"), strict=False)
        adapter = adapter_injection(unet_hair, device="cpu", dtype=torch.float32, use_resampler=False)
        adapter.load_state_dict(w("stage2", "pytorch_model_1.bin"), strict=False)
        set_scale(unet_hair, 1.0)
        for m in (unet_base, bald, controlnet, unet_hair, encoder):
            m.to(dtype=dt).eval()
        # 공용(VAE·글 인코더)은 늘 장치에, 큰 것들은 단계마다
        vae.to(device)
        te.to(device)
        self.stage1 = [unet_base, bald]
        self.stage2 = [unet_hair, encoder, controlnet]
        if self.keep:
            for m in self.stage1 + self.stage2:
                m.to(device)
        self.remove = StableDiffusionControlNetPipeline(
            vae=vae, text_encoder=te, tokenizer=tok, unet=unet_base, controlnet=bald,
            scheduler=UniPCMultistepScheduler.from_pretrained(sd15, subfolder="scheduler"),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        self.pipe = StableHairPipeline(vae=vae, text_encoder=te, tokenizer=tok, unet=unet_hair, controlnet=controlnet,
                                       scheduler=UniPCMultistepScheduler.from_pretrained(sd15, subfolder="scheduler"))
        self.encoder = encoder
        self.repo = repo
        pred = find_predictor(predictor)
        if not pred:
            raise SystemExit("얼굴 점 모델(shape_predictor_68_face_landmarks.dat)이 없습니다. python setup.py 를 다시 실행하세요")
        self.predictor = dlib.shape_predictor(pred)

    def _on(self, mods: list, on: bool) -> None:
        if self.keep:
            return
        for m in mods:
            m.to(self.device if on else "cpu")
        if not on and self.device == "cuda":
            torch.cuda.empty_cache()

    def _aligned(self, img: Image.Image, what: str):
        import face_align  # noqa: PLC0415

        lm = face_align.landmarks68(img, self.predictor)
        if lm is None:
            raise ValueError(f"{what}에서 얼굴을 찾지 못했습니다. 정면에 가깝게, 얼굴이 크게 보이게 해 주세요")
        return face_align.align(img, lm)

    def swap(self, face: Image.Image, shape: Image.Image, steps: int = 30, seed: int | None = None):
        import face_align  # noqa: PLC0415

        S = self.SIZE
        face_al, quad = self._aligned(face, "내 사진")
        try:
            ref_al, _ = self._aligned(shape, "고른 사진")
        except ValueError:
            # 고른 사진에 얼굴이 잘 안 보이면(뒷모습·머리만) 가운데를 정사각형으로
            w, h = shape.size
            s = min(w, h)
            ref_al = shape.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        src = face_al.resize((S, S), Image.LANCZOS)
        ref = np.asarray(ref_al.resize((S, S), Image.LANCZOS))
        g = torch.Generator(device=self.device)
        g.manual_seed(int(seed) if seed is not None else int(time.time()) % 2**31)
        t0 = time.time()
        self._on(self.stage1, True)
        try:
            bald = self.remove(prompt="", negative_prompt="", num_inference_steps=steps, guidance_scale=1.5, width=S, height=S,
                               image=src, controlnet_conditioning_scale=0.9, generator=g).images[0]
        finally:
            self._on(self.stage1, False)
        log(f"  ① 머리 지우기 {time.time() - t0:.1f}초")
        t1 = time.time()
        self._on(self.stage2, True)
        try:
            out = self.pipe("", negative_prompt="", num_inference_steps=steps, guidance_scale=1.5, width=S, height=S,
                            controlnet_condition=np.asarray(bald), controlnet_conditioning_scale=1.0, generator=g,
                            reference_encoder=self.encoder, ref_image=ref).samples
        finally:
            self._on(self.stage2, False)
        log(f"  ② 머리 입히기 {time.time() - t1:.1f}초")
        res = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).resize(face_al.size, Image.LANCZOS)
        warped, valid = face_align.paste_back(res, quad, face.size)
        return warped, valid, res, bald


class HairBody(BaseModel):
    face: str
    shape: str
    color: str | None = None  # 8767 과 같은 모양으로 받되 쓰지 않는다(모양·색을 한 사진에서)
    steps: int = 30
    seed: int | None = None


@app.get("/health")
def health() -> dict:
    return {"ok": True, "hair": STATE["model"] is not None, "device": STATE["device"], "log": STATE["log"][-5:]}


@app.post("/hair")
def hair(body: HairBody) -> dict:
    if STATE["model"] is None:
        raise HTTPException(503, "헤어 2 모델이 아직 준비 전입니다")
    t0 = time.time()
    face, shape = decode(body.face), decode(body.shape)
    cwd = os.getcwd()
    os.chdir(STATE["model"].repo)
    try:
        with torch.inference_mode():
            warped, valid, res, bald = STATE["model"].swap(face, shape, steps=max(10, min(50, body.steps)), seed=body.seed)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        log("헤어 2: GPU 메모리 부족")
        raise HTTPException(503, "GPU 메모리가 모자랍니다. 사진처럼 서버를 run.bat --no-makeup 으로 켜면 여유가 생깁니다") from e
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        log(f"헤어 2 실패: {e}\n{traceback.format_exc()[-1500:]}")
        raise HTTPException(500, f"헤어 2 생성 실패: {e}") from e
    finally:
        os.chdir(cwd)
    ms = int((time.time() - t0) * 1000)
    log(f"헤어 2 완료 {ms}ms")
    return {"image": encode(Image.fromarray(warped)), "valid": encode(Image.fromarray(valid), "PNG"), "aligned": encode(res), "bald": encode(bald), "elapsed_ms": ms}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--repo", default=os.path.join(HERE, "Stable-Hair"))
    p.add_argument("--sd15", default=os.path.join(HERE, "Stable-Makeup", "models", "sd15"))
    p.add_argument("--device", default=None)
    p.add_argument("--predictor", default=None, help="dlib 68점 모델 경로(기본: HairFastGAN 설치본)")
    p.add_argument("--keep-on-gpu", action="store_true", help="모델을 모두 GPU 에 둔다(24GB 이상 카드)")
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto", help="CPU 검증은 bf16 이면 메모리 절반")
    a = p.parse_args()
    device = a.device or STATE["device"]
    STATE["device"] = device
    if device == "cpu":
        sys.path.insert(0, HERE)
        import cpu_patch  # noqa: F401  CPU 검증용
    cwd = os.getcwd()
    log("헤어 2 모델(Stable-Hair) 불러오는 중...")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(a.dtype)
    STATE["model"] = StableHairModel(os.path.abspath(a.repo), os.path.abspath(a.sd15), device, True if a.keep_on_gpu else None, dtype, a.predictor)
    os.chdir(cwd)
    log(f"헤어 2 준비 완료 (장치 {device}) → http://{a.host}:{a.port}/health")
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
