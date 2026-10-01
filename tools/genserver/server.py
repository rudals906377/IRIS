"""IRIS 생성 서버(사진 한 장 모드).

웹 앱이 웹캠 한 장 + 참고 사진 + 설명을 보내면, 헤어 모양·네일아트·타투를 생성형 AI로 그려 돌려준다.
사용자 컴퓨터(엔비디아 GPU 권장)에서 돌리고, 사진은 이 컴퓨터 밖으로 나가지 않는다.

실행:  python server.py [--port 8765] [--dry-run] [--host 127.0.0.1]
확인:  http://127.0.0.1:8765/health
"""
from __future__ import annotations

import argparse
import base64
import io
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from masks import Masker
from pipelines import Generator, GenRequest

app = FastAPI(title="IRIS 생성 서버")
# 웹 앱(다른 주소)에서 부를 수 있게. 로컬 전용이므로 모든 출처 허용
import inspect

_cors_kwargs = dict(allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
# 최신 Starlette: https 사이트 → 127.0.0.1 요청(사설망 접근)을 명시적으로 허용해야 한다
if "allow_private_network" in inspect.signature(CORSMiddleware.__init__).parameters:
    _cors_kwargs["allow_private_network"] = True
app.add_middleware(CORSMiddleware, **_cors_kwargs)


@app.middleware("http")
async def allow_private_network(request, call_next):
    """https 사이트(GitHub Pages)에서 내 컴퓨터(127.0.0.1)로 부를 때 크롬이 요구하는 헤더."""
    resp = await call_next(request)
    resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp

masker = Masker()
generator = Generator()
LOG: list[str] = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.append(line)
    del LOG[:-200]


def decode(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        return Image.open(io.BytesIO(base64.b64decode(data_url))).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"그림을 읽지 못했습니다: {e}") from e


def encode(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


class GenerateBody(BaseModel):
    category: str  # hair | nail | tattoo
    image: str  # dataURL
    reference: str | None = None
    desc: str | None = None
    negative: str | None = None
    place: str = "forearmL"
    grow: float = 0.25
    extend: float = 0.0
    steps: int = 30
    strength: float | None = None  # 없으면 분야별 기본값(헤어 0.95, 네일 0.65, 타투 0.45)
    guidance: float | None = None
    ip_scale: float = 0.6
    seed: int | None = None
    mask_only: bool = False


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "ok": True,
        "dry_run": generator.dry_run,
        "device": generator.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "model_loaded": generator.pipe is not None,
        "model": generator.loaded_model,
        "log": LOG[-5:],
    }


@app.post("/generate")
def generate(body: GenerateBody) -> dict:
    if body.category not in ("hair", "nail", "tattoo"):
        raise HTTPException(400, "category는 hair, nail, tattoo 중 하나여야 합니다")
    img = decode(body.image)
    ref = decode(body.reference) if body.reference else None
    t0 = time.time()
    found = None
    if body.category == "hair":
        mask = masker.hair_mask(img, grow=body.grow)
    elif body.category == "nail":
        mask, found = masker.nail_mask(img, extend=body.extend)
        if found == 0:
            raise HTTPException(422, "손톱을 찾지 못했습니다. 손등이 카메라를 향하게 해 주세요")
    else:
        mask = masker.arm_mask(img, place=body.place)
    if int((mask > 0).sum()) < 64:
        raise HTTPException(422, "다시 그릴 영역을 찾지 못했습니다(머리카락·손·팔이 보여야 합니다)")
    log(f"마스크 {body.category} {int((mask > 0).sum())}px {int((time.time() - t0) * 1000)}ms")
    if body.mask_only:
        return {"mask": encode(Image.fromarray(mask), "PNG"), "found": found}
    try:
        out = generator.generate(
            GenRequest(
                category=body.category,
                image=img,
                mask=mask,
                reference=ref,
                desc=body.desc,
                negative=body.negative,
                steps=body.steps,
                strength=body.strength,
                guidance=body.guidance,
                ip_scale=body.ip_scale,
                seed=body.seed,
            ),
            log,
        )
    except Exception as e:  # noqa: BLE001
        log(f"생성 실패: {e}")
        raise HTTPException(500, f"생성 실패: {e}") from e
    log(f"완료 {out['elapsed_ms']}ms")
    return {"image": encode(out["image"]), "mask": encode(Image.fromarray(mask), "PNG"), "elapsed_ms": out["elapsed_ms"], "prompt": out["prompt"], "model": out["model"], "found": found}


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="IRIS 생성 서버")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 마스크만 표시(연결 시험)")
    ap.add_argument("--preload", action="store_true", help="시작할 때 모델을 미리 불러온다")
    a = ap.parse_args()
    generator.dry_run = a.dry_run
    if a.preload and not a.dry_run:
        generator.load(log)
    print(f"IRIS 생성 서버: http://{a.host}:{a.port}  (dry-run={a.dry_run})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
