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
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from masks import Masker
import anydoor_client
from pipelines import Generator, GenRequest, lock_hair_color

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
    color_lock: bool = True  # 헤어: 생성 뒤 머리색을 참고 사진에 맞춤
    engine: str = "default"  # default(SDXL/SD1.5 인페인팅) | anydoor(tools/anydoor 합성 서버)
    reference_mask: str | None = None  # 참고 사진에서 합성할 물체 마스크(dataURL, 흰색=물체). 없으면 서버가 추정


@app.get("/health")
def health() -> dict:
    import torch

    return {
        "ok": True,
        "dry_run": generator.dry_run,
        "device": generator.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "model_loaded": generator.pipe is not None,
        "model": generator.loaded_model,
        "model_choice": generator.model_name,
        "anydoor": anydoor_client.health() is not None,
        "log": LOG[-5:],
    }


# 비동기 작업: 무료 터널(trycloudflare)은 응답을 100초까지만 기다리므로, 오래 걸리는 생성은
# /generate/start 로 작업 번호를 바로 받고 /job/{id} 로 2초마다 물어보는 방식을 쓴다(웹 앱 기본).
JOBS: dict[str, dict] = {}
GPU_LOCK = threading.Lock()  # 생성은 한 번에 하나(GPU 메모리)


def _run_job(job_id: str, body: "GenerateBody") -> None:
    job = JOBS[job_id]
    try:
        with GPU_LOCK:
            job["result"] = generate(body)
        job["status"] = "done"
    except HTTPException as e:
        job.update(status="error", detail=str(e.detail))
    except Exception as e:  # noqa: BLE001
        job.update(status="error", detail=f"생성 실패: {e}")


@app.post("/generate/start")
def generate_start(body: "GenerateBody") -> dict:
    now = time.time()
    for k in [k for k, j in JOBS.items() if now - j["t0"] > 900]:
        del JOBS[k]
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "t0": now}
    threading.Thread(target=_run_job, args=(job_id, body), daemon=True).start()
    return {"job": job_id}


@app.get("/job/{job_id}")
def job_status(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "그런 작업이 없습니다(서버가 다시 켜졌을 수 있음)")
    out = {"status": job["status"], "elapsed_ms": int((time.time() - job["t0"]) * 1000), "log": LOG[-3:]}
    if job["status"] == "done":
        out["result"] = job["result"]
        del JOBS[job_id]
    elif job["status"] == "error":
        out["detail"] = job["detail"]
        del JOBS[job_id]
    return out


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
    # 타투는 자리가 너무 작으면(화면의 0.3% 미만, 예: 목이 살짝만 보임) 그리지 않는다
    if int((mask > 0).sum()) < (max(64, int(mask.size * 0.003)) if body.category == "tattoo" else 64):
        if body.category == "tattoo":
            where = {"forearm": "팔 아래쪽(팔꿈치~손목)", "upperArm": "팔 위쪽(어깨~팔꿈치)", "neck": "목(귀~어깨)", "chest": "가슴(양쪽 어깨)"}
            part = next((v for k, v in where.items() if body.place.startswith(k)), "선택한 부위")
            raise HTTPException(422, f"카메라에 {part} 부위가 보이지 않습니다. 타투 탭의 '위치'를 보이는 부위로 바꾸거나 그 부위를 비춰 주세요")
        raise HTTPException(422, "다시 그릴 영역을 찾지 못했습니다(머리카락·손·팔이 보여야 합니다)")
    log(f"마스크 {body.category} {int((mask > 0).sum())}px {int((time.time() - t0) * 1000)}ms")
    if body.mask_only:
        return {"mask": encode(Image.fromarray(mask), "PNG"), "found": found}
    if body.engine == "anydoor":
        return generate_anydoor(body, img, ref, mask, found, t0)
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
                color_lock=body.color_lock,
                color_fn=lambda res, mask, ref, lg: lock_hair_color(res, mask, ref, lambda im: masker.segment(im)["hair"], lg),
            ),
            log,
        )
    except Exception as e:  # noqa: BLE001
        log(f"생성 실패: {e}")
        raise HTTPException(500, f"생성 실패: {e}") from e
    log(f"완료 {out['elapsed_ms']}ms")
    return {"image": encode(out["image"]), "mask": encode(Image.fromarray(mask), "PNG"), "elapsed_ms": out["elapsed_ms"], "prompt": out["prompt"], "model": out["model"], "found": found}


def generate_anydoor(body: GenerateBody, img: Image.Image, ref: Image.Image | None, mask, found, t0: float) -> dict:
    """AnyDoor 엔진: 참고 물체 마스크 → 대상 자리마다 합성 → (헤어) 색 고정."""
    if ref is None:
        raise HTTPException(400, "AnyDoor 엔진은 참고 사진이 필요합니다(📷 사진 따라하기로 먼저 올리세요)")
    if anydoor_client.health() is None:
        raise HTTPException(503, f"AnyDoor 서버가 꺼져 있습니다({anydoor_client.ANYDOOR_URL}) — tools/anydoor/run.bat")
    given = None
    if body.reference_mask:
        given = (np.asarray(decode(body.reference_mask).convert("L")) > 127).astype(np.uint8)
        if given.shape != (ref.height, ref.width):
            given = cv2_resize_nearest(given, ref.size)
    ref_mask = anydoor_client.ref_object_mask(masker, body.category, ref, given)
    if ref_mask is None:
        raise HTTPException(422, "참고 사진에서 합성할 물체(머리카락·손톱·타투)를 찾지 못했습니다")
    targets = anydoor_client.target_masks(body.category, mask, ref_mask)
    if not targets:
        raise HTTPException(422, "합성할 자리를 찾지 못했습니다")
    steps = max(10, min(50, body.steps))
    guidance = body.guidance if body.guidance is not None else 5.0
    out = img
    log(f"AnyDoor {body.category}: 자리 {len(targets)}곳, {steps}단계")
    for i, tm in enumerate(targets):
        try:
            out = anydoor_client.compose(ref, ref_mask, out, tm, steps=steps, guidance=guidance, seed=body.seed)
        except RuntimeError as e:
            log(str(e))
            raise HTTPException(500, str(e)) from e
        log(f"  자리 {i + 1}/{len(targets)} 완료")
    if body.category == "hair" and body.color_lock:
        try:
            out = lock_hair_color(out, mask, ref, lambda im: masker.segment(im)["hair"], log)
        except Exception as e:  # noqa: BLE001
            log(f"색 맞추기 건너뜀: {e}")
    ms = int((time.time() - t0) * 1000)
    log(f"완료 {ms}ms (AnyDoor)")
    return {"image": encode(out), "mask": encode(Image.fromarray(mask), "PNG"), "elapsed_ms": ms, "prompt": "", "model": "anydoor", "found": found, "targets": len(targets)}


def cv2_resize_nearest(m, size):
    import cv2

    return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="IRIS 생성 서버")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dry-run", action="store_true", help="모델 없이 마스크만 표시(연결 시험)")
    ap.add_argument("--preload", action="store_true", help="시작할 때 모델을 미리 불러온다")
    ap.add_argument("--model", default=None, help="auto(기본: GPU 메모리로 결정) | sd15 | sdxl | Hugging Face 모델 id")
    a = ap.parse_args()
    generator.dry_run = a.dry_run
    if a.model:
        generator.model_name = a.model
    if a.preload and not a.dry_run:
        generator.load(log)
    print(f"IRIS 생성 서버: http://{a.host}:{a.port}  (dry-run={a.dry_run})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
