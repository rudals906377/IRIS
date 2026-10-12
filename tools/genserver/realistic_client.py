"""사진처럼 그리기 엔진(tools/realistic 의 서버 8767): 고른 사진의 헤어 모양·색(HairFastGAN)과 메이크업(Stable-Makeup)을
내 사진에 옮긴다. 이 파일은 그 서버를 부르고, 결과를 내 사진에 자연스럽게 섞는 일을 한다.

역할 분담
- realistic 서버: 얼굴 찾기·정렬·생성·원래 자리로 되돌리기 → 결과 사진(원래 크기) + 덮인 곳 마스크(valid)
- 여기: 어디를 섞을지 정한다
    헤어 = (원래 머리카락 ∪ 결과 머리카락) ∩ valid 를 부드럽게 → 결과를 그 자리에, 색감·결은 harmonize
    메이크업 = 얼굴 피부(구멍 메움) ∩ valid 를 부드럽게 → 결과를 그 자리에(harmonize 는 화장 색을 되돌리므로 쓰지 않음)
"""
from __future__ import annotations

import base64
import io
import json
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image

REALISTIC_URL = os.environ.get("REALISTIC_URL", "http://127.0.0.1:8767")


def _durl(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=93)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _img(data_url: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


def health(url: str = REALISTIC_URL, timeout: float = 0.8) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def _post(path: str, body: dict, url: str = REALISTIC_URL) -> dict:
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"사진처럼 그리기 서버 오류 {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"사진처럼 그리기 서버에 연결 못 함({url}): {e}") from e


def hair(face: Image.Image, ref: Image.Image, seed: int | None = None) -> tuple[Image.Image, np.ndarray]:
    out = _post("/hair", {"face": _durl(face), "shape": _durl(ref), "seed": seed})
    res = _img(out["image"]).convert("RGB")
    valid = (np.asarray(_img(out["valid"]).convert("L")) > 127).astype(np.uint8)
    if res.size != face.size:
        res = res.resize(face.size, Image.LANCZOS)
        valid = cv2.resize(valid, face.size, interpolation=cv2.INTER_NEAREST)
    # 정렬 사각형 밖은 검게 비어 있다 → 원래 사진으로 채워 둔다(분할 모델이 검은 곳을 머리카락으로 읽지 않게)
    merged = np.where(valid[..., None] > 0, np.asarray(res), np.asarray(face.convert("RGB")))
    return Image.fromarray(merged.astype(np.uint8)), valid


def attached_to(mask: np.ndarray, anchor: np.ndarray, reach_px: int) -> np.ndarray:
    """mask 의 덩어리 중 anchor(원래 머리카락)에 reach_px 안에서 닿는 것만 남긴다.
    생성 모델은 정렬 사각형 전체를 다시 그리므로 배경(천장·벽)이 흐려지며 '머리카락'으로 읽히는 조각이 생긴다 → 머리와 이어진 것만."""
    m = (mask > 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach_px + 1, 2 * reach_px + 1))
    near = cv2.dilate((anchor > 0).astype(np.uint8), k) > 0
    keep = np.zeros_like(m)
    for i in range(1, n):
        comp = lab == i
        if (comp & near).any():
            keep[comp] = 1
    return keep


def makeup(face: Image.Image, ref: Image.Image, steps: int = 30, guidance: float = 1.6, seed: int | None = None) -> tuple[Image.Image, np.ndarray]:
    out = _post("/makeup", {"face": _durl(face), "makeup": _durl(ref), "steps": steps, "guidance": guidance, "seed": seed})
    return _img(out["image"]).convert("RGB"), (np.asarray(_img(out["valid"]).convert("L")) > 127).astype(np.uint8)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    """0/1 마스크 → 가장자리를 px 만큼 부드럽게(0~1 실수)."""
    m = (mask > 0).astype(np.float32)
    k = max(1, px) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), px * 0.6)


def blend(before: Image.Image, after: Image.Image, soft: np.ndarray) -> Image.Image:
    a = np.asarray(before.convert("RGB")).astype(np.float32)
    b = np.asarray(after.convert("RGB")).astype(np.float32)
    w = soft[..., None]
    return Image.fromarray(np.clip(a * (1 - w) + b * w, 0, 255).astype(np.uint8))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """얼굴 피부 마스크의 눈·입·눈썹 구멍을 메운다(볼록 껍질)."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return m
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    out = np.zeros_like(m)
    cv2.fillConvexPoly(out, hull, 1)
    return out
