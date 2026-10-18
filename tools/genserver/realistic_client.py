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
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image

REALISTIC_URL = os.environ.get("REALISTIC_URL", "http://127.0.0.1:8767")
# 헤어 2(Stable-Hair, 디퓨전): 같은 PC 에서 따로 도는 서버. 느리고(10~20초) 품질은 HairFastGAN 과 비슷하지만(CPU 비교),
# 고른 사진에서 얼굴을 못 찾아도 사진 가운데를 참고해 그릴 수 있다 → 기본은 HairFastGAN, 그게 못 할 때 헤어 2
STABLEHAIR_URL = os.environ.get("STABLEHAIR_URL", "http://127.0.0.1:8768")


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


def hair_model(choice: str = "auto") -> str | None:
    """쓸 헤어 모델: 'stablehair' | 'hairfast' | None(둘 다 꺼짐). auto 는 HairFastGAN(빠름), 없으면 Stable-Hair"""
    h2 = health(STABLEHAIR_URL)
    h1 = health()
    has2 = bool(h2 and h2.get("hair"))
    has1 = bool(h1 and h1.get("hair"))
    if choice == "hairfast":
        return "hairfast" if has1 else ("stablehair" if has2 else None)
    if choice == "stablehair":
        return "stablehair" if has2 else ("hairfast" if has1 else None)
    return "hairfast" if has1 else ("stablehair" if has2 else None)


def stablehair_on() -> bool:
    h = health(STABLEHAIR_URL)
    return bool(h and h.get("hair"))


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


def hair(face: Image.Image, ref: Image.Image, seed: int | None = None, model: str = "hairfast", steps: int = 30) -> tuple[Image.Image, np.ndarray]:
    if model == "stablehair":
        out = _post("/hair", {"face": _durl(face), "shape": _durl(ref), "seed": seed, "steps": steps}, STABLEHAIR_URL)
    else:
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


def inner_fade(valid: np.ndarray, px: int) -> np.ndarray:
    """valid(덮인 곳) 안쪽에서 1, 가장자리로 갈수록 0 이 되는 가중치(가장자리 px 폭). 사진 테두리에 붙은 쪽은 줄이지 않는다."""
    v = (valid > 0).astype(np.uint8)
    # 사진 밖은 '덮인 곳'으로 쳐서, 사진 테두리 쪽은 깎이지 않게
    pad = cv2.copyMakeBorder(v, px, px, px, px, cv2.BORDER_REPLICATE)
    d = cv2.distanceTransform(pad, cv2.DIST_L2, 5)[px:-px, px:-px]
    return np.clip(d / max(1, px), 0, 1).astype(np.float32)


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


def hair_strands(ref_rgb: np.ndarray, face: np.ndarray, hair: np.ndarray) -> np.ndarray:
    """얼굴 위를 가로지르는 가는 머리카락 가닥(0/1). 얼굴 크기의 3% 보다 가는 어두운 선 중, 머리카락 확률이 있는 곳에 닿은 것만.
    눈썹·눈동자처럼 굵은 어두운 덩어리는 잡히지 않고, 속눈썹은 머리카락과 떨어져 있어 빠진다."""
    f = (face > 0).astype(np.uint8)
    ys, xs = np.where(f > 0)
    if len(xs) < 50:
        return np.zeros_like(f)
    size = max(xs.max() - xs.min(), ys.max() - ys.min())
    k = max(5, int(size * 0.03)) | 1
    g = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thin = ((bh > 28) & (f > 0)).astype(np.uint8)
    hp = cv2.GaussianBlur(hair.astype(np.float32), (0, 0), max(1.0, size * 0.01))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    keep = np.zeros_like(f)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < max(10, size * 0.05):
            continue
        comp = lab == i
        if hp[comp].max() > 0.25:
            keep[comp] = 1
    return cv2.dilate(keep, np.ones((3, 3), np.uint8))


def swap(img: Image.Image, ref: Image.Image, timeout: float = 120) -> tuple[Image.Image, np.ndarray, list[int]]:
    """얼굴 바꾸기(InsightFace inswapper): 내 사진(img)의 얼굴을 고른 사진(ref)의 얼굴 자리에.
    → (고른 사진 크기의 결과, 덮인 곳 마스크(0/1, 얼굴 상자), 얼굴 상자[x0,y0,x1,y1])"""
    body = json.dumps({"face": _durl(img), "target": _durl(ref)}).encode()
    req = urllib.request.Request(REALISTIC_URL + "/swap", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(detail) from e
    out = _img(res["image"]).convert("RGB")
    valid = (np.array(_img(res["valid"]).convert("L")) > 127).astype(np.uint8)
    return out, valid, [int(v) for v in res.get("box", [0, 0, 0, 0])]
