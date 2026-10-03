"""AnyDoor 엔진(tools/anydoor 의 합성 서버를 부른다): 참고 사진의 물체를 웹캠 사진의 자리에 합성.

역할 분담
- 참고 물체 마스크: 웹이 보내 주면(reference_mask) 그대로, 없으면 분야별로 추정
    헤어 = 참고 사진의 머리카락 분할, 네일 = 손 점으로 찾은 손톱 중 가장 큰 것, 타투 = 피부 위의 어두운 도안(잉크)
- 대상 자리: 헤어 = 머리카락 마스크 그대로, 타투 = 팔 마스크 가운데에 참고 물체와 같은 비율의 상자,
    네일 = 손톱마다 하나씩(여러 번 합성)
- 합성 결과는 AnyDoor 가 대상 사진에 되붙여 돌려준다. 헤어는 그 뒤 색 고정을 적용한다(pipelines.lock_hair_color).
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

ANYDOOR_URL = os.environ.get("ANYDOOR_URL", "http://127.0.0.1:8766")


def _durl(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=93)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def _mask_durl(m: np.ndarray) -> str:
    return _durl(Image.fromarray((m > 0).astype(np.uint8) * 255), "PNG")


def health(url: str = ANYDOOR_URL, timeout: float = 0.8) -> dict | None:
    try:
        with urllib.request.urlopen(url + "/health", timeout=timeout) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return None


def compose(ref: Image.Image, ref_mask: np.ndarray, tar: Image.Image, tar_mask: np.ndarray, steps: int = 30, guidance: float = 5.0, seed: int | None = None, url: str = ANYDOOR_URL) -> Image.Image:
    body = {"ref_image": _durl(ref), "ref_mask": _mask_durl(ref_mask), "tar_image": _durl(tar), "tar_mask": _mask_durl(tar_mask), "steps": steps, "guidance": guidance, "seed": seed}
    req = urllib.request.Request(url + "/compose", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")[:300]
        raise RuntimeError(f"AnyDoor 서버 오류 {e.code}: {detail}") from e
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"AnyDoor 서버에 연결 못 함({url}): {e}") from e
    data = out["image"].split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def largest_component(m: np.ndarray, min_px: int = 16) -> np.ndarray | None:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return None
    return (lab == k).astype(np.uint8)


def components(m: np.ndarray, min_px: int = 16, limit: int = 5) -> list[np.ndarray]:
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    order = sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])
    out = []
    for k in order:
        if stats[k, cv2.CC_STAT_AREA] < min_px:
            continue
        out.append((lab == k).astype(np.uint8))
        if len(out) >= limit:
            break
    return out


def ref_object_mask(masker, category: str, ref: Image.Image, given: np.ndarray | None) -> np.ndarray | None:
    """참고 사진에서 합성할 물체의 마스크(0/1). given 이 있으면 그것(가장 큰 덩어리)."""
    if given is not None:
        return largest_component(given)
    if category == "hair":
        s = masker.segment(ref)
        return largest_component((s["hair"] > 0.5).astype(np.uint8), 200)
    if category == "nail":
        m, n = masker.nail_mask(ref)
        if n:
            return largest_component(m, 30)
        # 손 전체가 안 보이는 네일 근접 사진: 가운데 45% 상자를 물체로 본다
        H, W = ref.height, ref.width
        return box_mask((H, W), W / 2, H / 2, W * 0.45, H * 0.45)
    # 타투: 사람 안의 피부 위 어두운·무채색 픽셀(잉크)
    s = masker.segment(ref)
    rgb = np.asarray(ref).astype(np.float32)
    mx = rgb.max(-1)
    mn = rgb.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1)
    dark = (mx < 110) & (sat < 0.45)
    person = (s["bg"] < 0.5)
    skin_near = cv2.dilate(((s["body"] > 0.5) | (s["face"] > 0.5)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    ink = (dark & person & skin_near).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return largest_component(ink, 150)


def box_mask(shape: tuple[int, int], cx: float, cy: float, w: float, h: float) -> np.ndarray:
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    x0, y0 = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    x1, y1 = int(min(W, cx + w / 2)), int(min(H, cy + h / 2))
    m[y0:y1, x0:x1] = 1
    return m


def target_masks(category: str, region: np.ndarray, ref_mask: np.ndarray) -> list[np.ndarray]:
    """대상 사진에서 합성할 자리들(0/1 마스크 목록)."""
    if category == "hair":
        return [(region > 0).astype(np.uint8)]
    if category == "nail":
        return components(region, 30, 5)
    # 타투: 팔 영역 가운데에 참고 물체와 같은 가로세로 비율의 상자. 크기는 팔 영역 폭의 70% 정도
    ys, xs = np.where(ref_mask > 0)
    rw, rh = max(1, xs.max() - xs.min()), max(1, ys.max() - ys.min())
    ys2, xs2 = np.where(region > 0)
    if len(xs2) == 0:
        return []
    cx, cy = xs2.mean(), ys2.mean()
    bw = xs2.max() - xs2.min()
    bh = ys2.max() - ys2.min()
    short = min(bw, bh) * 0.7
    if rw >= rh:
        w = short * 1.3
        h = w * rh / rw
    else:
        h = short * 1.3
        w = h * rw / rh
    w = min(w, bw * 0.95)
    h = min(h, bh * 0.95)
    return [box_mask(region.shape, cx, cy, w, h)]
