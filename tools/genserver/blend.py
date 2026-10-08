"""생성 결과를 원본에 자연스럽게 붙이기(기본 엔진·AnyDoor 공통).

생성 모델은 자른 영역 전체를 다시 그리므로 (1) 전체 색감이 조금 바뀌고 (2) 원본보다 매끈하거나(잡음 없음) 더 선명하며
(3) 그대로 붙이면 경계가 보인다. 그래서:
  1) 색감 맞추기: 마스크 바로 바깥 띠에서 '원본 − 생성' 평균 차이(Lab)를 재서 마스크 안 생성 결과에 더한다
     (모델이 바꾼 전체 색감만 되돌리고, 마스크 안에서 의도한 변화는 그대로).
  2) 결 맞추기: 고주파(잡음·질감) 세기를 원본 바깥 띠와 비교해, 생성 쪽이 매끈하면 원본과 같은 세기의 잡음을 더하고,
     더 날카로우면 살짝 흐린다.
  3) 경계 섞기: 마스크를 조금 넓혀 부드럽게 한 가중치로 섞는다(마스크 밖은 원본 그대로).
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _hf_sigma(gray: np.ndarray, region: np.ndarray) -> float:
    """고주파 세기(밝기에서 흐린 것을 뺀 표준편차)."""
    if region.sum() < 50:
        return 0.0
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    v = hf[region]
    # 중앙값 기반(경계·윤곽 같은 큰 값에 끌려가지 않게): 순수 잡음이면 표준편차와 같다
    return float(np.median(np.abs(v - np.median(v))) * 1.4826)


def harmonize(orig: Image.Image, gen: Image.Image, mask: np.ndarray, feather_px: int | None = None, seed: int = 0) -> Image.Image:
    """orig·gen: 같은 크기 RGB. mask: 0~255 또는 0/1(생성으로 바꿀 자리)."""
    o = np.asarray(orig.convert("RGB")).astype(np.float32)
    g = np.asarray(gen.convert("RGB").resize(orig.size, Image.LANCZOS)).astype(np.float32)
    H, W = o.shape[:2]
    m = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)
    if m.sum() == 0:
        return orig.copy()
    short = min(H, W)
    fp = feather_px or max(3, int(short * 0.012))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fp + 1, 2 * fp + 1))
    grown = cv2.dilate(m, k)
    ring = (cv2.dilate(grown, k) > 0) & (grown == 0)
    inside = grown > 0

    # 1) 색감: Lab 평균 차이(바깥 띠). 띠가 너무 작으면 건너뛴다
    if ring.sum() > 100:
        ol = cv2.cvtColor(o.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        gl = cv2.cvtColor(g.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        d = (ol[ring] - gl[ring]).mean(0)
        # 과한 보정 방지(밝기 ±12, 색 ±8)
        d = np.clip(d, [-12, -8, -8], [12, 8, 8])
        gl[inside] += d
        g = cv2.cvtColor(gl.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # 2) 결: 고주파 세기 맞추기
    og = cv2.cvtColor(o.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gg = cv2.cvtColor(g.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    so = _hf_sigma(og, ring if ring.sum() > 100 else ~inside)
    sgen = _hf_sigma(gg, m > 0)
    if so > 0 and sgen > 0:
        if sgen < so * 0.85:
            rng = np.random.default_rng(seed)
            n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.6)  # 웹캠 잡음처럼 한두 픽셀 크기
            # 흐리면 세기가 줄므로, 같은 고주파 세기 측정으로 다시 맞춘다
            n *= np.sqrt(max(so * so - sgen * sgen, 0)) / max(_hf_sigma(n, np.ones_like(n, bool)), 1e-6)
            g[inside] += n[inside][:, None]
        elif sgen > so * 1.4:
            b = cv2.GaussianBlur(g, (0, 0), 0.8)
            g[inside] = b[inside]

    # 3) 경계 섞기
    a = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), fp * 0.6)[..., None]
    out = g * a + o * (1 - a)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))
