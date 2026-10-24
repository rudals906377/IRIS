"""사진처럼 헤어의 긴 머리 이어 그리기.

HairFastGAN·Stable-Hair 는 얼굴에 맞춘 정사각형(FFHQ 정렬: 정수리 위 ~ 턱 아래 조금) 안만 그린다.
그래서 고른 사진이 어깨·가슴까지 오는 긴 머리여도 결과는 사각형 아래 끝(대개 목·옷깃 높이)에서 잘린다.
여기서는 ① 고른 사진에서 머리 길이(턱 아래로 얼굴 길이의 몇 배)를 재고, ② 결과 머리카락이 사각형 아래 가장자리에 닿는
기둥(왼쪽·오른쪽 다발)을 그 길이까지 아래로 늘린 마스크를 만든다. 실제 그리기는 생성 서버의 인페인팅(SD1.5 + IP-Adapter)이
위쪽 머리를 이어받아 한다.
"""
from __future__ import annotations

import cv2
import numpy as np


def face_rows(face_prob: np.ndarray, thr: float = 0.5) -> tuple[int, int] | None:
    """얼굴 피부 마스크의 (위, 아래) 행. 가장 큰 덩어리만."""
    m = (face_prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 200:
        return None
    top = int(stats[i, cv2.CC_STAT_TOP])
    return top, top + int(stats[i, cv2.CC_STAT_HEIGHT]) - 1


def hair_length_ratio(hair_prob: np.ndarray, face_prob: np.ndarray) -> float | None:
    """(머리카락 아래 끝 − 턱) / 얼굴 길이. 머리카락이 사진 아래 끝에 닿으면 적어도 그만큼(실제는 더 길 수 있음)."""
    fr = face_rows(face_prob)
    if fr is None:
        return None
    top, bottom = fr
    fh = max(1, bottom - top)
    hair = (hair_prob > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hair, connectivity=8)
    if n <= 1:
        return None
    # 얼굴 근처의 큰 덩어리들(배경의 머리카락 비슷한 것 제외)
    keep = np.zeros_like(hair)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < fh * fh * 0.02:
            continue
        if stats[i, cv2.CC_STAT_TOP] > bottom + fh * 0.5:
            continue  # 턱 한참 아래에서 시작하는 덩어리는 머리와 이어진 것이 아니다
        keep[lab == i] = 1
    ys = np.where(keep.any(axis=1))[0]
    if len(ys) == 0:
        return None
    return (int(ys.max()) - bottom) / fh


def extension_mask(
    after_hair: np.ndarray,
    valid: np.ndarray,
    face_prob: np.ndarray,
    ratio: float,
    band_px: int,
) -> tuple[np.ndarray | None, str]:
    """늘릴 자리(0/1). 없으면 (None, 이유).

    after_hair: 결과의 머리카락(0/1, valid 안), valid: 생성 모델이 덮은 사각형, face_prob: 내 얼굴 피부 확률,
    ratio: 고른 사진의 머리 길이(턱 아래로 얼굴 길이의 몇 배), band_px: 사각형 가장자리에서 '닿았다'고 볼 폭.
    """
    H, W = valid.shape
    fr = face_rows(face_prob)
    if fr is None:
        return None, "내 얼굴을 못 찾음"
    top, bottom = fr
    fh = max(1, bottom - top)
    target = int(min(H - 1, bottom + ratio * fh))
    v = (valid > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    inner = cv2.erode(v, k)
    band = (v > 0) & (inner == 0)
    ys = np.arange(H)[:, None]
    seeds = (after_hair > 0) & band & (ys > bottom)
    if seeds.sum() < band_px * 4:
        return None, "결과 머리가 사각형 아래 끝에 닿지 않음(짧은 머리)"
    seed_bottom = int(np.where(seeds.any(axis=1))[0].max())
    if target <= seed_bottom + fh * 0.08:
        return None, "고른 사진의 머리가 사각형 안에서 끝남"
    cols = np.where(seeds.any(axis=0))[0]
    out = np.zeros((H, W), np.uint8)
    # 이어진 열 묶음(왼쪽 다발·오른쪽 다발)마다 가운데가 가장 길고 가장자리가 짧게(머리 끝이 둥글게)
    runs: list[list[int]] = []
    for c in cols:
        if runs and c - runs[-1][-1] <= max(2, band_px // 2):
            runs[-1].append(c)
        else:
            runs.append([c])
    rng = np.random.default_rng(int(target) * 7 + len(cols))
    for run in runs:
        if len(run) < 3:
            continue
        x0, x1 = run[0], run[-1]
        n = x1 - x0 + 1
        # 머리 끝이 일자로 잘린 듯 보이지 않게: 다발 가운데가 가장 길고 바깥으로 짧아지며(U자), 굵은 갈래로 조금씩 들쭉날쭉
        wiggle = rng.normal(size=n)
        kw = max(3, n // 6) | 1
        wiggle = cv2.GaussianBlur(wiggle.reshape(1, -1).astype(np.float32), (kw, 1), 0).ravel()
        wiggle = wiggle / (np.abs(wiggle).max() + 1e-6)
        for c in range(x0, x1 + 1):
            col = np.where(seeds[:, c])[0]
            if len(col) == 0:
                continue
            # 사각형 가장자리 띠(섞을 때 원래 사진으로 돌아가는 곳)까지 덮도록 조금 위에서 시작
            y_s = max(0, int(col.min()) - int(band_px * 1.5))
            t = (c - x0) / max(1, x1 - x0)
            shape = np.sin(np.pi * t) ** 0.6  # 가장자리 0 → 가운데 1
            L = target - y_s
            y_e = int(y_s + L * (0.55 + 0.45 * shape) + 0.07 * L * wiggle[c - x0])
            out[y_s : min(H, y_e + 1), c] = 1
    if out.sum() == 0:
        return None, "늘릴 다발이 없음"
    # 다발 사이 틈을 메우고 옆으로 조금 넓힌다(머리 다발은 아래로 갈수록 살짝 퍼진다)
    kk = max(3, int(W * 0.015)) | 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kk * 2 + 1, kk)))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    # 모서리를 둥글게(네모난 마스크는 인페인팅이 네모난 머리 끝을 그린다)
    out = (cv2.GaussianBlur(out.astype(np.float32), (kk * 2 + 1, kk * 2 + 1), 0) > 0.5).astype(np.uint8)
    # 내 얼굴 피부는 덮지 않는다
    out[face_prob > 0.5] = 0
    return out, f"턱 아래 {ratio:.2f}배까지 {int(out.sum())}px"


def prefill(img: np.ndarray, ext: np.ndarray, hair: np.ndarray, src_h: int, source: np.ndarray | None = None) -> np.ndarray:
    """늘릴 자리를 바로 위 머리카락으로 미리 채운다(열마다 위쪽 src_h 행의 머리 픽셀을 아래로 늘여 붙임).

    인페인팅이 처음부터 그리면(strength 1) 머리 대신 옷·배경을 그리는 일이 있어서, 머리 색·결이 깔린 상태에서
    조금만 다시 그리게(strength ~0.75) 하려는 것. img: RGB uint8, ext: 0/1, hair: 0/1(결과 머리카락),
    source: 머리 픽셀을 가져올 그림(없으면 img). 섞은 결과는 가장자리가 옅으므로 모델이 그린 그대로를 준다.
    """
    out = img.copy()
    src_img = img if source is None else source
    cols = np.where(ext.any(axis=0))[0]
    for c in cols:
        ys = np.where(ext[:, c] > 0)[0]
        y0, y1 = int(ys.min()), int(ys.max())
        top = max(0, y0 - src_h)
        src_rows = np.arange(top, y0)
        src_rows = src_rows[hair[top:y0, c] > 0]
        if len(src_rows) < 5:
            continue
        src = src_img[src_rows, c].astype(np.float32)
        n = y1 - y0 + 1
        idx = np.linspace(0, len(src) - 1, n)
        for ch in range(3):
            out[y0 : y1 + 1, c, ch] = np.interp(idx, np.arange(len(src)), src[:, ch]).astype(np.uint8)
    return out
