"""실제 머리카락 사진(testdata/ref/hair/*.png)의 색·명암 분포를 잰다 → 염색 자연스러움 평가의 기준.

머리카락 영역(얼굴 파싱 hair, 확실한 안쪽만)에서 감마 공간 sRGB 0~1 로:
  mean     평균 색
  cov      밝기 변동계수(표준편차/평균): 결·명암이 얼마나 살아 있는지
  p10, p90 밝기 10·90 백분위 / 평균: 그늘과 윤기의 폭
결과: tools/eval/results/hair_refs.json
사용법(저장소 루트): python tools/eval/hair_refs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_hair_gt import Parser  # noqa: E402


def stats(rgb: np.ndarray, mask: np.ndarray) -> dict | None:
    m = mask > 0
    if m.sum() < 400:
        return None
    px = rgb[m].astype(np.float64) / 255.0
    L = px @ np.array([0.299, 0.587, 0.114])
    mean_l = L.mean()
    return {
        "mean": px.mean(0).round(4).tolist(),
        "cov": float(L.std() / max(mean_l, 1e-4)),
        "p10": float(np.percentile(L, 10) / max(mean_l, 1e-4)),
        "p90": float(np.percentile(L, 90) / max(mean_l, 1e-4)),
        "n": int(m.sum()),
    }


def main() -> None:
    parser = Parser()
    out = {}
    for p in sorted((ROOT / "web/public/testdata/ref/hair").glob("*.png")):
        img = Image.open(p).convert("RGB")
        img.thumbnail((768, 768))
        hp = parser.hair(img)
        core = cv2.erode((hp > 0.8).astype(np.uint8), np.ones((9, 9), np.uint8))
        s = stats(np.asarray(img), core)
        if s:
            out[p.stem] = s
            print(p.stem, s, flush=True)
    dst = ROOT / "tools/eval/results/hair_refs.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
