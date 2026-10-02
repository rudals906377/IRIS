"""손톱 자동 라벨(초안): 손 점(MediaPipe)으로 손톱 자리를 짐작하고, 그 자리를 SAM(Segment Anything)에 물어
손톱판 마스크를 받아 라벨 도구 형식(nail-labels.json)으로 저장한다. 사람이 label.html 로 불러와 고치면 완성.

사용법:
  python autolabel.py --images 손사진폴더 --out nail-labels-auto.json [--videos 영상폴더 --frames 30]
  (--videos 를 주면 영상에서 프레임을 뽑아 --images 폴더에 먼저 저장한다)
SAM(facebook/sam-vit-base, 약 375MB)은 처음 한 번 내려받는다. CPU로 사진 한 장에 3~6초.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "genserver"))
from masks import Masker  # noqa: E402

FINGERS = [(4, 3, 2), (8, 7, 5), (12, 11, 9), (16, 15, 13), (20, 19, 17)]
WIDTH = [0.56, 0.5, 0.5, 0.48, 0.44]


def extract_frames(videos: str, images: str, per_video: int) -> int:
    Path(images).mkdir(parents=True, exist_ok=True)
    n = 0
    for v in sorted(Path(videos).iterdir()):
        if v.suffix.lower() not in (".mp4", ".webm", ".mov"):
            continue
        cap = cv2.VideoCapture(str(v))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // per_video)
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                cv2.imwrite(os.path.join(images, f"{v.stem}_{i:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 93])
                n += 1
            i += 1
        cap.release()
    return n


def nail_guesses(mk: Masker, img: Image.Image) -> list[dict]:
    """손 점 → 손톱마다 (중심, 방향, 길이, 폭, 손가락 번호). nail-place.ts 와 같은 비율."""
    res = mk.hand().detect(mk._image(img))
    W, H = img.size
    out = []
    for lm, handed in zip(res.hand_landmarks, res.handedness):
        p = np.array([(q.x * W, q.y * H) for q in lm])
        # 손등이 보이는 정도(nail-place.ts backFacing)
        w, a, b = p[0], p[5], p[17]
        nz = (a[0] - w[0]) * (b[1] - w[1]) - (a[1] - w[1]) * (b[0] - w[0])
        s = nz / max(1e-6, np.linalg.norm(a - w) * np.linalg.norm(b - w))
        facing = s if handed[0].category_name == "Right" else -s
        if facing < -0.1:
            continue  # 손바닥 쪽
        for i, (tip, dip, mcp) in enumerate(FINGERS):
            t, d = p[tip], p[dip]
            seg = float(np.linalg.norm(t - d))
            if seg < 6:
                continue
            dirv = (t - d) / seg
            if i > 0:
                hd = p[mcp] - p[0]
                hd = hd / max(1e-6, np.linalg.norm(hd))
                if float(dirv @ hd) < -0.1:
                    continue  # 접힌 손가락
            thumb = i == 0
            length = seg * (0.55 if thumb else 0.5)
            width = seg * WIDTH[i] * (0.75 if thumb else 1.0)
            c = d + dirv * seg * (0.72 if thumb else 0.75)
            out.append({"c": c, "dir": dirv, "len": length, "width": width, "finger": i, "seg": seg})
    return out


def polygon_from_mask(m: np.ndarray, n_pts: int = 12) -> list[list[float]] | None:
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    # 둘레를 n_pts 등분
    pts = c[:, 0, :].astype(np.float32)
    seglen = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    out = []
    for k in range(n_pts):
        tgt = total * k / n_pts
        j = int(np.searchsorted(cum, tgt, side="right") - 1)
        j = min(j, len(pts) - 1)
        a, b = pts[j], pts[(j + 1) % len(pts)]
        f = (tgt - cum[j]) / max(seglen[j], 1e-6)
        q = a + (b - a) * f
        out.append([float(q[0]), float(q[1])])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="nail-labels-auto.json")
    ap.add_argument("--videos", default=None)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.videos:
        print("프레임 추출:", extract_frames(a.videos, a.images, a.frames))
    from transformers import SamModel, SamProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(dev).eval()
    mk = Masker()
    files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if a.limit:
        files = files[: a.limit]
    entries = []
    stats = {"img": 0, "nail": 0, "sam": 0, "fallback": 0}
    for f in files:
        img = Image.open(f).convert("RGB")
        W, H = img.size
        guesses = nail_guesses(mk, img)
        if not guesses:
            continue
        stats["img"] += 1
        inputs = proc(img, return_tensors="pt").to(dev)
        with torch.no_grad():
            emb = sam.get_image_embeddings(inputs["pixel_values"])
        nails = []
        for g in guesses:
            c, d, L, Wd = g["c"], g["dir"], g["len"], g["width"]
            nrm = np.array([-d[1], d[0]])
            # 상자: 손톱 추정 사각형을 1.4배 넓힌 것
            corners = [c + d * L * 0.7 * sy + nrm * Wd * 0.7 * sx for sx in (-1, 1) for sy in (-1, 1)]
            xs = [q[0] for q in corners]
            ys = [q[1] for q in corners]
            box = [max(0, min(xs)), max(0, min(ys)), min(W - 1, max(xs)), min(H - 1, max(ys))]
            pos = c
            neg = c - d * g["seg"] * 0.55  # 손톱 아래 손가락 피부
            pi = proc(img, input_points=[[[float(pos[0]), float(pos[1])], [float(neg[0]), float(neg[1])]]], input_labels=[[1, 0]], input_boxes=[[box]], return_tensors="pt").to(dev)
            with torch.no_grad():
                o = sam(image_embeddings=emb, input_points=pi["input_points"], input_labels=pi["input_labels"], input_boxes=pi["input_boxes"], multimask_output=True)
            masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), pi["original_sizes"].cpu(), pi["reshaped_input_sizes"].cpu())[0][0]
            scores = o.iou_scores[0, 0].cpu().numpy()
            # 후보 3개 중 손톱 추정 넓이에 가장 가까운 것
            est_area = L * Wd * 0.8
            best = None
            for k in range(masks.shape[0]):
                m = masks[k].numpy()
                area = float(m.sum())
                if area < est_area * 0.3 or area > est_area * 2.5:
                    continue
                # 중심이 마스크 안에 있어야
                if not m[int(min(H - 1, max(0, pos[1]))), int(min(W - 1, max(0, pos[0])))]:
                    continue
                sc = float(scores[k]) - abs(math.log(area / est_area)) * 0.3
                if best is None or sc > best[0]:
                    best = (sc, m)
            if best is not None:
                poly = polygon_from_mask(best[1])
                if poly:
                    nails.append({"pts": [[x / W, y / H] for x, y in poly], "finger": g["finger"], "auto": "sam"})
                    stats["sam"] += 1
                    continue
            # 실패: 손 점 추정 8각형
            pts = []
            for k in range(8):
                ang = 2 * math.pi * k / 8
                q = c + nrm * math.cos(ang) * Wd / 2 * 1.2 + d * math.sin(ang) * L / 2 * 1.2
                pts.append([float(q[0]) / W, float(q[1]) / H])
            nails.append({"pts": pts, "finger": g["finger"], "auto": "guess"})
            stats["fallback"] += 1
        stats["nail"] += len(nails)
        entries.append({"file": f.name, "width": W, "height": H, "done": True, "nails": nails})
        print(f"{f.name}: 손톱 {len(nails)}개", flush=True)
    json.dump({"version": 1, "images": entries}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장:", a.out, stats)


if __name__ == "__main__":
    main()
