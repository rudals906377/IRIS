"""머리카락 평가용 정답(가짜 정답) 만들기: 웹캠 같은 영상에서 프레임을 뽑아 강한 오프라인 모델로 머리카락 알파를 만든다.

앱이 쓰는 MediaPipe 와 무관한 모델을 써야 점수가 앱 쪽으로 기울지 않는다:
  1) 얼굴 파싱(SegFormer, jonathandinu/face-parsing, 19부위 중 hair=13)으로 거친 머리카락
  2) ViTMatte 로 경계 띠를 알파로 정밀화, BiRefNet-portrait 로 배경 쪽 새는 것 제거(tools/train/hair/prepare.py 와 같은 방법)
머리 주변 정사각형(앱의 tracker.ts 와 같은 비율)을 1024로 키워 처리하고 원래 프레임 위치에 되붙인다.

사용법(저장소 루트에서):
  python tools/eval/make_hair_gt.py --split eval  --out web/public/testdata/eval/hair     # 평가용(사람 겹치지 않게 고정)
  python tools/eval/make_hair_gt.py --split train --out data/hair_webcam --per 12 --no-person  # 재학습용 웹캠 도메인 자료
결과: <out>/<이름>.png(프레임), <이름>_gt.png(알파), <이름>.webm(앱에 넣을 정지 영상), index.json(머리 영역)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "train" / "hair"))
sys.path.insert(0, str(ROOT / "tools" / "genserver"))
TD = ROOT / "web" / "public" / "testdata"

# 사람이 겹치지 않게 나눈다(같은 사람의 다른 영상이 평가·학습에 섞이면 점수가 부풀려진다)
EVAL = [
    "webcam/10443.mp4", "webcam/4863.mp4", "webcam/11589.mp4", "webcam/15909.mp4", "webcam/4844.mp4",
    "webcam/48686.mp4", "webcam/6083.mp4", "webcam/8746.mp4", "webcam/39781.mp4",
    "male-turn.webm", "female.webm", "face/sam1.webm", "face/will1.webm",
]
# 평가에 쓴 사람의 다른 영상(학습에서도 뺀다)
EXCLUDE = ["webcam/10441.mp4", "webcam/10446.mp4", "webcam/10451.mp4", "webcam/4834.mp4", "webcam/4853.mp4"]
TRAIN_EXTRA = ["turn/23455.mp4", "turn/28324.mp4", "turn/39733.mp4", "turn/41435.mp4", "turn/41460.mp4", "turn/41489.mp4",
               "face/00034_00.webm", "face/00055_00.webm", "face/01992_00.webm", "face/Jensen.webm"]


def train_list() -> list[str]:
    skip = set(EVAL) | set(EXCLUDE)
    cams = sorted(f"webcam/{p.name}" for p in (TD / "webcam").glob("*.mp4"))
    return [c for c in cams if c not in skip] + TRAIN_EXTRA


class Parser:
    def __init__(self) -> None:
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self.torch = torch
        self.proc = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").eval()

    def hair(self, img: Image.Image) -> np.ndarray:
        """머리카락 확률(0~1, img 크기)."""
        torch = self.torch
        with torch.no_grad():
            x = self.proc(images=img, return_tensors="pt")
            logits = self.model(**x).logits
            up = torch.nn.functional.interpolate(logits, size=(img.height, img.width), mode="bilinear", align_corners=False)
            p = up.softmax(1)[0, 13].numpy()
        return p


def head_rect(lm, W: int, H: int) -> tuple[int, int, int] | None:
    """tracker.ts headRect 와 같은 정사각형(x, y, size)."""
    xs = [q.x * W for q in lm]
    ys = [q.y * H for q in lm]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    size = max(fw * 2.4, fh * 2.0)
    if size < 64:
        return None
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2 - fh * 0.35
    sz = min(size, W, H)
    x = max(0.0, min(W - sz, cx - size / 2))
    y = max(0.0, min(H - sz, cy - size / 2))
    return int(x), int(y), int(sz)


def frames(path: Path, k: int):
    c = cv2.VideoCapture(str(path))
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = [0] if n <= 1 else sorted({int(n * (i + 1) / (k + 1)) for i in range(k)})
    for i in idx:
        c.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = c.read()
        if ok:
            yield i, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))


def write_still(img: Image.Image, path: Path) -> None:
    """앱에 넣을 정지 영상(같은 프레임 6장, 12fps, VP8)."""
    a = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"VP80"), 12, (img.width, img.height))
    for _ in range(6):
        w.write(a)
    w.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["eval", "train"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=3, help="영상마다 뽑을 프레임 수")
    ap.add_argument("--no-person", action="store_true", help="BiRefNet 생략(빠름, 학습 자료용)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.split == "train":
        (out / "images").mkdir(exist_ok=True)
        (out / "alphas").mkdir(exist_ok=True)

    from masks import Masker
    from prepare import Teachers, refine, save_pair

    mk = Masker()
    parser = Parser()
    teachers = Teachers("cpu", use_person=not a.no_person)
    index = json.loads((out / "index.json").read_text()) if (out / "index.json").exists() else {}
    srcs = EVAL if a.split == "eval" else train_list()
    for src in srcs:
        p = TD / src
        if not p.exists():
            print("없음:", src)
            continue
        base = src.replace("/", "_").rsplit(".", 1)[0]
        for fi, img in frames(p, a.per):
            name = f"{base}_{fi:04d}"
            if name in index or (a.split == "train" and (out / "alphas" / f"{name}.png").exists()):
                continue
            res = mk.face().detect(mk._image(img))
            if not res.face_landmarks:
                print("얼굴 없음:", name)
                continue
            r = head_rect(res.face_landmarks[0], img.width, img.height)
            if r is None:
                continue
            x, y, s = r
            crop = img.crop((x, y, x + s, y + s)).resize((1024, 1024), Image.LANCZOS)
            hp = parser.hair(crop)
            coarse = (hp > 0.5).astype(np.uint8)
            if coarse.sum() < 1024 * 1024 * 0.01:
                print("머리카락 거의 없음:", name)
                continue
            alpha = refine(teachers, crop, coarse)
            if a.split == "train":
                save_pair(out, name, crop, alpha)
                print("학습 자료", name, flush=True)
                index[name] = {"src": src, "frame": fi}
            else:
                full = np.zeros((img.height, img.width), np.float32)
                full[y : y + s, x : x + s] = cv2.resize(alpha, (s, s), interpolation=cv2.INTER_AREA)
                img.save(out / f"{name}.png")
                Image.fromarray((full * 255).round().astype(np.uint8)).save(out / f"{name}_gt.png")
                write_still(img, out / f"{name}.webm")
                index[name] = {"src": src, "frame": fi, "rect": [x, y, s], "w": img.width, "h": img.height}
                print("평가 정답", name, flush=True)
            (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
