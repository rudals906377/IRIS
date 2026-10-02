"""머리카락 매팅 모델 학습 데이터 준비.

공개 데이터(CelebAMask-HQ, Hugging Face 미러 FrsECM/CelebAHQ_mask)의 거친 머리카락 라벨을
'선생님' 모델 두 개로 올 단위 알파(0~1)로 정밀화한다.
  - ViTMatte: 라벨 경계 띠(trimap)를 주면 그 안의 올을 알파로 풀어낸다(머리카락 vs 피부·배경 양쪽 경계)
  - BiRefNet-portrait: 사람 전체 알파 → 머리카락 바깥 경계(배경 쪽)가 머리카락 밖으로 새지 않게 곱한다
결과: out/images/NNNNN.jpg (512), out/alphas/NNNNN.png (512, 0~255)

사용법(GPU PC):
  python prepare.py --out data --shards 1          # 먼저 1조각(약 4,700장)으로 시작, 잘 되면 6
  python prepare.py --out data --webcam 내사진폴더    # 내 웹캠 사진도 같은 방식으로 라벨(도메인 맞추기)
  python prepare.py --out data --limit 40 --no-person  # 시험용(적게, BiRefNet 생략)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HF = "https://huggingface.co/datasets/FrsECM/CelebAHQ_mask/resolve/main/data/"
SHARDS = [
    "train-00000-of-00006-ebbd668976eceea4.parquet",
    "train-00001-of-00006-1c40f4ef6f2ab963.parquet",
    "train-00002-of-00006-0cf9c5f0887bf0ea.parquet",
    "train-00003-of-00006-ca789c327922bd6f.parquet",
    "train-00004-of-00006-9956d05db3c5f151.parquet",
    "train-00005-of-00006-c1d31c8a69d22a4c.parquet",
    "test-00000-of-00001-8eec9212829d7cee.parquet",
]
HAIR_CLASS = 3  # 이 미러의 라벨 번호(이미지 위쪽의 큰 영역으로 확인)
OUT_SIZE = 512


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    log("내려받는 중:", url.rsplit("/", 1)[-1])
    tmp = dst.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)
    return dst


def shard_list(n: int) -> list[str]:
    """train 조각 n개(+ test 조각은 --test 로 따로). 이름에 *가 있는 조각은 목록 API로 이름을 알아낸다."""
    names = []
    for s in SHARDS[:6][:n]:
        if "*" in s:
            import json

            info = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/FrsECM/CelebAHQ_mask"))
            cands = [x["rfilename"].rsplit("/", 1)[-1] for x in info["siblings"] if x["rfilename"].startswith("data/" + s.split("*")[0])]
            if cands:
                s = cands[0]
            else:
                continue
        names.append(s)
    return names


class Teachers:
    def __init__(self, device: str, use_person: bool) -> None:
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        self.device = device
        log("ViTMatte 불러오는 중")
        self.vm_proc = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        self.vm = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(device).eval()
        self.person = None
        if use_person:
            from transformers import AutoModelForImageSegmentation

            log("BiRefNet-portrait 불러오는 중")
            self.person = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet-portrait", trust_remote_code=True).to(device).eval()
            if device == "cuda":
                self.person = self.person.half()

    @torch.no_grad()
    def matte(self, img: Image.Image, trimap: np.ndarray) -> np.ndarray:
        inp = self.vm_proc(images=img, trimaps=Image.fromarray(trimap), return_tensors="pt").to(self.device)
        a = self.vm(**inp).alphas[0, 0].float().cpu().numpy()
        return a[: img.height, : img.width].clip(0, 1)

    @torch.no_grad()
    def person_alpha(self, img: Image.Image) -> np.ndarray | None:
        if self.person is None:
            return None
        from torchvision import transforms

        tf = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(img)[None].to(self.device)
        if self.device == "cuda":
            x = x.half()
        p = self.person(x)[-1].sigmoid()[0, 0].float().cpu().numpy()
        return cv2.resize(p, img.size, interpolation=cv2.INTER_LINEAR)


def trimap_from(coarse: np.ndarray, band: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band | 1, band | 1))
    er = cv2.erode(coarse, k)
    di = cv2.dilate(coarse, k)
    t = np.full(coarse.shape, 128, np.uint8)
    t[er > 0] = 255
    t[di == 0] = 0
    return t


def refine(t: Teachers, img: Image.Image, coarse: np.ndarray) -> np.ndarray:
    """거친 0/1 머리카락 마스크 → 0~1 알파."""
    band = max(9, int(min(img.size) * 0.05))
    tri = trimap_from(coarse, band)
    alpha = t.matte(img, tri)
    person = t.person_alpha(img)
    if person is not None:
        # 사람 밖(배경)으로 샌 알파를 없앤다. 사람 알파는 조금 넓혀 올이 깎이지 않게
        pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        person = cv2.dilate(person, pk)
        alpha = np.minimum(alpha, person)
    # 띠 밖은 라벨 그대로(ViTMatte는 띠 안만 바꾼다)
    alpha[tri == 255] = 1.0
    alpha[tri == 0] = 0.0
    return alpha


def save_pair(out: Path, name: str, img: Image.Image, alpha: np.ndarray) -> None:
    im = img.resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS) if img.size != (OUT_SIZE, OUT_SIZE) else img
    a = cv2.resize(alpha, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    im.save(out / "images" / f"{name}.jpg", quality=93)
    Image.fromarray((a * 255).round().astype(np.uint8)).save(out / "alphas" / f"{name}.png")


def do_celeba(a, t: Teachers, out: Path) -> int:
    import pyarrow.parquet as pq

    cache = out / "parquet"
    cache.mkdir(parents=True, exist_ok=True)
    names = shard_list(a.shards)
    if a.test:
        names.append(SHARDS[6])
    done = 0
    for s in names:
        p = download(HF + s, cache / s)
        pf = pq.ParquetFile(p)
        log(f"{s}: {pf.metadata.num_rows}장")
        for batch in pf.iter_batches(batch_size=16, columns=["image_id", "image", "annotation"]):
            for row in batch.to_pylist():
                name = "c" + row["image_id"]
                if (out / "alphas" / f"{name}.png").exists():
                    done += 1
                    continue
                img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
                ann = np.asarray(Image.open(io.BytesIO(row["annotation"]["bytes"])))
                coarse = (ann == a.hair_class).astype(np.uint8)
                if coarse.sum() < img.width * img.height * 0.01:
                    continue  # 머리카락이 거의 없는 사진(모자·민머리)은 뺀다
                if img.size != (1024, 1024):
                    img = img.resize((1024, 1024), Image.LANCZOS)
                    coarse = cv2.resize(coarse, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                alpha = refine(t, img, coarse)
                save_pair(out, name, img, alpha)
                done += 1
                if done % 50 == 0:
                    log(f"{done}장 완료")
                if a.limit and done >= a.limit:
                    return done
    return done


def do_webcam(a, t: Teachers, out: Path) -> int:
    """내 웹캠 사진: MediaPipe 분할로 거친 머리카락을 얻고 같은 방법으로 정밀화(자동 라벨)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "genserver"))
    from masks import Masker

    mk = Masker()
    files = sorted(p for p in Path(a.webcam).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    done = 0
    for p in files:
        name = "w" + p.stem
        if (out / "alphas" / f"{name}.png").exists():
            continue
        img = Image.open(p).convert("RGB")
        # 머리 주변 정사각형으로 자른다(얼굴 점이 있으면 tracker.ts와 같은 비율, 없으면 가운데)
        img = head_crop(mk, img)
        if img is None:
            continue
        hair = mk.segment(img)["hair"]
        coarse = (hair > 0.5).astype(np.uint8)
        if coarse.sum() < img.width * img.height * 0.01:
            continue
        alpha = refine(t, img, coarse)
        save_pair(out, name, img, alpha)
        done += 1
    return done


def head_crop(mk, img: Image.Image) -> Image.Image | None:
    res = mk.face().detect(mk._image(img))
    W, H = img.size
    if not res.face_landmarks:
        s = min(W, H)
        return img.crop(((W - s) // 2, 0, (W - s) // 2 + s, s)).resize((1024, 1024), Image.LANCZOS)
    lm = res.face_landmarks[0]
    xs = [q.x * W for q in lm]
    ys = [q.y * H for q in lm]
    fw, fh = max(xs) - min(xs), max(ys) - min(ys)
    size = min(max(fw * 2.4, fh * 2.0), W, H)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2 - fh * 0.35
    x0 = int(max(0, min(W - size, cx - size / 2)))
    y0 = int(max(0, min(H - size, cy - size / 2)))
    return img.crop((x0, y0, int(x0 + size), int(y0 + size))).resize((1024, 1024), Image.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--shards", type=int, default=1, help="CelebAMask-HQ train 조각 수(0~6, 조각당 약 4,700장)")
    ap.add_argument("--test", action="store_true", help="test 조각(1,500장)도 포함")
    ap.add_argument("--webcam", default=None, help="내 웹캠 사진 폴더(자동 라벨)")
    ap.add_argument("--limit", type=int, default=0, help="시험용: 이 수만큼만")
    ap.add_argument("--no-person", action="store_true", help="BiRefNet(사람 알파) 생략(빠르지만 배경 쪽 경계가 덜 깨끗)")
    ap.add_argument("--hair-class", type=int, default=HAIR_CLASS)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "alphas").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("장치:", dev)
    t = Teachers(dev, not a.no_person)
    n = 0
    if a.shards > 0 or a.test:
        n += do_celeba(a, t, out)
    if a.webcam:
        n += do_webcam(a, t, out)
    log(f"끝: {n}장 → {out}/images, {out}/alphas")


if __name__ == "__main__":
    main()
