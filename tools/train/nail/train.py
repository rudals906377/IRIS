"""손톱 전용 분할 모델 학습(작은 U-Net).

데이터: 손 사진 폴더 + 라벨 도구(web/label.html)가 만든 nail-labels.json
사용법:
  python train.py --images 사진폴더 --labels nail-labels.json --out out --epochs 40
결과: out/best.pt (PyTorch), out/nail-seg.onnx (웹·서버용), out/samples/*.png (검증 그림)

RTX 3080 기준 사진 500장이면 40회 반복에 약 20~30분. CPU로도 돌지만 느리다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance

SIZE = 256


# ---------------- 데이터 ----------------
def load_labels(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    imgs = data["images"] if isinstance(data, dict) else data
    return [e for e in imgs if e.get("done") and e.get("nails")]


def polygon_mask(w: int, h: int, nails: list[dict]) -> Image.Image:
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for n in nails:
        pts = [(x * w, y * h) for x, y in n["pts"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    return m


def crop_box(nails: list[dict], w: int, h: int, jitter: float = 0.0) -> tuple[int, int, int, int]:
    """라벨 손톱들을 감싸는 상자를 손 크기만큼 넓힌다(손톱 상자의 2.6배, 흔들림 jitter)."""
    xs = [x * w for n in nails for x, _ in n["pts"]]
    ys = [y * h for n in nails for _, y in n["pts"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    size = max(max(xs) - min(xs), max(ys) - min(ys)) * 2.6
    size = max(size, min(w, h) * 0.25)
    if jitter:
        cx += random.uniform(-jitter, jitter) * size
        cy += random.uniform(-jitter, jitter) * size
        size *= random.uniform(1 - jitter, 1 + jitter)
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    return x0, y0, int(x0 + size), int(y0 + size)


class NailData(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], images: str, train: bool) -> None:
        self.entries = entries
        self.images = images
        self.train = train

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        e = self.entries[i]
        img = Image.open(os.path.join(self.images, e["file"])).convert("RGB")
        w, h = img.size
        mask = polygon_mask(w, h, e["nails"])
        box = crop_box(e["nails"], w, h, jitter=0.15 if self.train else 0.0)
        img = img.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
        mask = mask.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                img, mask = img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            ang = random.uniform(-35, 35)
            img = img.rotate(ang, resample=Image.BILINEAR)
            mask = mask.rotate(ang, resample=Image.BILINEAR)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy((np.asarray(mask, np.float32) / 255.0 > 0.5).astype(np.float32))[None]
        return x, y


# ---------------- 모델 ----------------
def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """작은 U-Net(약 0.5M 매개변수): 웹에서도 실시간으로 돌 수 있는 크기."""

    def __init__(self, ch=(16, 32, 64, 128)) -> None:
        super().__init__()
        self.enc = nn.ModuleList()
        cin = 3
        for c in ch:
            self.enc.append(block(cin, c))
            cin = c
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2))
            self.dec.append(block(ch[i - 1] * 2, ch[i - 1]))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def dice_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()


# ---------------- 학습 ----------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="손 사진 폴더")
    ap.add_argument("--labels", required=True, help="nail-labels.json")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    entries = load_labels(a.labels)
    entries = [e for e in entries if os.path.exists(os.path.join(a.images, e["file"]))]
    if len(entries) < 20:
        raise SystemExit(f"완료된 라벨 사진이 {len(entries)}장뿐입니다(최소 20장, 권장 300장 이상)")
    random.shuffle(entries)
    nval = max(2, len(entries) // 10)
    val, tr = entries[:nval], entries[nval:]
    print(f"학습 {len(tr)}장, 검증 {len(val)}장")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = torch.utils.data.DataLoader(NailData(tr, a.images, True), batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    dv = torch.utils.data.DataLoader(NailData(val, a.images, False), batch_size=a.batch, shuffle=False, num_workers=0)
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            logit = model(x)
            loss = F.binary_cross_entropy_with_logits(logit, y) + dice_loss(logit, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        # 검증: IoU
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for k, (x, y) in enumerate(dv):
                x, y = x.to(dev), y.to(dev)
                p = (torch.sigmoid(model(x)) > 0.5).float()
                inter += (p * y).sum().item()
                union += ((p + y) > 0).float().sum().item()
                if k == 0 and (ep % 5 == 4 or ep == a.epochs - 1):
                    save_samples(x, y, p, out / "samples" / f"ep{ep + 1:03d}.png")
        iou = inter / max(union, 1)
        print(f"[{ep + 1}/{a.epochs}] 손실 {tot / len(dl):.3f}  검증 IoU {iou:.3f}")
        if iou > best:
            best = iou
            torch.save(model.state_dict(), out / "best.pt")
    print(f"가장 좋은 검증 IoU {best:.3f} → {out / 'best.pt'}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    model.eval()
    dummy = torch.zeros(1, 3, SIZE, SIZE, device=dev)
    torch.onnx.export(model, dummy, str(out / "nail-seg.onnx"), input_names=["image"], output_names=["logit"], dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}}, opset_version=17)
    print(f"ONNX 저장: {out / 'nail-seg.onnx'}")


def save_samples(x: torch.Tensor, y: torch.Tensor, p: torch.Tensor, path: Path) -> None:
    n = min(6, x.shape[0])
    rows = []
    for i in range(n):
        img = (x[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt = np.stack([y[i, 0].cpu().numpy() * 255] * 3, -1).astype(np.uint8)
        pr = np.stack([p[i, 0].cpu().numpy() * 255] * 3, -1).astype(np.uint8)
        rows.append(np.concatenate([img, gt, pr], 1))
    Image.fromarray(np.concatenate(rows, 0)).save(path)


if __name__ == "__main__":
    main()
