"""머리카락 매팅 학생 모델 학습(브라우저 실시간용, 가벼운 U-Net).

입력 384×384 RGB → 출력 알파(0~1). prepare.py가 만든 data/images, data/alphas 를 쓴다.
  python train.py --data data --out out --epochs 20
결과: out/best.pt, out/hair-matte.onnx(384), out/samples/*.png

RTX 3080 기준 5,000장 × 20회 ≈ 40분. 28,000장이면 3~4시간.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

SIZE = 384


class HairData(torch.utils.data.Dataset):
    def __init__(self, names: list[str], root: Path, train: bool) -> None:
        self.names = names
        self.root = root
        self.train = train

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        n = self.names[i]
        img = Image.open(self.root / "images" / f"{n}.jpg").convert("RGB")
        alpha = Image.open(self.root / "alphas" / f"{n}.png").convert("L")
        if self.train:
            # 웹캠처럼: 무작위 확대·이동·회전·좌우반전·색·흐림
            s = random.uniform(0.6, 1.0)
            w = int(img.width * s)
            x0 = random.randint(0, img.width - w)
            y0 = random.randint(0, img.height - w)
            img = img.crop((x0, y0, x0 + w, y0 + w))
            alpha = alpha.crop((x0, y0, x0 + w, y0 + w))
            if random.random() < 0.5:
                img, alpha = img.transpose(Image.FLIP_LEFT_RIGHT), alpha.transpose(Image.FLIP_LEFT_RIGHT)
            ang = random.uniform(-20, 20)
            img = img.rotate(ang, resample=Image.BILINEAR)
            alpha = alpha.rotate(ang, resample=Image.BILINEAR)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
            if random.random() < 0.3:
                # 웹캠 화질: 작게 줄였다 키우기
                k = random.choice([128, 192, 256])
                img = img.resize((k, k), Image.BILINEAR)
        img = img.resize((SIZE, SIZE), Image.BILINEAR)
        alpha = alpha.resize((SIZE, SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.asarray(alpha, np.float32) / 255.0)[None]
        return x, y


def cbr(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DS(nn.Module):
    """깊이별 분리 합성곱 블록(가볍다)."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.pw(self.dw(x))


class HairNet(nn.Module):
    """약 1M 매개변수. 1/2, 1/4, 1/8, 1/16 특징을 모아 원래 크기 알파를 낸다."""

    def __init__(self) -> None:
        super().__init__()
        self.s1 = nn.Sequential(cbr(3, 24, 2), DS(24, 24))  # 1/2
        self.s2 = nn.Sequential(DS(24, 48, 2), DS(48, 48))  # 1/4
        self.s3 = nn.Sequential(DS(48, 96, 2), DS(96, 96), DS(96, 96))  # 1/8
        self.s4 = nn.Sequential(DS(96, 160, 2), DS(160, 160), DS(160, 160))  # 1/16
        self.u3 = DS(160 + 96, 96)
        self.u2 = DS(96 + 48, 48)
        self.u1 = DS(48 + 24, 24)
        self.u0 = nn.Sequential(cbr(24 + 3, 16), nn.Conv2d(16, 1, 3, padding=1))

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        d3 = self.u3(torch.cat([up(f4, f3), f3], 1))
        d2 = self.u2(torch.cat([up(d3, f2), f2], 1))
        d1 = self.u1(torch.cat([up(d2, f1), f1], 1))
        d0 = self.u0(torch.cat([up(d1, x), x], 1))
        return d0  # logit


def grad_loss(p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]  # noqa: E731
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]  # noqa: E731
    return (dx(p) - dx(y)).abs().mean() + (dy(p) - dy(y)).abs().mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    root = Path(a.data)
    names = sorted(p.stem for p in (root / "alphas").glob("*.png") if (root / "images" / f"{p.stem}.jpg").exists())
    if len(names) < 20:
        raise SystemExit(f"데이터가 {len(names)}장뿐입니다. 먼저 prepare.py")
    random.shuffle(names)
    nval = max(8, len(names) // 20)
    val, tr = names[:nval], names[nval:]
    print(f"학습 {len(tr)}장, 검증 {len(val)}장")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dl = torch.utils.data.DataLoader(HairData(tr, root, True), batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=dev == "cuda")
    dv = torch.utils.data.DataLoader(HairData(val, root, False), batch_size=a.batch, shuffle=False, num_workers=a.workers)
    model = HairNet().to(dev)
    print(f"매개변수 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(dl))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev, enabled=dev == "cuda"):
                logit = model(x)
                p = torch.sigmoid(logit)
                # 알파 L1 + 경계 기울기 + 분류(0/1 영역을 확실하게)
                loss = (p - y).abs().mean() + 2.0 * grad_loss(p, y) + 0.5 * F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        model.eval()
        mad = 0.0
        cnt = 0
        with torch.no_grad():
            for k, (x, y) in enumerate(dv):
                x, y = x.to(dev), y.to(dev)
                p = torch.sigmoid(model(x))
                mad += (p - y).abs().mean().item() * x.shape[0]
                cnt += x.shape[0]
                if k == 0 and (ep % 4 == 3 or ep == a.epochs - 1):
                    save_samples(x, y, p, out / "samples" / f"ep{ep + 1:03d}.png")
        mad /= max(cnt, 1)
        print(f"[{ep + 1}/{a.epochs}] 손실 {tot / len(dl):.4f}  검증 평균오차(MAD) {mad * 100:.2f}%")
        if mad < best:
            best = mad
            torch.save(model.state_dict(), out / "best.pt")
    print(f"가장 좋은 검증 MAD {best * 100:.2f}% → {out / 'best.pt'}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    model.eval().cpu()
    for size in (384, 256):
        dummy = torch.zeros(1, 3, size, size)
        path = out / f"hair-matte-{size}.onnx"
        torch.onnx.export(model, dummy, str(path), input_names=["image"], output_names=["logit"], dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}}, opset_version=17)
        print("ONNX 저장:", path)


def save_samples(x, y, p, path: Path) -> None:
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
