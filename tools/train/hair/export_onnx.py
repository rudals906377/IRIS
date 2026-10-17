"""best.pt → ONNX(384·256) 내보내기만 따로:  python export_onnx.py --weights out/best.pt --out out"""
import argparse
from pathlib import Path

import torch

from train import HairNet

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--out", default="out")
a = ap.parse_args()
model = HairNet()
model.load_state_dict(torch.load(a.weights, map_location="cpu"))
model.eval()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for size in (384, 256):
    path = out / f"hair-matte-{size}.onnx"
    torch.onnx.export(model, torch.zeros(1, 3, size, size), str(path), input_names=["image"], output_names=["logit"], dynamic_axes={"image": {0: "n"}, "logit": {0: "n"}}, opset_version=17, external_data=False)
    print("ONNX 저장:", path)
