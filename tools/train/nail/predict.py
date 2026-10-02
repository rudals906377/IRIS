"""학습한 손톱 모델을 사진 한 장에 돌려 본다:  python predict.py --onnx out/nail-seg.onnx --image 손.jpg --out 결과.png
(손 전체가 보이는 사진이면 가운데 정사각형을, 손톱 부근을 잘라 넣으면 더 정확하다)
"""
import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--out", default="pred.png")
a = ap.parse_args()
img = Image.open(a.image).convert("RGB")
w, h = img.size
s = min(w, h)
box = ((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)
x = np.asarray(img.crop(box).resize((256, 256)), np.float32) / 255.0
sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
logit = sess.run(None, {"image": x.transpose(2, 0, 1)[None]})[0][0, 0]
p = 1 / (1 + np.exp(-logit))
mask = Image.fromarray((p * 255).astype(np.uint8)).resize((s, s))
over = np.asarray(img.crop(box)).astype(np.float32)
m = (np.asarray(mask).astype(np.float32) / 255.0)[..., None]
over = over * (1 - 0.6 * m) + np.array([60, 160, 255], np.float32) * 0.6 * m
Image.fromarray(over.clip(0, 255).astype(np.uint8)).save(a.out)
print("저장:", a.out, "손톱 픽셀 비율", round(float((p > 0.5).mean()), 4))
