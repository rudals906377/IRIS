"""Stable-Makeup 의 얼굴 구조 그림(SPIGA 점 → 색 선 그림) 만들기.

원본 spiga_draw.py 는 학습용 `datasets` 를 불러와 추론만 할 때도 실패하므로, 추론에 필요한 부분만 옮겼다.
(얼굴 검출 FaceLib RetinaFace → SPIGA 98점 → 68점 윤곽을 matplotlib 로 검은 바탕에 그림, 512×512)
"""
from __future__ import annotations

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from PIL import Image  # noqa: E402


class StructureDrawer:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from facelib import FaceDetector
        from spiga.inference.config import ModelConfig
        from spiga.inference.framework import SPIGAFramework

        self.detector = FaceDetector(weight_path=None, device=torch.device(device))
        # SPIGA 가중치: 자동 내려받기는 Google Drive 의 바이러스 검사 안내 페이지를 받아 깨진다 → setup 이 미리 받아 둔 파일을 쓴다
        import pkg_resources

        cfg = ModelConfig("300wpublic")
        local = os.path.join(pkg_resources.resource_filename("spiga", "models/weights"), cfg.model_weights)
        if os.path.isfile(local) and os.path.getsize(local) > 10_000_000:
            cfg.load_model_url = False
        self.processor = SPIGAFramework(cfg)

    def landmarks(self, image: Image.Image) -> list | None:
        """사진 속 얼굴들의 SPIGA 점. 얼굴이 없으면 None."""
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        _, boxes, _, _ = self.detector.detect_align(bgr)
        boxes = boxes.cpu().numpy()
        box_ls = [(x, y, x1 - x, y1 - y) for x, y, x1, y1 in boxes]
        if not box_ls:
            return None
        lms = np.array(self.processor.inference(bgr, box_ls)["landmarks"])
        # 점 범위로 다시 상자를 만들어 한 번 더(원본과 같은 두 단계)
        bbox = []
        for lm in lms:
            xs, ys = zip(*[(float(x), float(y)) for x, y in lm])
            bbox.append((min(xs) - 5, min(ys) - 5, max(xs) - min(xs) + 10, max(ys) - min(ys) + 10))
        return self.processor.inference(bgr, bbox)["landmarks"]

    def draw(self, image: Image.Image, size: int = 512) -> Image.Image:
        lms = self.landmarks(image)
        if lms is None:
            return Image.new("RGB", (size, size), (0, 0, 0))
        return conditioning_from_landmarks(lms, size)


def _patch(landmarks, color="lime", closed=False):
    contour = [tuple(p) for p in landmarks]
    ops = [Path.MOVETO] + [Path.LINETO] * (len(contour) - 1)
    facecolor = (0, 0, 0, 0)
    if closed:
        contour.append(contour[0])
        ops.append(Path.CLOSEPOLY)
        facecolor = color
    return patches.PathPatch(Path(contour, ops), facecolor=facecolor, edgecolor=color, lw=4)


def conditioning_from_landmarks(landmarks_, size: int = 512) -> Image.Image:
    dpi = 72
    fig, ax = plt.subplots(1, figsize=[size / dpi, size / dpi], tight_layout={"pad": 0})
    fig.set_dpi(dpi)
    ax.imshow(np.zeros((size, size, 3)))
    for lm in landmarks_:
        for p in (
            _patch(lm[0:17]),
            _patch(lm[17:22], "yellow"),
            _patch(lm[22:27], "yellow"),
            _patch(lm[27:31], "orange"),
            _patch(lm[31:36], "orange"),
            _patch(lm[36:42], "magenta", True),
            _patch(lm[42:48], "magenta", True),
            _patch(lm[48:60], "cyan", True),
            _patch(lm[60:68], "blue", True),
        ):
            ax.add_patch(p)
        plt.axis("off")
        fig.canvas.draw()
    buffer, (w, h) = fig.canvas.print_to_buffer()
    assert w == h == size, (w, h, size)
    out = np.frombuffer(buffer, np.uint8).reshape((h, w, 4))[:, :, :3]
    plt.close(fig)
    return Image.fromarray(out.copy())
