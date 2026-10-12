"""FFHQ 식 얼굴 정렬(HairFastGAN 의 utils/shape_predictor.align_face 와 같은 계산)과 되돌리기.

HairFastGAN 은 1024×1024 로 정렬된 얼굴만 다룬다. 웹캠 사진을 그대로 넣으면 저장소가 알아서 정렬하지만
결과를 원래 사진 자리에 되붙일 수 없다. 여기서는 같은 식으로 정렬하되 **정렬 사각형(quad)** 을 기억해 두었다가,
결과를 어파인 변환으로 원래 좌표에 되돌린다(정렬 사각형은 회전한 정사각형이라 어파인으로 정확히 맞는다).
"""
from __future__ import annotations

import cv2
import dlib
import numpy as np
import PIL.Image
import scipy.ndimage

OUTPUT = 1024
TRANSFORM = 4096


def landmarks68(img: PIL.Image.Image, predictor) -> np.ndarray | None:
    """가장 큰 얼굴의 dlib 68점. 없으면 None."""
    detector = dlib.get_frontal_face_detector()
    arr = np.array(img.convert("RGB"))
    dets = detector(arr, 1)
    if len(dets) == 0:
        return None
    det = sorted(dets, key=lambda d: d.width() * d.height(), reverse=True)[0]
    shape = predictor(arr, det)
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)


def quad_from_landmarks(lm: np.ndarray) -> tuple[np.ndarray, float]:
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    return quad, float(np.hypot(*x) * 2)


def align(img: PIL.Image.Image, lm: np.ndarray) -> tuple[PIL.Image.Image, np.ndarray]:
    """정렬된 1024 얼굴과, 원래 사진 좌표계의 정렬 사각형(좌상·좌하·우하·우상)을 돌려준다."""
    quad, qsize = quad_from_landmarks(lm)
    quad_orig = quad.copy()
    img = img.convert("RGB")
    shrink = int(np.floor(qsize / OUTPUT * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        arr = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "reflect")
        h, w, _ = arr.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]), 1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        arr += (scipy.ndimage.gaussian_filter(arr, [blur, blur, 0]) - arr) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        arr += (np.median(arr, axis=(0, 1)) - arr) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(arr), 0, 255)), "RGB")
        quad += pad[:2]
    img = img.transform((TRANSFORM, TRANSFORM), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    img = img.resize((OUTPUT, OUTPUT), PIL.Image.LANCZOS)
    return img, quad_orig


def paste_back(result: PIL.Image.Image, quad_orig: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """정렬 공간의 결과(1024)를 원래 사진 좌표로 되돌린다. (RGB 배열, 덮인 곳 0/255 마스크)"""
    W, H = size
    n = result.width
    src = np.float32([[0, 0], [0, n], [n, n]])
    dst = np.float32([quad_orig[0], quad_orig[1], quad_orig[2]])
    M = cv2.getAffineTransform(src, dst)
    arr = np.asarray(result.convert("RGB"))
    warped = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.warpAffine(np.full((n, n), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return warped, valid
