"""생성 모드에서 "어디를 다시 그릴지" 마스크를 만든다(MediaPipe).

- 헤어: 머리카락 분할 + 머리 주변을 넓힌 영역(길이·모양이 바뀔 수 있게). 얼굴 피부는 뺀다.
- 네일: 손 점 21개 → 손가락 끝 마디 위 손톱 사각형(웹의 nail-place.ts와 같은 비율).
- 타투: 자세 점 → 팔(아래팔·위팔) 캡슐 ∩ 몸 피부, 또는 목·쇄골.
모델 파일은 처음 한 번 내려받아 ~/.cache/iris-genserver 에 둔다.
"""
from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MODEL_BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "seg": f"{MODEL_BASE}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite",
    "hand": f"{MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose": f"{MODEL_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "face": f"{MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}
CACHE = Path(os.environ.get("IRIS_GEN_CACHE", Path.home() / ".cache" / "iris-genserver"))

# 다중 클래스 셀피 분할의 클래스 번호
CLS_BG, CLS_HAIR, CLS_BODY, CLS_FACE, CLS_CLOTHES = 0, 1, 2, 3, 4


def model_path(key: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / MODELS[key].rsplit("/", 1)[-1]
    if not p.exists():
        urllib.request.urlretrieve(MODELS[key], p)
    return str(p)


class Masker:
    """MediaPipe 작업들을 느리게(필요할 때) 만든다."""

    def __init__(self) -> None:
        self._seg = None
        self._hand = None
        self._pose = None
        self._face = None

    # ---- 모델 ----
    def _mp(self):
        import mediapipe as mp  # 느린 import

        return mp

    def seg(self):
        if self._seg is None:
            mp = self._mp()
            from mediapipe.tasks.python import vision

            opts = vision.ImageSegmenterOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path("seg")),
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._seg = vision.ImageSegmenter.create_from_options(opts)
        return self._seg

    def hand(self):
        if self._hand is None:
            mp = self._mp()
            from mediapipe.tasks.python import vision

            opts = vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model_path("hand")), num_hands=2)
            self._hand = vision.HandLandmarker.create_from_options(opts)
        return self._hand

    def pose(self):
        if self._pose is None:
            mp = self._mp()
            from mediapipe.tasks.python import vision

            opts = vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model_path("pose")), num_poses=1)
            self._pose = vision.PoseLandmarker.create_from_options(opts)
        return self._pose

    def face(self):
        if self._face is None:
            mp = self._mp()
            from mediapipe.tasks.python import vision

            opts = vision.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=model_path("face")), num_faces=1)
            self._face = vision.FaceLandmarker.create_from_options(opts)
        return self._face

    def _image(self, img: Image.Image):
        mp = self._mp()
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(img.convert("RGB")))

    # ---- 분할 ----
    def segment(self, img: Image.Image) -> dict[str, np.ndarray]:
        """각 클래스의 확률(0~1, 이미지 크기) 배열."""
        res = self.seg().segment(self._image(img))
        masks = res.confidence_masks
        out = {}
        for name, i in (("bg", CLS_BG), ("hair", CLS_HAIR), ("body", CLS_BODY), ("face", CLS_FACE), ("clothes", CLS_CLOTHES)):
            m = np.squeeze(masks[i].numpy_view()).astype(np.float32)
            if m.shape[:2] != (img.height, img.width):
                m = cv2.resize(m, (img.width, img.height), interpolation=cv2.INTER_LINEAR)
            out[name] = m
        return out

    # ---- 헤어 ----
    def hair_mask(self, img: Image.Image, grow: float = 0.25) -> np.ndarray:
        """머리카락 + 머리 주변(grow: 머리 크기 대비 넓힐 비율). 얼굴 피부·몸 피부는 뺀다. 0~255 uint8."""
        s = self.segment(img)
        hair = (s["hair"] > 0.5).astype(np.uint8)
        ys, xs = np.where(hair > 0)
        if len(xs) < 50:
            # 머리카락이 거의 없으면(민머리·모자) 얼굴 위쪽을 기준으로 영역을 만든다
            face = (s["face"] > 0.5).astype(np.uint8)
            ys, xs = np.where(face > 0)
            if len(xs) < 50:
                return np.zeros((img.height, img.width), np.uint8)
        size = max(xs.max() - xs.min(), ys.max() - ys.min())
        k = max(3, int(size * grow)) | 1
        grown = cv2.dilate(hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        # 얼굴·몸 피부는 그대로 두고, 그 가장자리는 조금만 포함(이마 선·귀 옆이 자연스럽게 이어지게)
        keep = ((s["face"] > 0.6) | (s["body"] > 0.6)).astype(np.uint8)
        kk = max(3, int(size * 0.02)) | 1
        keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        m = grown.copy()
        m[keep > 0] = 0
        m = np.maximum(m, hair)  # 원래 머리카락은 항상 포함
        return (m * 255).astype(np.uint8)

    # ---- 네일 ----
    FINGERS = [(4, 3, 2), (8, 7, 5), (12, 11, 9), (16, 15, 13), (20, 19, 17)]

    def nail_mask(self, img: Image.Image, extend: float = 0.0) -> tuple[np.ndarray, int]:
        """손톱 사각형(끝이 둥근) 마스크와 찾은 손톱 수. extend: 손가락 끝 너머로 늘릴 길이(손톱 길이 비율)."""
        res = self.hand().detect(self._image(img))
        m = np.zeros((img.height, img.width), np.uint8)
        n = 0
        W, H = img.width, img.height
        for lm in res.hand_landmarks:
            p = [(q.x * W, q.y * H) for q in lm]
            for i, (tip, dip, _mcp) in enumerate(self.FINGERS):
                t, d = np.array(p[tip]), np.array(p[dip])
                seg = float(np.linalg.norm(t - d))
                if seg < 4:
                    continue
                dirv = (t - d) / seg
                nrm = np.array([-dirv[1], dirv[0]])
                thumb = i == 0
                length = seg * (0.66 if thumb else 0.6) * (1 + extend)
                width = seg * [0.78, 0.66, 0.66, 0.64, 0.6][i] * 1.15
                c = d + dirv * seg * (0.6 if thumb else 0.62) + dirv * seg * 0.3 * extend
                pts = []
                for k in range(24):
                    a = 2 * math.pi * k / 24
                    u, v = math.cos(a) * width / 2, math.sin(a) * length / 2
                    pts.append(c + nrm * u + dirv * v)
                cv2.fillPoly(m, [np.array(pts, np.int32)], 255)
                n += 1
        return m, n

    # ---- 타투(팔·목) ----
    POSE = {"shoulderL": 11, "shoulderR": 12, "elbowL": 13, "elbowR": 14, "wristL": 15, "wristR": 16, "earL": 7, "earR": 8}

    def arm_mask(self, img: Image.Image, place: str = "forearmL") -> np.ndarray:
        """부위 캡슐 ∩ 몸 피부(넓힌 것). 자세를 못 찾으면 몸 피부 전체."""
        s = self.segment(img)
        skin = ((s["body"] > 0.5) | (s["face"] > 0.5)).astype(np.uint8)
        res = self.pose().detect(self._image(img))
        W, H = img.width, img.height
        if not res.pose_landmarks:
            return (skin * 255).astype(np.uint8)
        P = {k: np.array([res.pose_landmarks[0][i].x * W, res.pose_landmarks[0][i].y * H]) for k, i in self.POSE.items()}
        side = "L" if place.endswith("L") else "R"
        if place.startswith("forearm"):
            a, b, r = P["elbow" + side], P["wrist" + side], 0.22
        elif place.startswith("upperArm"):
            a, b, r = P["shoulder" + side], P["elbow" + side], 0.26
        elif place.startswith("neck"):
            a, b, r = P["ear" + side], P["shoulder" + side], 0.25
        else:  # chest: 어깨 사이 아래
            a = (P["shoulderL"] + P["shoulderR"]) / 2
            b = a + np.array([0, np.linalg.norm(P["shoulderL"] - P["shoulderR"]) * 0.35])
            r = 0.9
        m = np.zeros((H, W), np.uint8)
        rad = int(np.linalg.norm(a - b) * r)
        cv2.line(m, tuple(a.astype(int)), tuple(b.astype(int)), 255, thickness=max(1, rad * 2))
        cv2.circle(m, tuple(a.astype(int)), rad, 255, -1)
        cv2.circle(m, tuple(b.astype(int)), rad, -1)
        k = max(3, int(min(W, H) * 0.03)) | 1
        skin_grown = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m[skin_grown == 0] = 0
        return m


def feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """마스크 가장자리를 부드럽게(0~1 float)."""
    r = max(1, radius) | 1
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (r, r), 0)
