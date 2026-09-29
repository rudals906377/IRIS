"""IRIS 데모용 상품 이미지 생성기.

쇼핑몰 사진을 공개 사이트에 올릴 수 없으므로(저작권·약관) 저작권 문제가 없는 상품 이미지를 직접 만든다.
실사처럼 보이도록 원단 결, 주름 음영, 시접·밑단 스티치, 립 조직을 합성하고, 착용 엔진이 쓰는
기준점(keypoints)을 도형 좌표에서 그대로 기록한다(수동 주석 불필요, 기준점 오차 0).

좌표 규칙: 이미지 픽셀 좌표(x 오른쪽, y 아래). 이름의 L/R은 **착용자 기준** 왼쪽/오른쪽이다.
앞면 사진에서 착용자의 왼쪽 소매는 이미지의 오른쪽에 보인다(MediaPipe의 left_* 랜드마크와 같은 규칙).

사용법: python3 tools/garments/generate.py [--out web/public/products]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RNG_SEED = 7
SS = 2  # 슈퍼샘플링 배율(가장자리 계단 현상 제거)
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

Point = tuple[float, float]


# ---------------------------------------------------------------------------
# 기하 유틸


def bezier(p0: Point, p1: Point, p2: Point, p3: Point, n: int = 24) -> list[Point]:
    pts = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t**2
        d = t**3
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def lerp(a: Point, b: Point, t: float) -> Point:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def curve(a: Point, b: Point, bend: float, n: int = 24) -> list[Point]:
    """a→b 직선을 법선 방향으로 bend 픽셀만큼 휘게 한 곡선."""
    mx, my = lerp(a, b, 0.5)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    c = (mx + nx * bend, my + ny * bend)
    return bezier(a, lerp(a, c, 0.66), lerp(b, c, 0.66), b, n)


def mirror_x(p: Point, width: int) -> Point:
    return (width - p[0], p[1])


def poly_mask(size: tuple[int, int], polys: list[list[Point]], scale: int = SS) -> np.ndarray:
    """다각형들을 슈퍼샘플링으로 채운 뒤 축소한 부드러운 알파(0~1)."""
    w, h = size
    img = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(img)
    for poly in polys:
        draw.polygon([(x * scale, y * scale) for x, y in poly], fill=255)
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def polyline_mask(size: tuple[int, int], line: list[Point], width: float, scale: int = SS) -> np.ndarray:
    w, h = size
    img = Image.new("L", (w * scale, h * scale), 0)
    draw = ImageDraw.Draw(img)
    draw.line([(x * scale, y * scale) for x, y in line], fill=255, width=max(1, int(width * scale)), joint="curve")
    img = img.resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def offset_polyline(line: list[Point], dist: float) -> list[Point]:
    """폴리라인을 진행 방향 기준 왼쪽 법선으로 dist만큼 평행 이동(스티치 줄 그리기용)."""
    out = []
    for i, p in enumerate(line):
        a = line[max(0, i - 1)]
        b = line[min(len(line) - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        out.append((p[0] - dy / length * dist, p[1] + dx / length * dist))
    return out


def dashed(line: list[Point], dash: float, gap: float) -> list[list[Point]]:
    """폴리라인을 점선 조각으로 나눈다."""
    pieces: list[list[Point]] = []
    cur: list[Point] = []
    acc = 0.0
    on = True
    for a, b in zip(line, line[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        pos = 0.0
        while pos < seg:
            limit = dash if on else gap
            step = min(limit - acc, seg - pos)
            t0, t1 = pos / seg, (pos + step) / seg
            if on:
                if not cur:
                    cur.append(lerp(a, b, t0))
                cur.append(lerp(a, b, t1))
            pos += step
            acc += step
            if acc >= limit - 1e-6:
                if on and cur:
                    pieces.append(cur)
                cur = []
                acc = 0.0
                on = not on
    if on and len(cur) > 1:
        pieces.append(cur)
    return pieces


# ---------------------------------------------------------------------------
# 원단 합성


def fabric_shading(size: tuple[int, int], alpha: np.ndarray, rng: np.random.Generator, folds: list[tuple[Point, Point, float]]) -> np.ndarray:
    """곱해 줄 밝기 계수(1.0 기준). 조명 기울기 + 주름 + 가장자리 두께감 + 니트 결."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 + 0.05 * (1.0 - (xx / w * 0.6 + yy / h * 0.4))  # 왼쪽 위에서 오는 부드러운 조명

    for (a, b, strength) in folds:
        mask = polyline_mask(size, curve(a, b, rng.uniform(-25, 25)), width=3)
        blur = cv2.GaussianBlur(mask, (0, 0), sigmaX=rng.uniform(9, 16))
        blur /= max(1e-6, float(blur.max()))
        # 주름: 한쪽은 어둡고 한쪽은 밝은 능선 느낌
        ridge = cv2.GaussianBlur(np.roll(mask, 6, axis=1), (0, 0), sigmaX=12)
        ridge /= max(1e-6, float(ridge.max()))
        shade -= strength * blur
        shade += strength * 0.45 * ridge

    # 가장자리 근처를 살짝 어둡게(원단이 접혀 두께가 있는 느낌)
    inside = (alpha > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    shade *= 1.0 - 0.10 * np.exp(-dist / 9.0)

    # 저주파 불균일 + 니트 결(고주파)
    low = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=40)
    low /= max(1e-6, float(np.abs(low).max()))
    knit = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=0.7)
    knit /= max(1e-6, float(np.abs(knit).max()))
    shade *= 1.0 + 0.025 * low + 0.03 * knit
    return shade


def rib_texture(size: tuple[int, int], line: list[Point], width: float, period: float) -> np.ndarray:
    """띠(립 조직) 영역에 곡선을 따르는 세로 골 무늬 밝기 변화."""
    w, h = size
    band = polyline_mask(size, offset_polyline(line, width / 2), width)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 곡선 위 가장 가까운 점까지의 호 길이를 근사하기 위해 샘플 점을 촘촘히 두고 최근접 탐색
    pts = np.array(line, dtype=np.float32)
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(seg)])
    ys, xs = np.nonzero(band > 0.01)
    rib = np.zeros((h, w), np.float32)
    if len(xs):
        d = (xs[:, None] - pts[None, :, 0]) ** 2 + (ys[:, None] - pts[None, :, 1]) ** 2
        idx = d.argmin(1)
        rib[ys, xs] = np.sin(arc[idx] / period * 2 * math.pi)
    return band, rib


@dataclass
class Garment:
    id: str
    name: str
    category: str
    size: tuple[int, int]
    outline: list[list[Point]]
    keypoints: dict[str, Point]
    meta: dict = field(default_factory=dict)


def render_garment(
    g: Garment,
    base: tuple[int, int, int],
    rng: np.random.Generator,
    folds: list[tuple[Point, Point, float]],
    bands: list[tuple[list[Point], float]],
    stitches: list[tuple[list[Point], float]],
    seams: list[list[Point]],
    pattern=None,
    decorate=None,
) -> Image.Image:
    w, h = g.size
    alpha = poly_mask(g.size, g.outline)
    color = np.ones((h, w, 3), np.float32) * (np.array(base, np.float32) / 255.0)
    if pattern is not None:
        color = pattern(color)

    # 목 안쪽(뒷깃 안감): 앞 목선 위로 보이는 옷 뒤판 안쪽. 착용 시에는 원래 옷이 보이는 곳만 덮는다.
    inner = g.meta.get("inner")
    if inner is not None:
        inner_a = poly_mask(g.size, [inner["poly"]]) * (1.0 - alpha)
        inner_c = np.ones_like(color) * (np.array(base, np.float32) / 255.0) * 0.8
        band, rib = rib_texture(g.size, inner["band"], 14, period=5.0)
        inner_c *= (1.0 + 0.08 * band[..., None]) * (1.0 + 0.05 * rib[..., None] * band[..., None])
        color = color * alpha[..., None] + inner_c * inner_a[..., None]
        color /= np.maximum(alpha + inner_a, 1e-6)[..., None]
        alpha = alpha + inner_a

    # 밑단·소매단·목 립 조직
    for line, width in bands:
        band, rib = rib_texture(g.size, line, width, period=5.0)
        color *= (1.0 - 0.06 * band[..., None])
        color *= (1.0 + 0.05 * rib[..., None] * band[..., None])

    if decorate is not None:
        color = decorate(color)

    shade = fabric_shading(g.size, alpha, rng, folds)
    color *= shade[..., None]

    # 박음질(점선)과 봉제선(얇은 그림자 선)
    for line, dist in stitches:
        for piece in dashed(offset_polyline(line, dist), dash=7, gap=4):
            m = polyline_mask(g.size, piece, width=1.6)
            color *= (1.0 - 0.22 * m[..., None])
    for line in seams:
        m = polyline_mask(g.size, line, width=2.2)
        color *= (1.0 - 0.18 * m[..., None])
        hi = polyline_mask(g.size, offset_polyline(line, 2.5), width=1.5)
        color *= (1.0 + 0.08 * hi[..., None])

    rgba = np.dstack([np.clip(color, 0, 1), alpha])
    return Image.fromarray((rgba * 255).round().astype(np.uint8), "RGBA")


# ---------------------------------------------------------------------------
# 상의(티셔츠)


def tee_shape(sleeve: str) -> Garment:
    W, H = 1000, 900
    cx = W / 2
    neckL, neckR = (584.0, 96.0), (416.0, 96.0)
    neckFront = (cx, 172.0)
    shoulderL = (712.0, 142.0)
    armpitL = (742.0, 322.0) if sleeve == "short" else (738.0, 330.0)
    if sleeve == "short":
        outerL, innerL = (905.0, 318.0), (832.0, 392.0)
    else:
        outerL, innerL = (940.0, 700.0), (866.0, 722.0)
    hemL = (734.0, 790.0)

    def side(points_l):
        return [mirror_x(p, W) for p in points_l]

    shoulderR, armpitR, outerR, innerR, hemR = side([shoulderL, armpitL, outerL, innerL, hemL])

    # 외곽선(시계 방향): 오른쪽 목 → 목선(앞) → 왼쪽 목 → 왼쪽 어깨 → 소매 → 겨드랑이 → 옆선 → 밑단 → 반대편
    neckline = bezier(neckR, (neckR[0] + 10, 150), (cx - 60, neckFront[1]), neckFront) + bezier(
        neckFront, (cx + 60, neckFront[1]), (neckL[0] - 10, 150), neckL
    )[1:]
    shoulder_l = curve(neckL, shoulderL, -6)
    if sleeve == "short":
        sleeve_top_l = curve(shoulderL, outerL, -10)
        sleeve_end_l = curve(outerL, innerL, 4)
        sleeve_bot_l = curve(innerL, armpitL, -8)
    else:
        sleeve_top_l = curve(shoulderL, outerL, -26)
        sleeve_end_l = curve(outerL, innerL, 3)
        sleeve_bot_l = curve(innerL, armpitL, 20)
    side_l = curve(armpitL, hemL, 14)
    hem = curve(hemL, hemR, -10)

    def mirror_path(path):
        return [mirror_x(p, W) for p in reversed(path)]

    outline = (
        neckline
        + shoulder_l[1:]
        + sleeve_top_l[1:]
        + sleeve_end_l[1:]
        + sleeve_bot_l[1:]
        + side_l[1:]
        + hem[1:]
        + mirror_path(side_l)[1:]
        + mirror_path(sleeve_bot_l)[1:]
        + mirror_path(sleeve_end_l)[1:]
        + mirror_path(sleeve_top_l)[1:]
        + mirror_path(shoulder_l)[1:-1]
    )
    kp = {
        "neckL": neckL,
        "neckR": neckR,
        "neckFront": neckFront,
        "shoulderL": shoulderL,
        "shoulderR": shoulderR,
        "armpitL": armpitL,
        "armpitR": armpitR,
        "sleeveOuterL": outerL,
        "sleeveInnerL": innerL,
        "sleeveOuterR": outerR,
        "sleeveInnerR": innerR,
        "hemL": hemL,
        "hemR": hemR,
    }
    back_neck = bezier(neckR, (neckR[0] + 30, 116), (neckL[0] - 30, 116), neckL)
    inner_poly = back_neck + list(reversed(neckline))[1:-1]
    armhole_l = curve(shoulderL, armpitL, 10)
    sleeve_poly_l = [shoulderL] + sleeve_top_l[1:] + sleeve_end_l[1:] + sleeve_bot_l[1:] + list(reversed(armhole_l))[1:]
    g = Garment("", "", "top", (W, H), [outline], kp)
    g.meta = {
        "sleeve": sleeve,
        # 부위 번호: 1 몸판, 2 착용자 왼쪽 소매(이미지 오른쪽), 3 착용자 오른쪽 소매
        "parts": {2: sleeve_poly_l, 3: [mirror_x(q, W) for q in sleeve_poly_l]},
        "inner": {"poly": inner_poly, "band": back_neck},
        "paths": {
            "neckline": neckline,
            "sleeveEndL": sleeve_end_l,
            "sleeveEndR": mirror_path(sleeve_end_l),
            "armholeL": curve(shoulderL, armpitL, 10),
            "armholeR": mirror_path(curve(shoulderL, armpitL, 10)),
            "hem": hem,
        },
    }
    return g


def tee_folds(sleeve: str, W: int) -> list[tuple[Point, Point, float]]:
    folds = [
        ((745, 330), (650, 470), 0.10),
        ((255, 330), (350, 470), 0.10),
        ((520, 600), (585, 770), 0.08),
        ((380, 540), (335, 760), 0.07),
        ((640, 450), (700, 620), 0.06),
        ((470, 300), (430, 520), 0.05),
    ]
    if sleeve == "short":
        folds += [((800, 200), (880, 370), 0.08), ((200, 200), (120, 370), 0.08)]
    else:
        folds += [((800, 300), (880, 520), 0.09), ((200, 300), (120, 520), 0.09), ((860, 540), (920, 660), 0.08), ((140, 540), (80, 660), 0.08)]
    return folds


def make_tee(gid: str, name: str, base, sleeve="short", pattern=None, decorate=None, seed=0) -> tuple[Garment, Image.Image]:
    rng = np.random.default_rng(RNG_SEED + seed)
    g = tee_shape(sleeve)
    g.id, g.name = gid, name
    p = g.meta["paths"]
    bands = [(p["neckline"], 18), (p["sleeveEndL"], 26), (p["sleeveEndR"], 26), (p["hem"], 30)]
    stitches = [(p["sleeveEndL"], 22), (p["sleeveEndR"], 22), (p["hem"], 26), (p["hem"], 20), (p["neckline"], 21)]
    seams = [p["armholeL"], p["armholeR"]]
    img = render_garment(g, base, rng, tee_folds(sleeve, g.size[0]), bands, stitches, seams, pattern, decorate)
    return g, img


def stripes(color_a, color_b, period=64, duty=0.45):
    ca = np.array(color_a, np.float32) / 255.0
    cb = np.array(color_b, np.float32) / 255.0

    def apply(color):
        h, w, _ = color.shape
        s = np.arange(h, dtype=np.float32)[:, None] % period
        on = duty * period
        # 줄무늬 안쪽은 양수, 바깥은 음수인 거리 → 1.5px 폭의 부드러운 경계
        d = np.where(s < on, np.minimum(s, on - s), -np.minimum(s - on, period - s))
        m = np.clip(d / 1.5 + 0.5, 0, 1)
        m = np.repeat(m, w, axis=1)[..., None]
        return ca * (1 - m) + cb * m

    return apply


def chest_logo(text: str, fg, accent, y=300, size=110):
    def apply(color):
        h, w, _ = color.shape
        layer = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        font = ImageFont.truetype(FONT_BOLD, size * SS)
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx = w * SS // 2
        # 비대칭 엠블럼(좌우 반전을 알아볼 수 있도록): 고리 + 오른쪽 위로 치우친 눈동자
        r = 44 * SS
        ey = y * SS
        d.ellipse((cx - r, ey - r, cx + r, ey + r), outline=accent + (255,), width=10 * SS)
        pr = 15 * SS
        px, py = cx + 12 * SS, ey - 10 * SS
        d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=accent + (255,))
        ty = ey + r + 26 * SS
        d.text((cx - tw // 2 - bbox[0], ty - bbox[1]), text, font=font, fill=fg + (255,))
        small = ImageFont.truetype(FONT_BOLD, 30 * SS)
        sub = "VIRTUAL FIT · 2026"
        sb = d.textbbox((0, 0), sub, font=small)
        d.text((cx - (sb[2] - sb[0]) // 2 - sb[0], ty + th + 24 * SS), sub, font=small, fill=accent + (255,))
        layer = layer.resize((w, h), Image.LANCZOS)
        arr = np.asarray(layer, np.float32) / 255.0
        # 프린트는 원단보다 약간 거칠게(잉크 결)
        a = arr[..., 3:4]
        return color * (1 - a) + arr[..., :3] * a

    return apply


# ---------------------------------------------------------------------------


def save(g: Garment, img: Image.Image, out: Path, catalog: list[dict], credit: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / f"{g.id}.webp", "WEBP", quality=92, method=6)
    # 부위 라벨 지도(0 투명, 1 몸판, 2·3 소매 …). 보간에 안전하도록 값 × 80 으로 저장한다.
    alpha = np.asarray(img)[..., 3] > 8
    labels = np.where(alpha, 1, 0).astype(np.uint8)
    for part_id, poly in g.meta.get("parts", {}).items():
        # 외곽선의 안티앨리어싱 픽셀까지 소매에 포함되도록 3px 넓힌다(아니면 몸판으로 잘못 분류됨).
        m = (poly_mask(g.size, [poly], scale=1) > 0.5).astype(np.uint8)
        m = cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0
        labels[m & alpha] = part_id
    inner = g.meta.get("inner")
    if inner is not None:
        # 목 안쪽은 넓히지 않는다(앞 목선 립 조직을 침범하지 않도록). 외곽선 바깥쪽 픽셀만.
        outline_a = poly_mask(g.size, g.outline, scale=1) > 0.5
        m = (poly_mask(g.size, [inner["poly"]], scale=1) > 0.5) & ~outline_a
        labels[m & alpha] = 4
    Image.fromarray(labels * 80, "L").save(out / f"{g.id}_parts.png", optimize=True)
    thumb = img.copy()
    thumb.thumbnail((256, 256), Image.LANCZOS)
    thumb.save(out / f"{g.id}_thumb.webp", "WEBP", quality=88, method=6)
    # 쇼핑몰 스타일(흰 배경 + 그림자) 사본: 업로드 자동 분리 기능 테스트용
    bg = Image.new("RGB", img.size, (246, 246, 244))
    shadow = Image.fromarray(cv2.GaussianBlur(np.asarray(img)[..., 3], (0, 0), 14)).point(lambda v: int(v * 0.35))
    bg.paste((200, 200, 198), (10, 16), shadow)
    bg.paste(img, (0, 0), img)
    bg.save(out / "shop" / f"{g.id}.jpg", quality=90)
    catalog.append(
        {
            "id": g.id,
            "name": g.name,
            "category": g.category,
            "image": f"{g.id}.webp",
            "thumb": f"{g.id}_thumb.webp",
            "parts": f"{g.id}_parts.png",
            "size": list(g.size),
            "keypoints": {k: [round(v[0], 1), round(v[1], 1)] for k, v in g.keypoints.items()},
            "sleeve": g.meta.get("sleeve"),
            "credit": credit,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/public/products")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "shop").mkdir(parents=True, exist_ok=True)
    credit = "IRIS 자체 제작 이미지 (CC0)"
    catalog: list[dict] = []

    items = [
        make_tee("tee-white", "화이트 반팔 티셔츠", (236, 235, 230), seed=1),
        make_tee("tee-navy-logo", "네이비 로고 반팔 티셔츠", (34, 48, 86), decorate=chest_logo("IRIS", (240, 236, 226), (236, 170, 60), y=250, size=96), seed=2),
        make_tee("tee-stripe", "블루 스트라이프 반팔 티셔츠", (250, 250, 246), pattern=stripes((250, 250, 246), (44, 84, 160), period=58), seed=3),
        make_tee("tee-red", "레드 반팔 티셔츠", (178, 36, 44), seed=4),
        make_tee("longsleeve-green", "그린 긴팔 티셔츠", (52, 104, 76), sleeve="long", seed=5),
    ]
    for g, img in items:
        save(g, img, out, catalog, credit)
    (out / "catalog.json").write_text(json.dumps({"version": 1, "products": catalog}, ensure_ascii=False, indent=2))
    print(f"wrote {len(catalog)} products to {out}")


if __name__ == "__main__":
    main()
