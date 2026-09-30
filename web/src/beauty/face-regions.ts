// 얼굴 점(MediaPipe Face Landmarker 478점) → 메이크업 영역 다각형.
// 좌표는 영상 픽셀. L/R은 **사람 기준**(정면 영상에서 사람 왼쪽 = 이미지 오른쪽).
// 점 번호는 MediaPipe 얼굴 메쉬 표준 번호다.

import type { Vec2 } from '../engine/math.ts';

/** 입술 바깥 윤곽(시계 방향) */
export const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
/** 입술 안쪽(입 벌린 곳) 윤곽 */
export const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];

/** 사람 오른쪽 눈(이미지 왼쪽): 윗눈꺼풀(바깥 → 안쪽), 아랫눈꺼풀(안쪽 → 바깥) */
export const EYE_R_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133];
export const EYE_R_LOWER = [133, 155, 154, 153, 145, 144, 163, 7, 33];
/** 사람 왼쪽 눈(이미지 오른쪽) */
export const EYE_L_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362];
export const EYE_L_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263];

/** 눈썹 아랫선(바깥 → 안쪽)과 윗선(안쪽 → 바깥) */
export const BROW_R_LOWER = [46, 53, 52, 65, 55];
export const BROW_R_UPPER = [107, 66, 105, 63, 70];
export const BROW_L_LOWER = [276, 283, 282, 295, 285];
export const BROW_L_UPPER = [336, 296, 334, 293, 300];

/** 얼굴 폭 기준점(양쪽 광대 옆) */
export const FACE_SIDE_R = 234;
export const FACE_SIDE_L = 454;
/** 볼 중심(광대 아래) */
export const CHEEK_R = 205;
export const CHEEK_L = 425;
/** 광대 바깥쪽(블러셔가 관자놀이 쪽으로 퍼지는 방향) */
export const CHEEKBONE_R = 116;
export const CHEEKBONE_L = 345;

export interface FaceRegions {
  /** 얼굴 폭(px) — 경계 흐림 크기 등의 기준 */
  faceW: number;
  lipsOuter: Vec2[];
  lipsInner: Vec2[];
  /** 아이섀도: 윗눈꺼풀 선 + 눈썹 쪽으로 올린 선으로 닫은 다각형 */
  shadowR: Vec2[];
  shadowL: Vec2[];
  /** 아이라이너: 윗눈꺼풀 선(바깥 끝은 꼬리를 살짝 뺌) */
  linerR: Vec2[];
  linerL: Vec2[];
  /** 눈을 뜬 부분(흰자·눈동자): 아이섀도가 번져 들어가지 않게 빼는 영역 */
  eyeR: Vec2[];
  eyeL: Vec2[];
  browR: Vec2[];
  browL: Vec2[];
  /** 블러셔: 타원(중심, 가로·세로 반지름, 기울기 방향 단위 벡터) */
  blushR: { c: Vec2; rx: number; ry: number; dir: Vec2 };
  blushL: { c: Vec2; rx: number; ry: number; dir: Vec2 };
}

const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** 눈꺼풀 선(바깥 → 안쪽)과 눈썹 아랫선을 이용해 아이섀도 영역을 만든다. height 0~1: 눈썹까지 얼마나 올릴지 */
function shadowPoly(p: Vec2[], upper: number[], browLower: number[], height: number): Vec2[] {
  const lid = upper.map((i) => p[i]);
  // 눈썹 아랫선을 눈꺼풀 점 수에 맞춰 다시 표본(바깥 → 안쪽)
  const brow = browLower.map((i) => p[i]);
  const top: Vec2[] = lid.map((_, k) => {
    const t = k / (lid.length - 1);
    const f = t * (brow.length - 1);
    const i = Math.min(brow.length - 2, Math.floor(f));
    const b = lerp(brow[i], brow[i + 1], f - i);
    return lerp(lid[k], b, height);
  });
  return [...lid, ...top.reverse()];
}

/** 눈꼬리 쪽으로 살짝 빠지는 아이라이너 선 */
function linerLine(p: Vec2[], upper: number[], lower: number[], wing: number): Vec2[] {
  const line = upper.map((i) => p[i]);
  const outer = line[0];
  // 꼬리 방향: 아랫눈꺼풀 바깥쪽에서 바깥 끝으로 향하는 방향을 조금 위로
  const lowOuter = p[lower[lower.length - 2]];
  const dx = outer.x - lowOuter.x;
  const dy = outer.y - lowOuter.y;
  const l = Math.hypot(dx, dy) || 1;
  const eyeW = dist(p[upper[0]], p[upper[upper.length - 1]]);
  const tip = { x: outer.x + (dx / l) * eyeW * wing, y: outer.y + (dy / l) * eyeW * wing - eyeW * wing * 0.35 };
  return wing > 0 ? [tip, ...line] : line;
}

/** 478점(픽셀 좌표)에서 메이크업 영역을 만든다. */
export function faceRegions(p: Vec2[], opts: { shadowHeight?: number; linerWing?: number } = {}): FaceRegions {
  const faceW = dist(p[FACE_SIDE_R], p[FACE_SIDE_L]);
  const sh = opts.shadowHeight ?? 0.62;
  const wing = opts.linerWing ?? 0.25;
  const blush = (cheek: number, bone: number): FaceRegions['blushR'] => {
    const c0 = p[cheek];
    const b = p[bone];
    // 볼 중심과 광대 바깥의 사이, 광대 쪽으로 조금 올린 위치
    const c = lerp(c0, b, 0.35);
    const dx = b.x - c0.x;
    const dy = b.y - c0.y;
    const l = Math.hypot(dx, dy) || 1;
    return { c, rx: faceW * 0.14, ry: faceW * 0.09, dir: { x: dx / l, y: dy / l } };
  };
  return {
    faceW,
    lipsOuter: LIPS_OUTER.map((i) => p[i]),
    lipsInner: LIPS_INNER.map((i) => p[i]),
    shadowR: shadowPoly(p, EYE_R_UPPER, BROW_R_LOWER, sh),
    shadowL: shadowPoly(p, EYE_L_UPPER, BROW_L_LOWER, sh),
    linerR: linerLine(p, EYE_R_UPPER, EYE_R_LOWER, wing),
    linerL: linerLine(p, EYE_L_UPPER, EYE_L_LOWER, wing),
    eyeR: [...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)].map((i) => p[i]),
    eyeL: [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)].map((i) => p[i]),
    browR: [...BROW_R_LOWER.map((i) => p[i]), ...BROW_R_UPPER.map((i) => p[i])],
    browL: [...BROW_L_LOWER.map((i) => p[i]), ...BROW_L_UPPER.map((i) => p[i])],
    blushR: blush(CHEEK_R, CHEEKBONE_R),
    blushL: blush(CHEEK_L, CHEEKBONE_L),
  };
}

/**
 * 단순 다각형을 삼각형으로 나눈다(귀 자르기). 결과는 점 번호 3개씩.
 * 얼굴 영역처럼 점이 수십 개인 다각형용(매 프레임 계산해도 가볍다).
 */
export function triangulate(poly: Vec2[]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  let area = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    area += a.x * b.y - b.x * a.y;
  }
  const ccw = area > 0;
  const idx = Array.from({ length: n }, (_, i) => i);
  const out: number[] = [];
  const cross = (a: Vec2, b: Vec2, c: Vec2): number => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  const inside = (p: Vec2, a: Vec2, b: Vec2, c: Vec2): boolean => {
    const d1 = cross(a, b, p);
    const d2 = cross(b, c, p);
    const d3 = cross(c, a, p);
    return ccw ? d1 >= 0 && d2 >= 0 && d3 >= 0 : d1 <= 0 && d2 <= 0 && d3 <= 0;
  };
  let guard = 0;
  while (idx.length > 3 && guard++ < n * n) {
    let clipped = false;
    for (let k = 0; k < idx.length; k++) {
      const i0 = idx[(k + idx.length - 1) % idx.length];
      const i1 = idx[k];
      const i2 = idx[(k + 1) % idx.length];
      const a = poly[i0];
      const b = poly[i1];
      const c = poly[i2];
      const cr = cross(a, b, c);
      if (ccw ? cr <= 0 : cr >= 0) continue; // 오목한 꼭짓점
      let ok = true;
      for (const j of idx) {
        if (j === i0 || j === i1 || j === i2) continue;
        if (inside(poly[j], a, b, c)) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      out.push(i0, i1, i2);
      idx.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) break; // 자기 교차 등: 남은 부분은 부채꼴로
  }
  for (let k = 1; k + 1 < idx.length; k++) out.push(idx[0], idx[k], idx[k + 1]);
  return out;
}

/** 두께 있는 선(폴리라인)을 삼각형 띠로. 끝으로 갈수록 가늘어지게(taper: 0~1). */
export function strokeStrip(line: Vec2[], width: number, taper = 0.7): Vec2[] {
  const out: Vec2[] = [];
  for (let i = 0; i < line.length; i++) {
    const a = line[Math.max(0, i - 1)];
    const b = line[Math.min(line.length - 1, i + 1)];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const l = Math.hypot(dx, dy) || 1;
    const nx = -dy / l;
    const ny = dx / l;
    const t = i / Math.max(1, line.length - 1);
    // 가운데가 두껍고 양 끝이 가늘다(바깥 끝 = 앞쪽이 더 두꺼운 날개)
    const w = (width / 2) * (1 - taper * Math.pow(Math.abs(t - 0.35) / 0.65, 2));
    out.push({ x: line[i].x + nx * w, y: line[i].y + ny * w }, { x: line[i].x - nx * w, y: line[i].y - ny * w });
  }
  return out;
}
