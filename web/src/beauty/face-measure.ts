// 얼굴 색 측정: 얼굴 점 478개와 그림(ImageData)에서 부위별 색을 잰다.
// 참고 사진과 합성한 내 얼굴을 **같은 방법**으로 재서 비교하는 데 쓴다(사진 따라하기의 되먹임).
// 조명·앞머리 그늘의 영향을 줄이려고 부위마다 바로 옆 피부(같은 쪽)를 기준으로 비율을 낸다.
// 색은 선형 RGB(0~1).

import type { Vec2 } from '../engine/math.ts';
import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_INNER, LIPS_OUTER } from './face-regions.ts';
import type { RGB } from './makeup.ts';
import { isSkin, luma } from './style-math.ts';

/** 얼굴 윤곽(MediaPipe 얼굴 메쉬 표준 번호, 시계 방향) */
export const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

/**
 * 피부 밝기 흐름: 얼굴 피부 표본(이목구비·화장 부위 제외)에 채널별 2차식(x, y)을 맞춘다.
 * 조명이 한쪽에서 오거나 앞머리 그늘이 있어도 "그 자리의 예상 피부색"을 알 수 있어, 부위색을 이것과 비교한다.
 */
export interface SkinField {
  at(q: Vec2): RGB;
  /** 표본 평균(선형) */
  mean: RGB;
  /** 표본 수 */
  n: number;
}

export function fitSkinField(img: ImageData, p: Vec2[], exclude: Vec2[][], skinOk?: (q: Vec2) => boolean): SkinField | null {
  const fw = Math.hypot(p[234].x - p[454].x, p[234].y - p[454].y);
  const cx = (p[234].x + p[454].x) / 2;
  const cy = (p[10].y + p[152].y) / 2;
  const S = new Sampler(img);
  // 얼굴 윤곽 안, 제외 영역 밖의 격자 표본
  const oval = FACE_OVAL.map((i) => p[i]);
  const step = Math.max(2, fw / 36);
  const pts: { u: number; v: number; c: RGB }[] = [];
  const inPoly = (q: Vec2, poly: Vec2[]): boolean => {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i];
      const b = poly[j];
      if (a.y > q.y !== b.y > q.y && q.x < ((b.x - a.x) * (q.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
    }
    return inside;
  };
  const { width: w, height: h, data: d } = img;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const q of oval) {
    x0 = Math.min(x0, q.x);
    y0 = Math.min(y0, q.y);
    x1 = Math.max(x1, q.x);
    y1 = Math.max(y1, q.y);
  }
  for (let y = y0; y <= y1; y += step)
    for (let x = x0; x <= x1; x += step) {
      const q = { x, y };
      if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) continue;
      if (!inPoly(q, oval)) continue;
      if (exclude.some((poly) => inPoly(q, poly))) continue;
      if (skinOk && !skinOk(q)) continue;
      // 3×3 평균
      const c: RGB = [0, 0, 0];
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const k = ((Math.round(y) + dy) * w + Math.round(x) + dx) * 4;
          c[0] += lin1(d[k]);
          c[1] += lin1(d[k + 1]);
          c[2] += lin1(d[k + 2]);
        }
      const cc = c.map((v) => v / 9) as RGB;
      if (!isSkin(cc)) continue;
      pts.push({ u: (x - cx) / fw, v: (y - cy) / fw, c: cc });
    }
  void S;
  if (pts.length < 30) return null;
  // 2차식 계수: [1, u, v, u², uv, v²]
  const basis = (u: number, v: number): number[] => [1, u, v, u * u, u * v, v * v];
  const solve = (sel: typeof pts): number[][] | null => {
    const coef: number[][] = [];
    for (let ch = 0; ch < 3; ch++) {
      const A = Array.from({ length: 6 }, () => new Array<number>(7).fill(0));
      for (const q of sel) {
        const b = basis(q.u, q.v);
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 6; j++) A[i][j] += b[i] * b[j];
          A[i][6] += b[i] * q.c[ch];
        }
      }
      // 가우스 소거
      for (let i = 0; i < 6; i++) {
        let piv = i;
        for (let r = i + 1; r < 6; r++) if (Math.abs(A[r][i]) > Math.abs(A[piv][i])) piv = r;
        [A[i], A[piv]] = [A[piv], A[i]];
        if (Math.abs(A[i][i]) < 1e-12) return null;
        for (let r = 0; r < 6; r++) {
          if (r === i) continue;
          const f = A[r][i] / A[i][i];
          for (let c = i; c <= 6; c++) A[r][c] -= f * A[i][c];
        }
      }
      coef.push(A.map((row, i) => row[6] / row[i]));
    }
    return coef;
  };
  let sel = pts;
  let coef = solve(sel);
  if (!coef) return null;
  // 튀는 표본(잔머리·점·반사광)을 빼고 한 번 더
  for (let pass = 0; pass < 2; pass++) {
    const cf = coef;
    const keep = sel.filter((q) => {
      const b = basis(q.u, q.v);
      const pred = cf.map((c) => b.reduce((s, x, i) => s + x * c[i], 0));
      const err = Math.abs(luma(q.c) / Math.max(luma(pred as RGB), 1e-4) - 1);
      return err < 0.22;
    });
    if (keep.length < 30 || keep.length === sel.length) break;
    const next = solve(keep);
    if (!next) break;
    sel = keep;
    coef = next;
  }
  const cf = coef;
  const mean: RGB = [0, 0, 0];
  for (const q of sel) for (let i = 0; i < 3; i++) mean[i] += q.c[i] / sel.length;
  return {
    mean,
    n: sel.length,
    at(q: Vec2): RGB {
      const b = basis((q.x - cx) / fw, (q.y - cy) / fw);
      return cf.map((c) => Math.max(1e-3, b.reduce((s, x, i) => s + x * c[i], 0))) as RGB;
    },
  };
}

export interface FaceMeasure {
  faceW: number;
  /** 기준 피부(여러 곳 중앙값, 선형) */
  skin: RGB;
  /** 립: 평균색, 밝기 중간값·90%값, 가장자리/가운데 채도 비율 */
  lip: RGB | null;
  lipP50: number;
  lipP90: number;
  lipEdgeSat: number | null;
  /** 립과 기준 피부의 차이(0~1) */
  lipDiff: number;
  /** 블러셔: 세 위치(눈 밑 사과존, 볼 가운데, 광대)의 색 / 그 자리의 예상 피부색 */
  blushApple: RGB | null;
  blushMid: RGB | null;
  blushBone: RGB | null;
  /** 아이섀도: 윗눈꺼풀 안쪽·바깥쪽 절반의 색 / 그 자리의 예상 피부색 */
  shadowIn: RGB | null;
  shadowOut: RGB | null;
  /** 아래 속눈썹 선 아래 띠 / 눈썹 위 뼈 피부 */
  underEye: RGB | null;
  /** 속눈썹 선 바로 위 띠 밝기 / 윗눈꺼풀 밝기 */
  linerL: number | null;
  /** 눈썹 색 / 눈썹 위 뼈 피부 */
  brow: RGB | null;
  /** 윤곽: 광대 아래 밝기 / 예상 피부 밝기, 콧대 밝기 / 예상 피부 밝기 */
  hollowL: number | null;
  noseBridgeL: number | null;
  /** 피부 흐름 맞추기에 쓴 표본 수(0이면 실패) */
  skinSamples: number;
}

const lin1 = (v: number): number => Math.pow(v / 255, 2.2);
const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const circle = (c: Vec2, r: number): Vec2[] =>
  Array.from({ length: 16 }, (_, k) => ({ x: c.x + Math.cos((k / 16) * Math.PI * 2) * r, y: c.y + Math.sin((k / 16) * Math.PI * 2) * r }));
export const ratioOf = (a: RGB, b: RGB): RGB => a.map((v, i) => v / Math.max(b[i], 1e-4)) as RGB;
/** 밝기를 1로 맞춘 색조만 남긴 비율(조명 밝기 차이를 무시하고 색 기운만 비교) */
export const chroma = (r: RGB): RGB => {
  const l = Math.max(luma(r), 1e-4);
  return r.map((v) => v / l) as RGB;
};

/** 다각형(구멍 제외) 안 픽셀을 선형 RGB로 모은다 */
export class Sampler {
  private readonly img: ImageData;
  private readonly mask: HTMLCanvasElement;
  private readonly mc: CanvasRenderingContext2D;
  constructor(img: ImageData) {
    this.img = img;
    this.mask = document.createElement('canvas');
    this.mask.width = img.width;
    this.mask.height = img.height;
    this.mc = this.mask.getContext('2d', { willReadFrequently: true })!;
  }

  pixels(polys: Vec2[][], holes: Vec2[][] = []): RGB[] {
    const { width: w, height: h, data: d } = this.img;
    const mc = this.mc;
    // 다각형들의 경계 상자만 처리해 빠르게
    let x0 = w;
    let y0 = h;
    let x1 = 0;
    let y1 = 0;
    for (const p of polys)
      for (const q of p) {
        if (q.x < x0) x0 = q.x;
        if (q.y < y0) y0 = q.y;
        if (q.x > x1) x1 = q.x;
        if (q.y > y1) y1 = q.y;
      }
    x0 = Math.max(0, Math.floor(x0));
    y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(w - 1, Math.ceil(x1));
    y1 = Math.min(h - 1, Math.ceil(y1));
    if (x1 <= x0 || y1 <= y0) return [];
    mc.clearRect(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    const fill = (p: Vec2[], color: string): void => {
      mc.fillStyle = color;
      mc.beginPath();
      p.forEach((q, k) => (k ? mc.lineTo(q.x, q.y) : mc.moveTo(q.x, q.y)));
      mc.closePath();
      mc.fill();
    };
    for (const p of polys) fill(p, '#fff');
    for (const p of holes) fill(p, '#000');
    const md = mc.getImageData(x0, y0, x1 - x0 + 1, y1 - y0 + 1).data;
    const out: RGB[] = [];
    const bw = x1 - x0 + 1;
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        if (md[((y - y0) * bw + (x - x0)) * 4] < 128) continue;
        const k = (y * w + x) * 4;
        out.push([lin1(d[k]), lin1(d[k + 1]), lin1(d[k + 2])]);
      }
    return out;
  }
}

/** 밝기 위아래 10%를 뺀 평균(반사광·그늘 제외). 표본이 적으면 null */
export function trimmedMean(px: RGB[], min = 12): RGB | null {
  if (px.length < min) return null;
  const s = [...px].sort((a, b) => luma(a) - luma(b));
  const a = Math.floor(s.length * 0.1);
  const sel = s.slice(a, s.length - a);
  const out: RGB = [0, 0, 0];
  for (const q of sel) for (let c = 0; c < 3; c++) out[c] += q[c];
  return out.map((v) => v / sel.length) as RGB;
}

export function percentileL(px: RGB[], p: number): number {
  if (px.length === 0) return 0;
  const s = px.map(luma).sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * p))];
}

const sat = (c: RGB): number => (Math.max(...c) - Math.min(...c)) / Math.max(Math.max(...c), 1e-3);

/** 두 색(선형)의 차이(0~1 정도, 감마 공간 거리) */
export function diffOf(a: RGB, b: RGB): number {
  const g = (c: RGB): RGB => c.map((v) => Math.pow(Math.max(0, v), 1 / 2.2)) as RGB;
  const ga = g(a);
  const gb = g(b);
  return Math.hypot(ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]) / Math.sqrt(3);
}

/**
 * 얼굴 색을 잰다.
 * @param skinOk (선택) 화면 좌표의 점이 얼굴 피부인지(분할) — 앞머리로 덮인 이마를 기준에서 빼는 데 쓴다
 */
export function measureFace(img: ImageData, p: Vec2[], skinOk?: (q: Vec2) => boolean): FaceMeasure {
  const S = new Sampler(img);
  const fw = Math.hypot(p[234].x - p[454].x, p[234].y - p[454].y);
  // 피부 조각: 머리카락·배경·그늘이 섞이지 않게 피부색 픽셀만 쓰고, 절반 이상 남아야 인정
  const patch = (c: Vec2, r: number): RGB | null => {
    const all = S.pixels([circle(c, fw * r)]);
    const sk = all.filter(isSkin);
    if (sk.length < Math.max(12, all.length * 0.5)) return null;
    return trimmedMean(sk);
  };

  // 제외 영역: 눈(속눈썹까지), 눈썹, 입술, 그리고 측정 부위(볼 세 곳·눈꺼풀)는 피부 흐름 맞추기에서 뺀다
  const grow = (pts: Vec2[], k: number): Vec2[] => {
    const c = pts.reduce((a, q) => ({ x: a.x + q.x / pts.length, y: a.y + q.y / pts.length }), { x: 0, y: 0 });
    return pts.map((q) => ({ x: c.x + (q.x - c.x) * k, y: c.y + (q.y - c.y) * k }));
  };
  const eyeR = [...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)].map((i) => p[i]);
  const eyeL = [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)].map((i) => p[i]);
  const browR = [...BROW_R_LOWER, ...BROW_R_UPPER].map((i) => p[i]);
  const browL = [...BROW_L_LOWER, ...BROW_L_UPPER].map((i) => p[i]);
  const exclude: Vec2[][] = [
    grow(eyeR, 1.8),
    grow(eyeL, 1.8),
    grow(browR, 1.4),
    grow(browL, 1.4),
    grow(LIPS_OUTER.map((i) => p[i]), 1.2),
    // 볼 전체(넓게 바른 블러셔가 피부 흐름에 섞이지 않게)와 광대
    ...[118, 347, 205, 425].map((i) => circle(p[i], fw * 0.13)),
    ...[116, 345].map((i) => circle(p[i], fw * 0.08)),
  ];
  const field = fitSkinField(img, p, exclude, skinOk);
  const skin: RGB = field?.mean ?? patch(p[199], 0.035) ?? [0.5, 0.35, 0.28];
  /** 부위 평균색 / 그 자리의 예상 피부색 */
  const rel = (c: Vec2, r: number): RGB | null => {
    const m = patch(c, r);
    if (!m) return null;
    return ratioOf(m, field ? field.at(c) : skin);
  };

  // 립
  const outer = LIPS_OUTER.map((i) => p[i]);
  const inner = LIPS_INNER.map((i) => p[i]);
  const lipPx = S.pixels([outer], [inner]);
  const lip = trimmedMean(lipPx);
  const mid = outer.map((q, k) => lerp(q, inner[k], 0.5));
  const edge = trimmedMean(S.pixels([outer], [mid]));
  const core = trimmedMean(S.pixels([mid], [inner]));
  const lipEdgeSat = edge && core ? sat(edge) / Math.max(sat(core), 1e-3) : null;

  // 블러셔: 세 위치(눈 밑 사과존·볼 가운데·광대)의 색 / 그 자리의 예상 피부색, 양쪽 평균
  const avg = (x: RGB | null | undefined, y: RGB | null | undefined): RGB | null => (x && y ? (x.map((v, i) => (v + y[i]) / 2) as RGB) : (x ?? y ?? null));
  const blushApple = avg(rel(p[118], 0.06), rel(p[347], 0.06));
  const blushMid = avg(rel(p[205], 0.06), rel(p[425], 0.06));
  const blushBone = avg(rel(p[116], 0.05), rel(p[345], 0.05));

  // 눈: 눈썹 위 뼈(눈썹 윗선 가운데에서 이마 쪽으로 조금) 기준
  const eye = (upper: number[], lower: number[], browLower: number[], browUpper: number[]): {
    inn: RGB;
    out: RGB;
    under: RGB;
    linerL: number;
    brow: RGB;
  } | null => {
    const bu = browUpper.map((i) => p[i]);
    const bl = browLower.map((i) => p[i]);
    const lid = upper.map((i) => p[i]);
    // 기준: 눈꺼풀 가운데 자리의 예상 피부색(피부 흐름). 흐름이 없으면 눈썹 위 뼈 조각
    const eyeC = p[upper[4]];
    const browC = bu[2];
    const up = { x: browC.x - eyeC.x, y: browC.y - eyeC.y };
    const bone = field ? field.at(lerp(eyeC, browC, 0.5)) : patch({ x: browC.x + up.x * 0.35, y: browC.y + up.y * 0.35 }, 0.035);
    if (!bone) return null;
    const top = lid.map((q, k) => lerp(q, p[browLower[Math.round((k / (lid.length - 1)) * (browLower.length - 1))]], 0.5));
    // 바깥 절반(0~4)과 안쪽 절반(4~8)
    const half = (a: number, b: number): Vec2[] => [...lid.slice(a, b + 1), ...top.slice(a, b + 1).reverse()];
    const outPx = S.pixels([half(0, 4)]);
    const inPx = S.pixels([half(4, 8)]);
    const strip = [...lid, ...lid.map((q, k) => lerp(q, top[k], 0.16)).reverse()];
    const stripPx = S.pixels([strip]);
    const lidPx = S.pixels([[...lid, ...top.reverse()]], [strip]);
    const lowLine = lower.map((i) => p[i]);
    const underPoly = [...lowLine, ...lowLine.map((q) => ({ x: q.x, y: q.y + fw * 0.03 })).reverse()];
    const inn = trimmedMean(inPx);
    const out = trimmedMean(outPx);
    const under = trimmedMean(S.pixels([underPoly]));
    const stripM = trimmedMean(stripPx, 6);
    const lidM = trimmedMean(lidPx);
    const brow = trimmedMean(S.pixels([[...bl, ...bu]]));
    if (!inn || !out || !under || !stripM || !lidM || !brow) return null;
    return { inn: ratioOf(inn, bone), out: ratioOf(out, bone), under: ratioOf(under, bone), linerL: luma(stripM) / Math.max(luma(lidM), 1e-4), brow: ratioOf(brow, bone) };
  };
  const eR = eye(EYE_R_UPPER, EYE_R_LOWER, BROW_R_LOWER, BROW_R_UPPER);
  const eL = eye(EYE_L_UPPER, EYE_L_LOWER, BROW_L_LOWER, BROW_L_UPPER);
  const pick = <K extends 'inn' | 'out' | 'under' | 'brow'>(k: K): RGB | null => avg(eR?.[k], eL?.[k]);
  const linerL = eR && eL ? (eR.linerL + eL.linerL) / 2 : (eR?.linerL ?? eL?.linerL ?? null);

  // 윤곽: 광대 아래(147)·콧대(197)의 색 / 예상 피부색
  const hollow = avg(rel(p[147], 0.035), rel(p[376], 0.035));
  const bridge = rel(p[197], 0.02);

  return {
    faceW: fw,
    skin,
    lip,
    lipP50: percentileL(lipPx, 0.5),
    lipP90: percentileL(lipPx, 0.9),
    lipEdgeSat,
    lipDiff: lip ? diffOf(lip, skin) : 0,
    blushApple,
    blushMid,
    blushBone,
    shadowIn: pick('inn'),
    shadowOut: pick('out'),
    underEye: pick('under'),
    linerL,
    brow: pick('brow'),
    hollowL: hollow ? luma(hollow) : null,
    noseBridgeL: bridge ? luma(bridge) : null,
    skinSamples: field?.n ?? 0,
  };
}
