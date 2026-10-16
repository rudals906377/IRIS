// 사진 모드 네일: 고른 사진의 손톱(실제 픽셀: 무늬·광택·그러데이션)을 내 손톱 모양에 맞춰 옮겨 그린다.
// 손톱 마스크(SAM)마다 주축(뿌리 → 끝)과 길이 방향 폭 분포를 구해, 내 손톱의 (길이 위치 u, 폭 위치 v)를
// 고른 손톱의 같은 (u, v)로 대응시킨다. 모양이 달라도(아몬드 ↔ 스퀘어) 가장자리끼리 맞는다.
// 내 손톱의 명암(빛 방향·윤기)을 조금 얹어 사진 속 조명과 어울리게 한다. 순수 계산(단위 테스트 대상).

import { rgb2lab } from './still-hair.ts';

const BINS = 24;

export interface NailFrame {
  cx: number;
  cy: number;
  /** 뿌리 → 끝 단위 벡터 */
  ax: number;
  ay: number;
  /** 주축 위 뿌리·끝 위치(중심 기준 투영) */
  t0: number;
  t1: number;
  /** 길이 구간별 반폭(px) */
  prof: Float32Array;
  area: number;
}

/**
 * 마스크 → 손톱 틀. hint: 손가락 방향(뿌리 → 끝) 단위 벡터를 알면 그쪽을 끝으로.
 * 모르면 rgba 로 판단: 끝 너머가 손가락 피부색에 가까운 쪽이 뿌리(손가락이 이어짐). 그래도 애매하면 위쪽이 끝.
 */
export function nailFrame(mask: Uint8Array, w: number, h: number, hint?: { x: number; y: number } | null, rgba?: Uint8ClampedArray): NailFrame | null {
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) { n++; sx += x; sy += y; }
  if (n < 20) return null;
  const cx = sx / n, cy = sy / n;
  let xx = 0, yy = 0, xy = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const dx = x - cx, dy = y - cy;
    xx += dx * dx; yy += dy * dy; xy += dx * dy;
  }
  const th = 0.5 * Math.atan2(2 * xy, xx - yy);
  let ax = Math.cos(th), ay = Math.sin(th);
  let t0 = Infinity, t1 = -Infinity;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const t = (x - cx) * ax + (y - cy) * ay;
    if (t < t0) t0 = t;
    if (t > t1) t1 = t;
  }
  // 방향 정하기
  let flip = false;
  if (hint) flip = ax * hint.x + ay * hint.y < 0;
  else {
    const decided = rgba ? cuticleSide(mask, w, h, rgba, cx, cy, ax, ay, t0, t1) : 0;
    if (decided !== 0) flip = decided > 0; // +1: 끝(t1) 쪽이 뿌리 → 뒤집기
    else flip = ay > 0; // 애매하면 위쪽(작은 y)이 끝
  }
  if (flip) {
    ax = -ax; ay = -ay;
    const a = -t1; t1 = -t0; t0 = a;
  }
  const prof = new Float32Array(BINS);
  const nx = -ay, ny = ax;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) {
    const t = (x - cx) * ax + (y - cy) * ay;
    const s = Math.abs((x - cx) * nx + (y - cy) * ny);
    const b = Math.min(BINS - 1, Math.max(0, Math.floor(((t - t0) / Math.max(1, t1 - t0)) * BINS)));
    if (s > prof[b]) prof[b] = s;
  }
  // 빈 구간은 이웃 값으로
  for (let b = 0; b < BINS; b++) if (prof[b] < 0.5) prof[b] = Math.max(b > 0 ? prof[b - 1] : 0, 0.5);
  return { cx, cy, ax, ay, t0, t1, prof, area: n };
}

/** 끝 너머의 색이 손가락 피부(마스크 둘레)에 가까운 쪽: +1 이면 t1 쪽이 뿌리, −1 이면 t0 쪽, 0 애매 */
function cuticleSide(mask: Uint8Array, w: number, h: number, rgba: Uint8ClampedArray, cx: number, cy: number, ax: number, ay: number, t0: number, t1: number): number {
  const L = t1 - t0;
  const sample = (px: number, py: number, r: number): [number, number, number] | null => {
    let n = 0, R = 0, G = 0, B = 0;
    for (let y = Math.round(py - r); y <= Math.round(py + r); y++) for (let x = Math.round(px - r); x <= Math.round(px + r); x++) {
      if (x < 0 || y < 0 || x >= w || y >= h || mask[y * w + x]) continue;
      const o = (y * w + x) * 4;
      R += rgba[o]; G += rgba[o + 1]; B += rgba[o + 2]; n++;
    }
    return n > 3 ? [R / n, G / n, B / n] : null;
  };
  // 손가락 피부색: 손톱 옆(주축에 수직으로 폭 바깥) 두 곳
  const nx = -ay, ny = ax;
  let hw = 0;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[y * w + x]) hw = Math.max(hw, Math.abs((x - cx) * nx + (y - cy) * ny));
  const r = Math.max(2, L * 0.08);
  const s1 = sample(cx + nx * (hw + r * 1.5), cy + ny * (hw + r * 1.5), r);
  const s2 = sample(cx - nx * (hw + r * 1.5), cy - ny * (hw + r * 1.5), r);
  const skins = [s1, s2].filter(Boolean) as [number, number, number][];
  if (!skins.length) return 0;
  const sk = skins.reduce((a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]], [0, 0, 0]).map((v) => v / skins.length);
  const d = (c: [number, number, number] | null): number => (c ? Math.hypot(c[0] - sk[0], c[1] - sk[1], c[2] - sk[2]) : 999);
  const e1 = sample(cx + ax * (t1 + L * 0.2), cy + ay * (t1 + L * 0.2), r);
  const e0 = sample(cx + ax * (t0 - L * 0.2), cy + ay * (t0 - L * 0.2), r);
  const d1 = d(e1), d0 = d(e0);
  if (Math.abs(d1 - d0) < 12) return 0;
  return d1 < d0 ? 1 : -1;
}

const hwAt = (f: NailFrame, u: number): number => {
  const p = Math.max(0, Math.min(BINS - 1.001, u * BINS - 0.5));
  const i = Math.floor(p);
  return f.prof[i] * (1 - (p - i)) + f.prof[Math.min(BINS - 1, i + 1)] * (p - i);
};

/**
 * dst(내 사진, RGBA) 안의 내 손톱(dMask·dF)을 src(고른 사진)의 손톱(sF)으로 칠한다(제자리 수정).
 * shadeK: 내 손톱 명암을 얹는 정도(0 이면 고른 손톱 그대로).
 */
export function transferNail(dst: Uint8ClampedArray, W: number, H: number, dMask: Uint8Array, dF: NailFrame, src: Uint8ClampedArray, sw: number, sh: number, sF: NailFrame, shadeK = 0.35, amount = 1): void {
  const dnx = -dF.ay, dny = dF.ax;
  const snx = -sF.ay, sny = sF.ax;
  // 경계 부드럽게: 손톱 폭에 맞춘 반지름(폭의 6%, 1~4px)으로 평균한 알파. 칼로 오린 듯한 테두리가 아니라 실제 손톱처럼 살짝 녹게
  const fr = Math.max(1, Math.min(4, Math.round(Math.max(...dF.prof) * 2 * 0.06)));
  const alphaAt = (x: number, y: number): number => {
    let s = 0, n = 0;
    for (let j = -fr; j <= fr; j++) for (let i = -fr; i <= fr; i++) {
      const xx = Math.min(W - 1, Math.max(0, x + i)), yy = Math.min(H - 1, Math.max(0, y + j));
      s += dMask[yy * W + xx];
      n++;
    }
    return s / n;
  };
  // 내 손톱 평균 밝기(명암 비율 기준)
  let lsum = 0, ln = 0;
  for (let i = 0; i < W * H; i++) if (dMask[i]) { const o = i * 4; lsum += 0.2126 * dst[o] + 0.7152 * dst[o + 1] + 0.0722 * dst[o + 2]; ln++; }
  const lmean = Math.max(1, lsum / Math.max(1, ln));
  // 범위: 틀 상자
  const L = dF.t1 - dF.t0;
  const R = Math.max(...dF.prof) + 2;
  const xs = [dF.t0, dF.t1].flatMap((t) => [-R, R].map((s) => dF.cx + dF.ax * t + dnx * s));
  const ys = [dF.t0, dF.t1].flatMap((t) => [-R, R].map((s) => dF.cy + dF.ay * t + dny * s));
  const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 2 - fr), x1 = Math.min(W - 1, Math.ceil(Math.max(...xs)) + 2 + fr);
  const y0 = Math.max(0, Math.floor(Math.min(...ys)) - 2 - fr), y1 = Math.min(H - 1, Math.ceil(Math.max(...ys)) + 2 + fr);
  const sL = sF.t1 - sF.t0;
  const samp = (px: number, py: number, out: number[]): void => {
    const x = Math.max(0, Math.min(sw - 1.001, px)), y = Math.max(0, Math.min(sh - 1.001, py));
    const xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
    for (let c = 0; c < 3; c++) {
      const a = src[(yi * sw + xi) * 4 + c], b = src[(yi * sw + xi + 1) * 4 + c];
      const d = src[((yi + 1) * sw + xi) * 4 + c], e = src[((yi + 1) * sw + xi + 1) * 4 + c];
      out[c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
    }
  };
  const col = [0, 0, 0];
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const al = alphaAt(x, y) * amount;
      if (al <= 0.01) continue;
      const dx = x - dF.cx, dy = y - dF.cy;
      const t = dx * dF.ax + dy * dF.ay;
      const s = dx * dnx + dy * dny;
      const u = Math.max(0, Math.min(1, (t - dF.t0) / Math.max(1, L)));
      const v = Math.max(-1, Math.min(1, s / Math.max(0.5, hwAt(dF, u))));
      // 고른 손톱의 같은 자리(가장자리는 살짝 안쪽: 손톱 테두리의 피부가 묻어나지 않게)
      const ts = sF.t0 + (0.03 + 0.94 * u) * sL;
      const ss = v * 0.9 * hwAt(sF, u);
      samp(sF.cx + sF.ax * ts + snx * ss, sF.cy + sF.ay * ts + sny * ss, col);
      const o = (y * W + x) * 4;
      const Ld = 0.2126 * dst[o] + 0.7152 * dst[o + 1] + 0.0722 * dst[o + 2];
      const shade = Math.pow(Math.max(0.2, Ld / lmean), shadeK);
      for (let c = 0; c < 3; c++) dst[o + c] = dst[o + c] + (Math.min(255, col[c] * shade) - dst[o + c]) * al;
    }
  }
}

/** 손톱은 볼록하다: 마스크를 볼록 껍질로 채워 반사광 때문에 빠진 구멍·들쭉날쭉한 가장자리를 메운다(제자리 수정) */
export function hullFill(mask: Uint8Array, w: number, h: number): Uint8Array {
  // 행마다 가장 왼쪽·오른쪽 점만으로 껍질을 구해도 충분
  const pts: [number, number][] = [];
  for (let y = 0; y < h; y++) {
    let l = -1, r = -1;
    for (let x = 0; x < w; x++) if (mask[y * w + x]) { if (l < 0) l = x; r = x; }
    if (l >= 0) { pts.push([l, y]); if (r !== l) pts.push([r, y]); }
  }
  if (pts.length < 3) return mask;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: number[], a: number[], b: number[]): number => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: number[][] = [], upper: number[][] = [];
  for (const p of pts) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop(); lower.push(p); }
  for (let i = pts.length - 1; i >= 0; i--) { const p = pts[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop(); upper.push(p); }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  // 다각형 채우기(행마다 교차점)
  let y0 = h, y1 = 0;
  for (const p of hull) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  for (let y = y0; y <= y1; y++) {
    const xs: number[] = [];
    for (let i = 0; i < hull.length; i++) {
      const a = hull[i], b = hull[(i + 1) % hull.length];
      if ((a[1] <= y && b[1] > y) || (b[1] <= y && a[1] > y)) xs.push(a[0] + ((y - a[1]) * (b[0] - a[0])) / (b[1] - a[1]));
      else if (a[1] === y && b[1] === y) xs.push(a[0], b[0]);
    }
    if (xs.length < 2) continue;
    const lo = Math.max(0, Math.ceil(Math.min(...xs))), hi = Math.min(w - 1, Math.floor(Math.max(...xs)));
    for (let x = lo; x <= hi; x++) mask[y * w + x] = 1;
  }
  return mask;
}

/** 마스크 안과 바로 바깥 띠(두께 r)의 평균색 차이(Lab ΔE 근사). 손톱이 아닌 손가락 피부를 잡으면 작다 */
export function maskContrast(rgba: Uint8ClampedArray, w: number, h: number, mask: Uint8Array, r = 3): number {
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { const x = i % w, y = (i - x) / w; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < x0) return 0;
  const lab = new Float32Array(3);
  const inS = [0, 0, 0], outS = [0, 0, 0];
  let nIn = 0, nOut = 0;
  for (let y = Math.max(0, y0 - r); y <= Math.min(h - 1, y1 + r); y++) for (let x = Math.max(0, x0 - r); x <= Math.min(w - 1, x1 + r); x++) {
    const i = y * w + x;
    let ring = false;
    if (!mask[i]) {
      for (let j = -r; j <= r && !ring; j += r) for (let k = -r; k <= r; k += r) {
        const xx = x + k, yy = y + j;
        if (xx >= 0 && yy >= 0 && xx < w && yy < h && mask[yy * w + xx]) { ring = true; break; }
      }
      if (!ring) continue;
    }
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    const t = mask[i] ? inS : outS;
    t[0] += lab[0]; t[1] += lab[1]; t[2] += lab[2];
    if (mask[i]) nIn++; else nOut++;
  }
  if (!nIn || !nOut) return 0;
  return Math.hypot(inS[0] / nIn - outS[0] / nOut, inS[1] / nIn - outS[1] / nOut, inS[2] / nIn - outS[2] / nOut);
}

/**
 * 색만 아는 네일(고른 사진에서 손톱 모양을 못 찾음)용 가상 손톱 그림: 끝이 둥근 손톱 모양에 그 색을 칠하고,
 * 길이 방향으로 살짝 밝아지는 그러데이션과 가운데 세로 광택 띠를 넣는다. 위가 끝(손끝), 아래가 뿌리.
 */
export function syntheticNail(color: [number, number, number], gloss = 0.5): { rgba: Uint8ClampedArray; w: number; h: number; mask: Uint8Array; frame: NailFrame } {
  const w = 64, h = 112;
  const rgba = new Uint8ClampedArray(w * h * 4), mask = new Uint8Array(w * h);
  const cx = w / 2, hw = w * 0.42, top = h * 0.06, bot = h * 0.94;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    // 끝(위)은 반원, 뿌리(아래)는 완만한 곡선
    const dy = y < top + hw ? (top + hw - y) / hw : 0;
    const halfW = dy > 0 ? hw * Math.sqrt(Math.max(0, 1 - dy * dy)) : hw * (1 - Math.max(0, (y - (bot - hw * 0.5)) / (hw * 0.5)) ** 2 * 0.35);
    const inside = y >= top && y <= bot && Math.abs(x - cx) <= halfW;
    const i = y * w + x;
    if (!inside) continue;
    mask[i] = 1;
    const u = (y - top) / (bot - top); // 0 끝 → 1 뿌리
    const s = (x - cx) / Math.max(1, halfW); // -1 ~ 1
    // 둥근 손톱 면: 가장자리로 갈수록 조금 어둡게, 끝 쪽은 조금 밝게, 가운데 왼쪽에 세로 광택
    const shade = 1 - 0.18 * s * s + 0.06 * (1 - u);
    const hl = gloss * Math.exp(-(((s + 0.25) / 0.12) ** 2)) * (0.3 + 0.7 * Math.sin(Math.PI * Math.min(1, u * 1.1)));
    for (let c = 0; c < 3; c++) rgba[i * 4 + c] = Math.min(255, color[c] * shade + (255 - color[c] * shade) * hl * 0.8);
    rgba[i * 4 + 3] = 255;
  }
  const frame = nailFrame(mask, w, h, { x: 0, y: -1 })!;
  return { rgba, w, h, mask, frame };
}
