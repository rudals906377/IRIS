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
  // 범위: 틀 상자
  const L = dF.t1 - dF.t0;
  const R = Math.max(...dF.prof) + 2;
  const xs = [dF.t0, dF.t1].flatMap((t) => [-R, R].map((s) => dF.cx + dF.ax * t + dnx * s));
  const ys = [dF.t0, dF.t1].flatMap((t) => [-R, R].map((s) => dF.cy + dF.ay * t + dny * s));
  const x0 = Math.max(0, Math.floor(Math.min(...xs)) - 2 - fr), x1 = Math.min(W - 1, Math.ceil(Math.max(...xs)) + 2 + fr);
  const y0 = Math.max(0, Math.floor(Math.min(...ys)) - 2 - fr), y1 = Math.min(H - 1, Math.ceil(Math.max(...ys)) + 2 + fr);
  const shadeL = nailShading(dst, W, dMask, x0, y0, x1, y1, Math.max(2, Math.round(Math.max(...dF.prof) * 0.5)));
  const bw = x1 - x0 + 1;
  // 내 손톱 평균 밝기(명암 비율 기준)
  let lsum = 0, ln = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (dMask[y * W + x]) { lsum += shadeL[(y - y0) * bw + x - x0]; ln++; }
  const lmean = Math.max(1, lsum / Math.max(1, ln));
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
      // 녹이는 테두리는 마스크 바깥쪽으로: 안쪽 가장자리까지 꽉 덮어야 원래 매니큐어 색이 테두리로 비치지 않는다
      const al = Math.min(1, alphaAt(x, y) * 2) * amount;
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
      const Ld = shadeL[(y - y0) * bw + x - x0];
      const shade = Math.pow(Math.max(0.2, Ld / lmean), shadeK);
      for (let c = 0; c < 3; c++) dst[o + c] = dst[o + c] + (Math.min(255, col[c] * shade) - dst[o + c]) * al;
    }
  }
}

/**
 * 손톱 명암(빛·윤기)으로 쓸 밝기. 손톱의 주된 색(마스크 Lab 가운데 값)과 색이 다르거나 L 12 넘게 어두운 픽셀(드립·프렌치·스톤 같은 원래 무늬)은
 * 조명이 아니라 칠한 색이므로, 둘레 반지름 [r] 안 주된 색 픽셀 밝기의 평균으로 바꾼다(a·b 거리 8~16 사이 섞음).
 * (흰 바탕에 로즈골드 드립 손톱이면 드립 밝기가 평균의 0.3~0.5 라 새 네일에 원래 드립 무늬가 어둡게 비쳤다)
 * 반환: 상자 [x0..x1]×[y0..y1] 크기의 밝기 배열.
 */
export function nailShading(rgba: Uint8ClampedArray, W: number, mask: Uint8Array, x0: number, y0: number, x1: number, y1: number, r: number): Float32Array {
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const lum = new Float32Array(bw * bh), wb = new Float32Array(bw * bh);
  const lv: number[] = [], av: number[] = [], bv: number[] = [];
  const lab = new Float32Array(3);
  const Lc = new Float32Array(bw * bh), A = new Float32Array(bw * bh), B = new Float32Array(bw * bh);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const o = (y * W + x) * 4, k = (y - y0) * bw + x - x0;
    lum[k] = 0.2126 * rgba[o] + 0.7152 * rgba[o + 1] + 0.0722 * rgba[o + 2];
    if (!mask[y * W + x]) continue;
    rgb2lab(rgba[o], rgba[o + 1], rgba[o + 2], lab);
    Lc[k] = lab[0]; A[k] = lab[1]; B[k] = lab[2];
    lv.push(lab[0]); av.push(lab[1]); bv.push(lab[2]);
  }
  if (av.length < 20) return lum;
  const med = (v: number[]) => v.sort((p, q) => p - q)[v.length >> 1];
  const ml = med(lv), ma = med(av), mb = med(bv);
  let other = 0;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const k = (y - y0) * bw + x - x0;
    if (!mask[y * W + x]) continue;
    // 색 차이 + 주된 밝기보다 L 12 넘게 어두운 만큼(어두운 크롬·검은 선은 채도가 낮아 색만으로는 못 가림). 밝은 쪽(광택)은 그대로
    const d = Math.hypot(A[k] - ma, B[k] - mb, Math.max(0, ml - Lc[k] - 12));
    wb[k] = 1 - Math.min(1, Math.max(0, (d - 8) / 8));
    if (wb[k] < 0.5) other++;
  }
  if (other < av.length * 0.03) return lum;
  // 주된 색 픽셀만의 둘레 평균 밝기(적분 영상 두 장)
  const S = new Float64Array((bw + 1) * (bh + 1)), N = new Float64Array((bw + 1) * (bh + 1));
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const k = y * bw + x, j = (y + 1) * (bw + 1) + x + 1;
    S[j] = lum[k] * wb[k] + S[j - 1] + S[j - bw - 1] - S[j - bw - 2];
    N[j] = wb[k] + N[j - 1] + N[j - bw - 1] - N[j - bw - 2];
  }
  let gs = 0, gn = 0;
  for (let k = 0; k < lum.length; k++) { gs += lum[k] * wb[k]; gn += wb[k]; }
  const g = gs / Math.max(1e-6, gn);
  const out = new Float32Array(lum);
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const k = y * bw + x;
    if (!mask[(y + y0) * W + x + x0] || wb[k] >= 1) continue;
    const xa = Math.max(0, x - r), xb = Math.min(bw, x + r + 1), ya = Math.max(0, y - r), yb = Math.min(bh, y + r + 1);
    const n = N[yb * (bw + 1) + xb] - N[ya * (bw + 1) + xb] - N[yb * (bw + 1) + xa] + N[ya * (bw + 1) + xa];
    const s = S[yb * (bw + 1) + xb] - S[ya * (bw + 1) + xb] - S[yb * (bw + 1) + xa] + S[ya * (bw + 1) + xa];
    const base = n > 1 ? s / n : g;
    out[k] = lum[k] * wb[k] + base * (1 - wb[k]);
  }
  return out;
}

/**
 * 매니큐어 바른 손톱 마스크를 칠한 색 쪽 픽셀만으로 다듬고, 떨어진 덩어리로 나눈다.
 * 옆으로 본 두 손가락처럼 SAM 이 손톱 둘과 그 사이 손가락 살을 한 덩어리로 잡으면 큰 네모 하나로 칠해졌다.
 * 둘레(3~8px 밖) 피부색과 ΔE 20 넘게 다른 마스크 픽셀의 가운데 색을 매니큐어 색으로 보고, 피부보다 그 색에 가까운 픽셀만 남긴다.
 * 매니큐어가 없거나(먼 픽셀 30% 미만·피부와 ΔE 30 미만) 다듬어도 거의 그대로(85% 넘게 남음)면 [mask] 그대로.
 * 반환 마스크는 새 배열(볼록 껍질 채우기 전 입력을 넣는다).
 */
export function splitPolished(mask: Uint8Array, w: number, h: number, rgba: Uint8ClampedArray): Uint8Array[] {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, area = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { const x = i % w, y = (i - x) / w; area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (area < 30) return [mask];
  const lab = new Float32Array(3);
  const at = (i: number): [number, number, number] => { rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab); return [lab[0], lab[1], lab[2]]; };
  const dist = (p: number[], q: number[]) => Math.hypot((p[0] - q[0]) * 0.5, p[1] - q[1], p[2] - q[2]);
  const med = (v: number[][]) => [0, 1, 2].map((c) => v.map((p) => p[c]).sort((a, b) => a - b)[v.length >> 1]);
  // 둘레 띠: 마스크 밖이면서 3~8px 안에 마스크가 있는 픽셀
  const ring: number[][] = [];
  for (let y = Math.max(0, y0 - 8); y <= Math.min(h - 1, y1 + 8); y += 2) for (let x = Math.max(0, x0 - 8); x <= Math.min(w - 1, x1 + 8); x += 2) {
    const i = y * w + x;
    if (mask[i]) continue;
    let near = false, close = false;
    for (let dy = -8; dy <= 8 && !close; dy += 2) for (let dx = -8; dx <= 8; dx += 2) {
      const xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= w || yy >= h || !mask[yy * w + xx]) continue;
      if (Math.abs(dx) < 3 && Math.abs(dy) < 3) { close = true; break; }
      near = true;
    }
    if (near && !close) ring.push(at(i));
  }
  if (ring.length < 10) return [mask];
  const skin = med(ring);
  const px: number[] = [], cols: number[][] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) { px.push(i); cols.push(at(i)); }
  const far = cols.filter((c) => dist(c, skin) > 20);
  if (far.length < cols.length * 0.3) return [mask];
  const pol = med(far);
  if (dist(pol, skin) < 30) return [mask];
  const keep = new Uint8Array(mask.length);
  let kept = 0;
  for (let k = 0; k < px.length; k++) if (dist(cols[k], pol) < dist(cols[k], skin)) { keep[px[k]] = 1; kept++; }
  if (kept > area * 0.85) return [mask];
  // 덩어리 나누기(4방향), 가장 큰 것의 15%·30px 미만은 버림
  const lab2 = new Int32Array(mask.length).fill(-1), comps: number[][] = [];
  for (const i0 of px) {
    if (!keep[i0] || lab2[i0] >= 0) continue;
    const st = [i0], cur: number[] = [];
    lab2[i0] = comps.length;
    while (st.length) {
      const i = st.pop()!, x = i % w;
      cur.push(i);
      for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i - w, i + w]) if (j >= 0 && j < mask.length && keep[j] && lab2[j] < 0) { lab2[j] = comps.length; st.push(j); }
    }
    comps.push(cur);
  }
  const big = Math.max(0, ...comps.map((c) => c.length));
  const out = comps.filter((c) => c.length >= 30 && c.length >= big * 0.15).map((c) => { const m = new Uint8Array(mask.length); for (const i of c) m[i] = 1; return m; });
  return out.length ? out : [mask];
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

/**
 * 손톱 마스크를 가장자리의 '손톱 색 쪽' 픽셀로 조금 넓힌다(제자리 수정). SAM 마스크는 실제 손톱보다 1~2px 작아,
 * 다른 네일을 입히면 원래 매니큐어 색(빨강)이 테두리로 비친다. 둘레 피부 평균 → 손톱 평균 선 위에서 손톱 쪽(35% 넘게)인 픽셀만, 최대 maxPx 번.
 */
export function growNailMask(mask: Uint8Array, w: number, h: number, rgba: Uint8ClampedArray, maxPx: number): void {
  const lab = new Float32Array(3);
  const at = (i: number): [number, number, number] => { rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab); return [lab[0], lab[1], lab[2]]; };
  let x0 = w, y0 = h, x1 = 0, y1 = 0, area = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { const x = i % w, y = (i / w) | 0; area++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (area < 10) return;
  const pad = maxPx + 6;
  const bx0 = Math.max(0, x0 - pad), by0 = Math.max(0, y0 - pad), bx1 = Math.min(w - 1, x1 + pad), by1 = Math.min(h - 1, y1 + pad);
  // 손톱 속(가장자리 2px 안쪽)과 둘레 띠(3~6px 밖)의 평균
  const dist = (x: number, y: number, r: number, want: number): boolean => {
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      const v = xx >= 0 && yy >= 0 && xx < w && yy < h ? mask[yy * w + xx] : 0;
      if (v === want) return true;
    }
    return false;
  };
  const nail = [0, 0, 0], ring = [0, 0, 0];
  let nn = 0, nr = 0;
  for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) {
    const i = y * w + x;
    if (mask[i] && !dist(x, y, 2, 0)) { const c = at(i); nail[0] += c[0]; nail[1] += c[1]; nail[2] += c[2]; nn++; }
    else if (!mask[i] && !dist(x, y, 2, 1) && dist(x, y, 6, 1)) { const c = at(i); ring[0] += c[0]; ring[1] += c[1]; ring[2] += c[2]; nr++; }
  }
  if (!nn || !nr) return;
  for (let c = 0; c < 3; c++) { nail[c] /= nn; ring[c] /= nr; }
  const d = [nail[0] - ring[0], nail[1] - ring[1], nail[2] - ring[2]];
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  // 손톱과 둘레 색이 비슷하면(맨 손톱·연한 네일) 넓히지 않는다: 손끝 살로 번진다
  if (dd < 25 * 25) return;
  for (let it = 0; it < maxPx; it++) {
    const add: number[] = [];
    for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) {
      const i = y * w + x;
      if (mask[i]) continue;
      if (!((x > 0 && mask[i - 1]) || (x < w - 1 && mask[i + 1]) || (y > 0 && mask[i - w]) || (y < h - 1 && mask[i + w]))) continue;
      const c = at(i);
      const t = ((c[0] - ring[0]) * d[0] + (c[1] - ring[1]) * d[1] + (c[2] - ring[2]) * d[2]) / dd;
      if (t > 0.35) add.push(i);
    }
    if (!add.length) break;
    for (const i of add) mask[i] = 1;
  }
}

/**
 * 큰 마스크(outer)에서 작은 마스크(inner)를 뺀 테두리 부분의 평균 색과, outer 바깥 r 칸 띠의 평균 색 사이 ΔE(Lab).
 * 크면 테두리는 둘레(손가락 피부)가 아니라 손톱의 일부(크롬 테두리·프렌치 끝)다.
 */
export function rimContrast(rgba: Uint8ClampedArray, w: number, h: number, inner: Uint8Array, outer: Uint8Array, r = 4): number {
  let x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (let i = 0; i < outer.length; i++) if (outer[i]) { const x = i % w, y = (i - x) / w; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  if (x1 < x0) return 0;
  const lab = new Float32Array(3);
  const rim = [0, 0, 0], out = [0, 0, 0];
  let nRim = 0, nOut = 0;
  for (let y = Math.max(0, y0 - r); y <= Math.min(h - 1, y1 + r); y++) for (let x = Math.max(0, x0 - r); x <= Math.min(w - 1, x1 + r); x++) {
    const i = y * w + x;
    if (outer[i]) {
      if (inner[i]) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      rim[0] += lab[0]; rim[1] += lab[1]; rim[2] += lab[2]; nRim++;
      continue;
    }
    let near = false;
    for (let j = -r; j <= r && !near; j += r) for (let k = -r; k <= r; k += r) {
      const xx = x + k, yy = y + j;
      if (xx >= 0 && yy >= 0 && xx < w && yy < h && outer[yy * w + xx]) { near = true; break; }
    }
    if (!near) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    out[0] += lab[0]; out[1] += lab[1]; out[2] += lab[2]; nOut++;
  }
  if (!nRim || !nOut) return 0;
  return Math.hypot(rim[0] / nRim - out[0] / nOut, rim[1] / nRim - out[1] / nOut, rim[2] / nRim - out[2] / nOut);
}
