// 사진 모드 타투: 고른 사진의 팔 타투를 내 사진의 같은 자리(위팔→위팔, 아래팔→아래팔)로 옮겨 굽는다.
// 두 사진의 어깨·팔꿈치·손목 점으로 팔 마디마다 (축 방향 t, 옆 방향 s) 좌표를 만들어 대응시키고,
// 고른 사진에서 뽑은 잉크 투과율을 내 피부에 곱한다(옷이 덮은 곳은 몸 피부 확률로 자연히 빠진다).
// 잉크는 피부 아래 색소라 피부의 반짝임(작은 밝은 결)은 잉크 위에 다시 살린다. 순수 계산(DOM 없음).

import { boxBlur } from './still-hair.ts';

export type P = { x: number; y: number };
/** 팔 하나: 어깨·팔꿈치·손목과 마디별 반폭(px) */
export interface Arm { s: P; e: P; w: P; rUp: number; rLo: number }
/** 잉크(투과율 RGB + 진하기 A)와 그것이 덮는 사진 속 사각형 */
export interface InkMap { rgba: Uint8ClampedArray; w: number; h: number; rect: { x: number; y: number; w: number; h: number } }

const SUB = [[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]];

/** 축 a→b 의 t(0~1)·s(반폭 기준, 오른쪽 +) */
function local(p: P, a: P, b: P, r: number): { t: number; s: number } {
  const dx = b.x - a.x, dy = b.y - a.y;
  const L2 = dx * dx + dy * dy || 1;
  const t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / L2;
  const s = ((p.x - a.x) * -dy + (p.y - a.y) * dx) / Math.sqrt(L2) / r;
  return { t, s };
}
function at(a: P, b: P, r: number, t: number, s: number): P {
  const dx = b.x - a.x, dy = b.y - a.y;
  const L = Math.hypot(dx, dy) || 1;
  return { x: a.x + dx * t + (-dy / L) * s * r, y: a.y + dy * t + (dx / L) * s * r };
}

/**
 * 몸 피부 확률에서 팔 마디 반폭을 잰다(축의 35·50·65% 자리에서 옆으로 피부가 끊기는 곳까지, 가운데 값).
 * 피부가 안 보이면 null
 */
export function measureHalfWidth(skin: Float32Array, W: number, H: number, a: P, b: P): number | null {
  const L = Math.hypot(b.x - a.x, b.y - a.y);
  if (L < 8) return null;
  const nx = -(b.y - a.y) / L, ny = (b.x - a.x) / L;
  const on = (x: number, y: number) => {
    const xi = Math.round(x), yi = Math.round(y);
    return xi >= 0 && yi >= 0 && xi < W && yi < H && skin[yi * W + xi] > 0.5;
  };
  const ws: number[] = [];
  for (const t of [0.35, 0.5, 0.65]) {
    const cx = a.x + (b.x - a.x) * t, cy = a.y + (b.y - a.y) * t;
    if (!on(cx, cy)) continue;
    let r1 = 0, r2 = 0;
    const max = L * 0.5;
    while (r1 < max && on(cx + nx * (r1 + 1), cy + ny * (r1 + 1))) r1++;
    while (r2 < max && on(cx - nx * (r2 + 1), cy - ny * (r2 + 1))) r2++;
    if (r1 < max && r2 < max) ws.push((r1 + r2 + 1) / 2);
  }
  if (!ws.length) return null;
  ws.sort((p, q) => p - q);
  return ws[Math.floor(ws.length / 2)];
}

/** 잉크 투과율 맵에서 사진 좌표 (x, y)의 [r, g, b, a](0~1, 겹선형) */
function sampleInk(ink: InkMap, x: number, y: number, o: Float32Array): void {
  const u = ((x - ink.rect.x) / ink.rect.w) * ink.w - 0.5, v = ((y - ink.rect.y) / ink.rect.h) * ink.h - 0.5;
  o[0] = o[1] = o[2] = 1;
  o[3] = 0;
  if (u < -0.5 || v < -0.5 || u > ink.w - 0.5 || v > ink.h - 0.5) return;
  const uu = Math.max(0, Math.min(ink.w - 1.001, u)), vv = Math.max(0, Math.min(ink.h - 1.001, v));
  const xi = Math.floor(uu), yi = Math.floor(vv), fx = uu - xi, fy = vv - yi;
  const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
  const i00 = (yi * ink.w + xi) * 4, i10 = i00 + 4, i01 = i00 + ink.w * 4, i11 = i01 + 4;
  const d = ink.rgba;
  // 진하기로 가중(잉크 없는 칸의 투과율 1 이 잉크 색을 묽히지 않게)
  const a = (d[i00 + 3] * w00 + d[i10 + 3] * w10 + d[i01 + 3] * w01 + d[i11 + 3] * w11) / 255;
  o[3] = a;
  if (a < 1e-3) return;
  for (let c = 0; c < 3; c++) {
    o[c] = (d[i00 + c] * d[i00 + 3] * w00 + d[i10 + c] * d[i10 + 3] * w10 + d[i01 + c] * d[i01 + 3] * w01 + d[i11 + c] * d[i11 + 3] * w11) / (255 * 255 * a);
  }
}

/**
 * 잉크 맵을 f×f 칸 평균으로 줄인다(진하기로 가중한 색). 큰 도안을 작은 자리에 옮길 때 미리 줄여 두면
 * 가는 선이 샘플 사이로 빠져 점·조각으로 깨지지 않고 옅게 이어진다. rect 는 그대로(남는 칸은 잉크 없음).
 */
export function inkMip(ink: InkMap, f: number): InkMap {
  if (f < 2) return ink;
  const w = Math.ceil(ink.w / f), h = Math.ceil(ink.h / f);
  const d = ink.rgba, o = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let a = 0, r = 0, g = 0, b = 0;
    for (let yy = y * f; yy < Math.min(ink.h, (y + 1) * f); yy++) for (let xx = x * f; xx < Math.min(ink.w, (x + 1) * f); xx++) {
      const i = (yy * ink.w + xx) * 4, ai = d[i + 3];
      a += ai; r += d[i] * ai; g += d[i + 1] * ai; b += d[i + 2] * ai;
    }
    const j = (y * w + x) * 4;
    o[j + 3] = a / (f * f);
    if (a > 0) { o[j] = r / a; o[j + 1] = g / a; o[j + 2] = b / a; } else o[j] = o[j + 1] = o[j + 2] = 255;
  }
  const r = ink.rect;
  return { rgba: o, w, h, rect: { x: r.x, y: r.y, w: (r.w * w * f) / ink.w, h: (r.h * h * f) / ink.h } };
}

/** 줄이는 배율(사진 1px 당 고른 사진 px)에 맞춰 잉크 맵을 미리 줄인다 */
function prepInk(ink: InkMap, refPx: number): InkMap {
  return inkMip(ink, mipFactor(ink, refPx));
}

/** 사진 1px 이 잉크 맵 몇 칸에 걸치는지 → 미리 줄일 배수(2×2 나눠 뽑기에 맞춰 사진 1px 당 2칸 남김) */
function mipFactor(ink: InkMap, refPx: number): number {
  return Math.floor((refPx * ink.w) / ink.rect.w / 2);
}

/** 팔 마디 안(옆 반폭의 1.15배, 축 앞뒤 조금 여유)에 든 잉크 진하기 합 */
export function inkOnArm(ink: InkMap, arm: Arm): { up: number; lo: number } {
  let up = 0, lo = 0;
  for (let y = 0; y < ink.h; y++) for (let x = 0; x < ink.w; x++) {
    const a = ink.rgba[(y * ink.w + x) * 4 + 3];
    if (a < 32) continue;
    const p = { x: ink.rect.x + ((x + 0.5) / ink.w) * ink.rect.w, y: ink.rect.y + ((y + 0.5) / ink.h) * ink.rect.h };
    const U = local(p, arm.s, arm.e, arm.rUp), D = local(p, arm.e, arm.w, arm.rLo);
    if (U.t > -0.15 && U.t < 1.05 && Math.abs(U.s) < 1.15) up += a / 255;
    else if (D.t > -0.05 && D.t < 1.15 && Math.abs(D.s) < 1.15) lo += a / 255;
  }
  return { up, lo };
}

/**
 * 내 사진(dst)의 팔에 고른 사진 팔(srcArm)의 잉크를 옮긴다.
 * mirror: 반대쪽 팔로 옮길 때(옆 방향을 뒤집어 바깥쪽이 바깥쪽으로 가게). skin: 내 사진 몸 피부 확률.
 */
export function transferArmInk(
  dst: Uint8ClampedArray, W: number, H: number, dstArm: Arm, skin: Float32Array,
  ink: InkMap, srcArm: Arm | ((lambda: number, s: number) => P | null), mirror = false, amount = 1,
): Uint8ClampedArray {
  // 고른 사진 쪽 대응: 팔 점이 있으면 마디끼리, 없으면 넘겨받은 함수(내 팔 길이 λ px, 옆 s → 고른 사진 좌표)
  const upLen = Math.hypot(dstArm.e.x - dstArm.s.x, dstArm.e.y - dstArm.s.y);
  const loLen = Math.hypot(dstArm.w.x - dstArm.e.x, dstArm.w.y - dstArm.e.y);
  const refAt = typeof srcArm === 'function'
    ? (g: number, t: number, s: number) => srcArm(g === 0 ? t * upLen : upLen + t * loLen, s)
    : (g: number, t: number, s: number) => (g === 0 ? at(srcArm.s, srcArm.e, srcArm.rUp, t, s) : at(srcArm.e, srcArm.w, srcArm.rLo, t, s));
  const out = new Uint8ClampedArray(dst);
  // 줄여 옮기는 배율: 마디 몇 곳에서 내 사진 1px(길이·옆) 이 고른 사진 몇 px 인지
  const steps: number[] = [];
  for (let g = 0; g < 2; g++) for (const t of [0.2, 0.4, 0.6, 0.8]) {
    const len = g === 0 ? upLen : loLen, r = g === 0 ? dstArm.rUp : dstArm.rLo;
    if (len < 4 || r < 2) continue;
    const p0 = refAt(g, t, 0), p1 = refAt(g, t + 1 / len, 0), p2 = refAt(g, t, 1 / r);
    if (!p0 || !p1 || !p2) continue;
    steps.push(Math.max(Math.hypot(p1.x - p0.x, p1.y - p0.y), Math.hypot(p2.x - p0.x, p2.y - p0.y)));
  }
  if (steps.length) {
    steps.sort((a, b) => a - b);
    ink = prepInk(ink, steps[steps.length >> 1]);
  }
  // 피부 반짝임(작은 밝은 결): 밝기 - 흐린 밝기
  const L = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) L[i] = 0.299 * dst[i * 4] + 0.587 * dst[i * 4 + 1] + 0.114 * dst[i * 4 + 2];
  const rb = Math.max(2, Math.round(dstArm.rUp * 0.15));
  const Lb = boxBlur(L, W, H, rb);
  const segs: [P, P, number][] = [
    [dstArm.s, dstArm.e, dstArm.rUp],
    [dstArm.e, dstArm.w, dstArm.rLo],
  ];
  // 처리할 범위: 두 마디를 감싸는 상자
  const pad = Math.max(dstArm.rUp, dstArm.rLo) * 1.3;
  const xs = [dstArm.s.x, dstArm.e.x, dstArm.w.x], ys = [dstArm.s.y, dstArm.e.y, dstArm.w.y];
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - pad)), x1 = Math.min(W - 1, Math.ceil(Math.max(...xs) + pad));
  const y0 = Math.max(0, Math.floor(Math.min(...ys) - pad)), y1 = Math.min(H - 1, Math.ceil(Math.max(...ys) + pad));
  const o = new Float32Array(4), acc = new Float32Array(4);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x;
    const k = Math.max(0, Math.min(1, (skin[i] - 0.3) / 0.4));
    if (k <= 0) continue;
    // 마디마다 대응점을 구해, 팔꿈치 근처에서는 두 마디를 거리로 섞는다(이음매 없이)
    acc.fill(0);
    let wsum = 0;
    // 고른 사진이 더 크게 찍혀 줄여 옮기는 경우가 많아 한 픽셀을 2×2 로 나눠 평균한다(가는 선이 계단처럼 깨지지 않게)
    for (const [ox, oy] of SUB) {
      const px = { x: x + ox, y: y + oy };
      for (let g = 0; g < 2; g++) {
        const [a, b, r] = segs[g];
        const q = local(px, a, b, r);
        if (Math.abs(q.s) > 1.25) continue;
        // 마디 밖(축 앞뒤)은 팔꿈치 쪽만 조금 이어 준다
        const out0 = g === 0 ? Math.max(0, q.t - 1) : Math.max(0, -q.t);
        const far = g === 0 ? Math.max(0, -q.t - 0.1) : Math.max(0, q.t - 1.1);
        if (far > 0) continue;
        const wgt = Math.max(0, 1 - out0 / 0.12);
        if (wgt <= 0) continue;
        const p = refAt(g, q.t, mirror ? -q.s : q.s);
        // 고른 사진 띠 밖은 '잉크 없음'으로 세어 끝이 부드럽게 끝나게
        if (!p) {
          wsum += wgt;
          continue;
        }
        sampleInk(ink, p.x, p.y, o);
        for (let c = 0; c < 4; c++) acc[c] += (c < 3 ? o[c] * o[3] : o[3]) * wgt;
        wsum += wgt;
      }
    }
    if (wsum <= 0) continue;
    const a = Math.min(1, (acc[3] / wsum) * amount) * k;
    if (a < 0.004) continue;
    const hl = Math.max(0, L[i] - Lb[i]);
    for (let c = 0; c < 3; c++) {
      const T = acc[c] / Math.max(1e-6, acc[3]);
      const v = dst[i * 4 + c];
      // 투과율 곱하기 + 피부 반짝임을 잉크 위에 다시(잉크가 진할수록 반짝임이 더 드러난다)
      const inked = v * Math.min(1, T) + hl * 0.7;
      out[i * 4 + c] = v + (inked - v) * a;
    }
  }
  return out;
}

/**
 * 팔 점을 못 찾는 가까운 팔 사진용 '띠': 피부·잉크 영역에서 가장 먼 두 끝을 잇는 영역 안 최단 경로를 팔 중심선으로 본다
 * (굽은 팔도 따라간다). 중심선 위 칸마다 옆으로 영역이 끝나는 곳을 재서 가운데·반폭을 고친다.
 * 방향은 굵은 쪽(어깨) → 가는 쪽(손목). mask: 0~1(W*H), roi 안만 본다.
 */
export interface Strip { p: P[]; n: P[]; h: Float32Array; len: number; ratio: number; mean: number }
export function armStrip(mask: Float32Array, W: number, H: number, roi: { x: number; y: number; w: number; h: number }): Strip | null {
  const rx = Math.max(0, roi.x), ry = Math.max(0, roi.y);
  const rw = Math.min(W, roi.x + roi.w) - rx, rh = Math.min(H, roi.y + roi.h) - ry;
  if (rw < 10 || rh < 10) return null;
  // 거친 격자(긴 변 ~240칸)에서 길 찾기
  const f = Math.max(1, Math.max(rw, rh) / 240);
  const gw = Math.max(1, Math.floor(rw / f)), gh = Math.max(1, Math.floor(rh / f));
  const g = new Uint8Array(gw * gh);
  const on = (x: number, y: number) => {
    const xi = Math.round(x), yi = Math.round(y);
    return xi >= 0 && yi >= 0 && xi < W && yi < H && mask[yi * W + xi] > 0.5;
  };
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) g[y * gw + x] = on(rx + (x + 0.5) * f, ry + (y + 0.5) * f) ? 1 : 0;
  // 가장 큰 덩어리
  const lab = new Int32Array(gw * gh).fill(-1);
  let best = -1, bestN = 0;
  const q = new Int32Array(gw * gh);
  for (let i = 0, id = 0; i < g.length; i++) {
    if (!g[i] || lab[i] >= 0) continue;
    let h0 = 0, t0 = 0;
    q[t0++] = i;
    lab[i] = id;
    while (h0 < t0) {
      const c = q[h0++], cx = c % gw, cy = (c - cx) / gw;
      for (const [dx, dy] of N4) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
        const j = y * gw + x;
        if (g[j] && lab[j] < 0) { lab[j] = id; q[t0++] = j; }
      }
    }
    if (t0 > bestN) { bestN = t0; best = id; }
    id++;
  }
  if (bestN < 50) return null;
  const bfs = (start: number, par: Int32Array | null): number => {
    const dist = new Int32Array(gw * gh).fill(-1);
    let h0 = 0, t0 = 0, far = start;
    q[t0++] = start;
    dist[start] = 0;
    while (h0 < t0) {
      const c = q[h0++], cx = c % gw, cy = (c - cx) / gw;
      if (dist[c] > dist[far]) far = c;
      for (const [dx, dy] of N4) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
        const j = y * gw + x;
        if (lab[j] === best && dist[j] < 0) { dist[j] = dist[c] + 1; if (par) par[j] = c; q[t0++] = j; }
      }
    }
    return far;
  };
  const A = bfs(lab.indexOf(best), null);
  const par = new Int32Array(gw * gh).fill(-1);
  const B = bfs(A, par);
  const raw: P[] = [];
  for (let c = B; c >= 0; c = c === A ? -1 : par[c]) raw.push({ x: rx + ((c % gw) + 0.5) * f, y: ry + (Math.floor(c / gw) + 0.5) * f });
  if (raw.length < 6) return null;
  // 같은 간격 K 점으로 다시 뽑고 고른다
  const K = 48;
  const cum = [0];
  for (let i = 1; i < raw.length; i++) cum.push(cum[i - 1] + Math.hypot(raw[i].x - raw[i - 1].x, raw[i].y - raw[i - 1].y));
  const total = cum[cum.length - 1];
  let pts: P[] = [];
  for (let k = 0, j = 0; k < K; k++) {
    const d = (k / (K - 1)) * total;
    while (j < raw.length - 2 && cum[j + 1] < d) j++;
    const w = (d - cum[j]) / Math.max(1e-6, cum[j + 1] - cum[j]);
    pts.push({ x: raw[j].x + (raw[j + 1].x - raw[j].x) * w, y: raw[j].y + (raw[j + 1].y - raw[j].y) * w });
  }
  const smoothP = (a: P[]) => a.map((_, k) => {
    const u = a[Math.max(0, k - 1)], v = a[k], w = a[Math.min(K - 1, k + 1)];
    return { x: (u.x + 2 * v.x + w.x) / 4, y: (u.y + 2 * v.y + w.y) / 4 };
  });
  const normals = (a: P[]) => a.map((_, k) => {
    const u = a[Math.max(0, k - 1)], w = a[Math.min(K - 1, k + 1)];
    const L = Math.hypot(w.x - u.x, w.y - u.y) || 1;
    return { x: -(w.y - u.y) / L, y: (w.x - u.x) / L };
  });
  let h = new Float32Array(K);
  for (let pass = 0; pass < 2; pass++) {
    for (let r = 0; r < 4; r++) pts = smoothP(pts);
    const ns = normals(pts);
    const maxR = total * 0.5;
    const cs = new Float32Array(K);
    for (let k = 0; k < K; k++) {
      const p = pts[k], n = ns[k];
      let r1 = 0, r2 = 0;
      while (r1 < maxR && on(p.x + n.x * (r1 + 1), p.y + n.y * (r1 + 1))) r1++;
      while (r2 < maxR && on(p.x - n.x * (r2 + 1), p.y - n.y * (r2 + 1))) r2++;
      cs[k] = (r1 - r2) / 2;
      h[k] = Math.max(2, (r1 + r2 + 1) / 2);
    }
    pts = pts.map((p, k) => ({ x: p.x + ns[k].x * cs[k], y: p.y + ns[k].y * cs[k] }));
  }
  // 마지막 가운데 맞춤 뒤에도 고른다: 팔 영역 가장자리의 들쭉날쭉함(분할 경계)이 칸마다 중심·반폭을 흔들어 도안이 지그재그로 휘었다
  for (let r = 0; r < 4; r++) pts = smoothP(pts);
  for (let r = 0; r < 3; r++) h = h.map((_, k) => (h[Math.max(0, k - 1)] + h[k] * 2 + h[Math.min(K - 1, k + 1)]) / 4);
  // 굵은 쪽이 앞(어깨)이 되게. 끝 칸은 잘린 모서리라 빼고 잰다
  const qn = Math.floor(K / 4);
  let hA = 0, hB = 0;
  for (let k = 2; k < qn + 2; k++) { hA += h[k]; hB += h[K - 1 - k]; }
  if (hB > hA) { pts.reverse(); h.reverse(); }
  let len = 0;
  for (let k = 1; k < K; k++) len += Math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y);
  let mean = 0;
  for (let k = 0; k < K; k++) mean += h[k];
  mean /= K;
  return { p: pts, n: normals(pts), h, len, ratio: Math.max(hA, hB) / Math.max(1e-3, Math.min(hA, hB)), mean };
}
const N4 = [[1, 0], [-1, 0], [0, 1], [0, -1]];

/** 띠의 어깨 쪽 끝에서 tau(px) 만큼, 옆으로 sigma(반폭 기준) 간 자리. 띠 밖이면 null */
export function stripPoint(st: Strip, tau: number, sigma: number): P | null {
  if (tau < 0 || tau > st.len) return null;
  const K = st.p.length;
  const f = Math.min(K - 1.001, (tau / st.len) * (K - 1));
  const k = Math.floor(f), w = f - k;
  const lerp = (a: number, b: number) => a * (1 - w) + b * w;
  const nx = lerp(st.n[k].x, st.n[k + 1].x), ny = lerp(st.n[k].y, st.n[k + 1].y);
  const nl = Math.hypot(nx, ny) || 1;
  const s = sigma * lerp(st.h[k], st.h[k + 1]);
  return { x: lerp(st.p[k].x, st.p[k + 1].x) + (nx / nl) * s, y: lerp(st.p[k].y, st.p[k + 1].y) + (ny / nl) * s };
}

/**
 * 도안 영역(rect)과 가장 많이 겹치는 피부·잉크 덩어리(팔 전체)와 그 둘레 상자.
 * 자동으로 고른 도안 영역이 소매의 일부만 잡았을 때 팔 전체로 넓히는 데 쓴다.
 */
export function limbRegion(m: Float32Array, W: number, H: number, rect: { x: number; y: number; w: number; h: number }): { mask: Float32Array; box: { x: number; y: number; w: number; h: number } } | null {
  const f = Math.max(1, Math.max(W, H) / 400);
  const gw = Math.ceil(W / f), gh = Math.ceil(H / f);
  const g = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const X = Math.min(W - 1, Math.floor((x + 0.5) * f)), Y = Math.min(H - 1, Math.floor((y + 0.5) * f));
    g[y * gw + x] = m[Y * W + X] > 0.4 ? 1 : 0;
  }
  const lab = new Int32Array(gw * gh).fill(-1);
  const q = new Int32Array(gw * gh);
  let best = -1, bestHit = 0;
  for (let i = 0, id = 0; i < g.length; i++) {
    if (!g[i] || lab[i] >= 0) continue;
    let h0 = 0, t0 = 0, hit = 0;
    q[t0++] = i;
    lab[i] = id;
    while (h0 < t0) {
      const c = q[h0++], cx = c % gw, cy = (c - cx) / gw;
      const X = (cx + 0.5) * f, Y = (cy + 0.5) * f;
      if (X >= rect.x && Y >= rect.y && X < rect.x + rect.w && Y < rect.y + rect.h) hit++;
      for (const [dx, dy] of N4) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= gw || y >= gh) continue;
        const j = y * gw + x;
        if (g[j] && lab[j] < 0) { lab[j] = id; q[t0++] = j; }
      }
    }
    if (hit > bestHit) { bestHit = hit; best = id; }
    id++;
  }
  if (best < 0) return null;
  // 진한 타투는 피부로 안 잡혀 구멍이 많다: 덩어리 칸을 한 칸 넓혔다 줄여(닫기) 꽉 찬 팔로
  const cell = new Uint8Array(gw * gh);
  for (let i = 0; i < cell.length; i++) cell[i] = lab[i] === best ? 1 : 0;
  const grow = (a: Uint8Array, v: number) => {
    const b = new Uint8Array(a);
    for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
      if (a[y * gw + x] === v) continue;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = x + dx, Y = y + dy;
        if (X >= 0 && Y >= 0 && X < gw && Y < gh && a[Y * gw + X] === v) { b[y * gw + x] = v; dy = dx = 2; }
      }
    }
    return b;
  };
  const closed = grow(grow(grow(grow(cell, 1), 1), 0), 0);
  const mask = new Float32Array(W * H);
  let x0 = W, y0 = H, x1 = 0, y1 = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const gx = Math.min(gw - 1, Math.floor(x / f)), gy = Math.min(gh - 1, Math.floor(y / f));
    if (!closed[gy * gw + gx]) continue;
    mask[y * W + x] = 1;
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return { mask, box: { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 } };
}

/** 세 점 대응(a[i] → b[i])의 아핀 변환 [m00, m01, m02, m10, m11, m12]: b = M·[x, y, 1] */
export function affine3(a: P[], b: P[]): number[] | null {
  const [p0, p1, p2] = a;
  const det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
  if (Math.abs(det) < 1e-6) return null;
  // a 쪽 기저의 역행렬
  const i00 = (p2.y - p0.y) / det, i01 = -(p2.x - p0.x) / det, i10 = -(p1.y - p0.y) / det, i11 = (p1.x - p0.x) / det;
  const row = (k: 'x' | 'y') => {
    const d1 = b[1][k] - b[0][k], d2 = b[2][k] - b[0][k];
    const m0 = d1 * i00 + d2 * i10, m1 = d1 * i01 + d2 * i11;
    return [m0, m1, b[0][k] - m0 * p0.x - m1 * p0.y];
  };
  return [...row('x'), ...row('y')];
}
const apply = (M: number[], p: P): P => ({ x: M[0] * p.x + M[1] * p.y + M[2], y: M[3] * p.x + M[4] * p.y + M[5] });

/**
 * 몸통 타투(등·가슴): 어깨 두 점과 엉덩이 가운데의 대응(내 사진 → 고른 사진 아핀)으로 잉크를 옮긴다.
 * toRef: 내 사진 좌표 → 고른 사진 좌표, fromRef: 그 반대(범위 계산용).
 */
export function transferTorsoInk(
  dst: Uint8ClampedArray, W: number, H: number, skin: Float32Array, ink: InkMap, toRef: number[], fromRef: number[], amount = 1,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(dst);
  // 내 사진 1px → 고른 사진 몇 px(어파인 넓이 배율의 제곱근)
  ink = prepInk(ink, Math.sqrt(Math.abs(toRef[0] * toRef[4] - toRef[1] * toRef[3])));
  const r = ink.rect;
  const cs = [{ x: r.x, y: r.y }, { x: r.x + r.w, y: r.y }, { x: r.x, y: r.y + r.h }, { x: r.x + r.w, y: r.y + r.h }].map((p) => apply(fromRef, p));
  const x0 = Math.max(0, Math.floor(Math.min(...cs.map((p) => p.x)))), x1 = Math.min(W - 1, Math.ceil(Math.max(...cs.map((p) => p.x))));
  const y0 = Math.max(0, Math.floor(Math.min(...cs.map((p) => p.y)))), y1 = Math.min(H - 1, Math.ceil(Math.max(...cs.map((p) => p.y))));
  if (x1 <= x0 || y1 <= y0) return out;
  const L = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) L[i] = 0.299 * dst[i * 4] + 0.587 * dst[i * 4 + 1] + 0.114 * dst[i * 4 + 2];
  const Lb = boxBlur(L, W, H, Math.max(2, Math.round((x1 - x0) * 0.02)));
  const o = new Float32Array(4), acc = new Float32Array(4);
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * W + x;
    const k = Math.max(0, Math.min(1, (skin[i] - 0.3) / 0.4));
    if (k <= 0) continue;
    acc.fill(0);
    for (const [ox, oy] of SUB) {
      const p = apply(toRef, { x: x + ox, y: y + oy });
      sampleInk(ink, p.x, p.y, o);
      for (let c = 0; c < 4; c++) acc[c] += (c < 3 ? o[c] * o[3] : o[3]) / 4;
    }
    const a = Math.min(1, acc[3] * amount) * k;
    if (a < 0.004) continue;
    const hl = Math.max(0, L[i] - Lb[i]);
    for (let c = 0; c < 3; c++) {
      const T = acc[c] / Math.max(1e-6, acc[3]);
      const v = dst[i * 4 + c];
      out[i * 4 + c] = v + (v * Math.min(1, T) + hl * 0.7 - v) * a;
    }
  }
  return out;
}
