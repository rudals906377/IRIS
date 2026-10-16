// 사진 속에 내 얼굴: 고른 사진(모델의 머리·화장·옷·배경)은 그대로 두고 얼굴 부분만 내 얼굴로 바꾼다.
// ① 두 얼굴의 점 468개로 삼각형 망을 만들어(들로네) 내 얼굴을 고른 사진 얼굴 모양에 맞춰 휘고
// ② 얼굴 윤곽 안쪽 중 두 사진 모두 피부·눈·코·입인 자리(정밀 얼굴 분할)만 바꾸고
//    (고른 사진의 앞머리·안경, 내 사진의 앞머리는 옮기지 않는다)
// ③ 피부끼리만 크게 흐린 두 그림의 비로 조명·피부색을 고른 사진에 맞추고 ④ 가장자리를 부드럽게 섞는다.
// 순수 계산(DOM 없음).

import { boxBlur } from './still-hair.ts';

export type P = { x: number; y: number };

/** 들로네 삼각분할(Bowyer–Watson). 삼각형마다 점 번호 3개 */
export function delaunay(pts: P[]): number[] {
  const n = pts.length;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) { minX = Math.min(minX, p.x); minY = Math.min(minY, p.y); maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y); }
  const d = Math.max(maxX - minX, maxY - minY) * 20 + 1;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const all = pts.concat([{ x: cx - d, y: cy - d }, { x: cx + d, y: cy - d }, { x: cx, y: cy + d }]);
  type T = { a: number; b: number; c: number; ux: number; uy: number; r2: number };
  const circ = (a: number, b: number, c: number): T => {
    const A = all[a], B = all[b], C = all[c];
    const D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) || 1e-12;
    const ux = ((A.x * A.x + A.y * A.y) * (B.y - C.y) + (B.x * B.x + B.y * B.y) * (C.y - A.y) + (C.x * C.x + C.y * C.y) * (A.y - B.y)) / D;
    const uy = ((A.x * A.x + A.y * A.y) * (C.x - B.x) + (B.x * B.x + B.y * B.y) * (A.x - C.x) + (C.x * C.x + C.y * C.y) * (B.x - A.x)) / D;
    return { a, b, c, ux, uy, r2: (A.x - ux) ** 2 + (A.y - uy) ** 2 };
  };
  let tris: T[] = [circ(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const p = all[i];
    const bad: T[] = [], keep: T[] = [];
    for (const t of tris) ((p.x - t.ux) ** 2 + (p.y - t.uy) ** 2 < t.r2 ? bad : keep).push(t);
    const edges = new Map<string, [number, number]>();
    for (const t of bad) {
      for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
        const k = u < v ? `${u},${v}` : `${v},${u}`;
        if (edges.has(k)) edges.delete(k);
        else edges.set(k, [u, v]);
      }
    }
    tris = keep;
    for (const [u, v] of edges.values()) tris.push(circ(u, v, i));
  }
  const out: number[] = [];
  for (const t of tris) if (t.a < n && t.b < n && t.c < n) out.push(t.a, t.b, t.c);
  return out;
}

/** 다각형 채우기(0/1) */
export function fillPoly(poly: P[], w: number, h: number): Float32Array {
  const m = new Float32Array(w * h);
  let y0 = h, y1 = 0;
  for (const p of poly) { y0 = Math.min(y0, Math.floor(p.y)); y1 = Math.max(y1, Math.ceil(p.y)); }
  for (let y = Math.max(0, y0); y <= Math.min(h - 1, y1); y++) {
    const xs: number[] = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) xs.push(a.x + ((y - a.y) * (b.x - a.x)) / (b.y - a.y));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      for (let x = Math.max(0, Math.ceil(xs[k])); x <= Math.min(w - 1, Math.floor(xs[k + 1])); x++) m[y * w + x] = 1;
    }
  }
  return m;
}

/**
 * 내 얼굴(src, 점 srcPts)을 고른 사진(dst, 점 dstPts) 얼굴 자리에 휘어 넣은 새 그림(RGBA, dst 크기).
 * oval: 얼굴 윤곽 점 번호(순서대로). skin: dst 의 얼굴 피부·이목구비 확률(0~1, dst 크기, 없으면 윤곽만).
 * hands: dst 의 몸 피부(얼굴 가린 손) 확률. srcSkin: src 의 얼굴 피부·이목구비 확률(src 크기, 내 앞머리 빼기).
 */
export function swapFace(
  dst: Uint8ClampedArray, W: number, H: number, dstPts: P[],
  src: Uint8ClampedArray, sw: number, sh: number, srcPts: P[],
  oval: number[], skin: Float32Array | null = null, hands: Float32Array | null = null,
  srcSkin: Float32Array | null = null,
): Uint8ClampedArray {
  const N = Math.min(468, dstPts.length, srcPts.length);
  const dp = dstPts.slice(0, N), sp = srcPts.slice(0, N);
  const tris = delaunay(dp);
  // ① 휘기
  const warped = new Float32Array(W * H * 3);
  const cover = new Float32Array(W * H);
  const wSkin = srcSkin ? new Float32Array(W * H) : null;
  for (let t = 0; t < tris.length; t += 3) {
    const [i, j, k] = [tris[t], tris[t + 1], tris[t + 2]];
    const A = dp[i], B = dp[j], C = dp[k];
    const den = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y);
    if (Math.abs(den) < 1e-6) continue;
    const x0 = Math.max(0, Math.floor(Math.min(A.x, B.x, C.x))), x1 = Math.min(W - 1, Math.ceil(Math.max(A.x, B.x, C.x)));
    const y0 = Math.max(0, Math.floor(Math.min(A.y, B.y, C.y))), y1 = Math.min(H - 1, Math.ceil(Math.max(A.y, B.y, C.y)));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const l1 = ((B.y - C.y) * (x - C.x) + (C.x - B.x) * (y - C.y)) / den;
        const l2 = ((C.y - A.y) * (x - C.x) + (A.x - C.x) * (y - C.y)) / den;
        const l3 = 1 - l1 - l2;
        if (l1 < -1e-4 || l2 < -1e-4 || l3 < -1e-4) continue;
        const sx = l1 * sp[i].x + l2 * sp[j].x + l3 * sp[k].x;
        const sy = l1 * sp[i].y + l2 * sp[j].y + l3 * sp[k].y;
        const xx = Math.max(0, Math.min(sw - 1.001, sx)), yy = Math.max(0, Math.min(sh - 1.001, sy));
        const xi = Math.floor(xx), yi = Math.floor(yy), fx = xx - xi, fy = yy - yi;
        const o = (y * W + x) * 3;
        for (let c = 0; c < 3; c++) {
          const a = src[(yi * sw + xi) * 4 + c], b = src[(yi * sw + xi + 1) * 4 + c];
          const d = src[((yi + 1) * sw + xi) * 4 + c], e = src[((yi + 1) * sw + xi + 1) * 4 + c];
          warped[o + c] = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
        }
        cover[y * W + x] = 1;
        if (wSkin) wSkin[y * W + x] = srcSkin![Math.round(yy) * sw + Math.round(xx)];
      }
    }
  }
  // ② 바꿀 자리: 윤곽 안 ∩ 두 사진 모두 피부·이목구비(앞머리·안경 제외) ∩ 휜 곳, 안쪽으로 조금 줄이고 부드럽게
  const ov = oval.map((i) => dp[i]);
  let mask = fillPoly(ov, W, H);
  const ramp = (v: number) => Math.max(0, Math.min(1, (v - 0.3) / 0.4));
  const fw = Math.hypot(dp[454].x - dp[234].x, dp[454].y - dp[234].y);
  for (let i = 0; i < mask.length; i++) {
    mask[i] *= cover[i] * (skin ? ramp(skin[i]) : 1) * (wSkin ? ramp(wSkin[i]) : 1);
    // 얼굴을 가린 손·팔(몸 피부)은 고른 사진 그대로
    if (hands) mask[i] *= 1 - Math.max(0, Math.min(1, (hands[i] - 0.3) / 0.3));
  }
  const fr = Math.max(2, Math.round(fw * 0.03));
  const shrink = boxBlur(mask, W, H, fr);
  for (let i = 0; i < mask.length; i++) mask[i] = Math.max(0, Math.min(1, (shrink[i] - 0.5) * 2.2));
  mask = boxBlur(mask, W, H, Math.max(1, Math.round(fr * 0.7)));
  // 내 얼굴이 고른 사진 얼굴보다 많이 작으면(크게 늘리면) 흐려진다 → 늘린 만큼 잔무늬를 살짝 살린다(언샤프)
  const sfw = Math.hypot(sp[454].x - sp[234].x, sp[454].y - sp[234].y);
  const up = fw / Math.max(1, sfw);
  if (up > 1.4) {
    const k = Math.min(0.9, (up - 1.4) * 0.35);
    for (let c = 0; c < 3; c++) {
      const ch = new Float32Array(W * H);
      for (let i = 0; i < W * H; i++) ch[i] = warped[i * 3 + c];
      const bl = boxBlur(ch, W, H, Math.max(1, Math.round(up * 0.7)));
      for (let i = 0; i < W * H; i++) if (cover[i]) warped[i * 3 + c] = ch[i] + (ch[i] - bl[i]) * k;
    }
  }
  // ③ 조명·피부색 맞추기: 두 사진 모두 피부인 곳끼리만 크게 흐려(머리카락·배경이 섞이면 이마가 하얗게 뜬다) 비를 곱한다
  const rb = Math.max(3, Math.round(fw * 0.12));
  const wt = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    wt[i] = cover[i] * (skin ? skin[i] : 1) * (wSkin ? wSkin[i] : 1) * (hands ? 1 - hands[i] : 1);
  }
  const wb = boxBlur(wt, W, H, rb);
  const out = new Uint8ClampedArray(dst);
  for (let c = 0; c < 3; c++) {
    const a = new Float32Array(W * H), b = new Float32Array(W * H);
    let sa = 0, sb = 0;
    for (let i = 0; i < W * H; i++) {
      b[i] = dst[i * 4 + c] * wt[i];
      a[i] = (cover[i] ? warped[i * 3 + c] : dst[i * 4 + c]) * wt[i];
      sa += a[i];
      sb += b[i];
    }
    // 피부가 거의 없는 곳(가장자리)은 얼굴 전체 비를 쓴다
    const gr = (sb + 4) / (sa + 4);
    const ab = boxBlur(a, W, H, rb), bb = boxBlur(b, W, H, rb);
    for (let i = 0; i < W * H; i++) {
      const m = mask[i];
      if (m <= 0.002) continue;
      const v = cover[i] ? warped[i * 3 + c] : dst[i * 4 + c];
      const t = Math.min(1, wb[i] / 0.15);
      const r = t * ((bb[i] + 4 * wb[i]) / (ab[i] + 4 * wb[i] + 1e-6)) + (1 - t) * gr;
      const d0 = dst[i * 4 + c];
      out[i * 4 + c] = d0 + (Math.max(0, Math.min(255, v * r)) - d0) * m;
    }
  }
  return out;
}
