// 사진 속에 내 얼굴: 고른 사진(모델의 머리·화장·옷·배경)은 그대로 두고 얼굴 부분만 내 얼굴로 바꾼다.
// ① 두 얼굴의 점 468개로 삼각형 망을 만들어(들로네) 내 얼굴을 고른 사진 얼굴 모양에 맞춰 휘고
// ② 얼굴 윤곽 안쪽 중 두 사진 모두 피부·눈·코·입인 자리(정밀 얼굴 분할)만 바꾸고
//    (고른 사진의 앞머리·안경, 내 사진의 앞머리는 옮기지 않는다)
// ③ 피부끼리만 크게 흐린 두 그림의 비로 조명·피부색을 고른 사진에 맞추고 ④ 가장자리를 부드럽게 섞는다.
// 순수 계산(DOM 없음).

import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, LIPS_INNER } from './face-regions.ts';
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
 * src 를 점 대응(srcPts → dstPts, 들로네 삼각형마다 무게중심 좌표)으로 휘어 dst 크기(W×H)로.
 * rgb: 0~255 실수 3채널, cover: 삼각형이 덮은 곳 1, extra: 함께 휠 src 크기 확률 맵들(가장 가까운 칸).
 * fold: src 쪽 삼각형이 뒤집힌 곳 1(옆얼굴 사진에서 코 뒤로 가려진 뺨처럼, 내 사진엔 보이지만 dst 에선 가려진 자리).
 */
export function warpTriangles(
  dp: P[], W: number, H: number, src: Uint8ClampedArray, sw: number, sh: number, sp: P[], extraIn: Float32Array[] = [],
): { rgb: Float32Array; cover: Float32Array; extra: Float32Array[]; fold: Float32Array } {
  const tris = delaunay(dp);
  const warped = new Float32Array(W * H * 3);
  const cover = new Float32Array(W * H);
  const fold = new Float32Array(W * H);
  // 두 얼굴 크기 비(넓이): 눌림 판단의 기준
  const span = (q: P[]) => Math.hypot(q[454].x - q[234].x, q[454].y - q[234].y);
  const scale2 = dp.length > 454 && sp.length > 454 ? (span(dp) / Math.max(1, span(sp))) ** 2 : 1;
  const extra = extraIn.map(() => new Float32Array(W * H));
  for (let t = 0; t < tris.length; t += 3) {
    const [i, j, k] = [tris[t], tris[t + 1], tris[t + 2]];
    const A = dp[i], B = dp[j], C = dp[k];
    const den = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y);
    if (Math.abs(den) < 1e-6) continue;
    const sden = (sp[j].y - sp[k].y) * (sp[i].x - sp[k].x) + (sp[k].x - sp[j].x) * (sp[i].y - sp[k].y);
    // 뒤집혔거나 크게 눌린(고개 돌린 쪽 코 옆: 내 사진 넓은 자리가 가는 띠로) 삼각형
    const flip = sden * den < 0 || Math.abs(den) < Math.abs(sden) * scale2 * 0.3 ? 1 : 0;
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
        if (flip) fold[y * W + x] = 1;
        for (let e = 0; e < extra.length; e++) extra[e][y * W + x] = extraIn[e][Math.round(yy) * sw + Math.round(xx)];
      }
    }
  }
  return { rgb: warped, cover, extra, fold };
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
  // ① 휘기
  const wr = warpTriangles(dp, W, H, src, sw, sh, sp, srcSkin ? [srcSkin] : []);
  const { rgb: warped, cover, fold } = wr;
  const wSkin = srcSkin ? wr.extra[0] : null;
  const fw0 = Math.hypot(dp[454].x - dp[234].x, dp[454].y - dp[234].y);
  if (wSkin) {
    // 내 앞머리·잔머리가 가린 가는 틈은 고른 사진을 비치게 두면 두 얼굴이 섞여 얼룩(두 겹 눈썹, 콧볼 옆 조각)이 된다
    // → 얇은 틈(둘레에 내 피부가 충분)은 둘레 내 피부로 메우고 내 피부로 친다. 넓게 가린 곳(이마 전체 앞머리)은 그대로 고른 사진
    const valid = new Float32Array(W * H);
    for (let i = 0; i < valid.length; i++) valid[i] = cover[i] && wSkin[i] > 0.6 ? 1 : 0;
    const r = Math.max(2, Math.round(fw0 * 0.035));
    const den = boxBlur(valid, W, H, r);
    const num = [0, 1, 2].map((c) => {
      const ch = new Float32Array(W * H);
      for (let i = 0; i < ch.length; i++) ch[i] = warped[i * 3 + c] * valid[i];
      return boxBlur(ch, W, H, r);
    });
    for (let i = 0; i < valid.length; i++) {
      if (!cover[i] || valid[i] || den[i] < 0.3) continue;
      const f = Math.min(1, (den[i] - 0.3) / 0.25);
      for (let c = 0; c < 3; c++) warped[i * 3 + c] += (num[c][i] / den[i] - warped[i * 3 + c]) * f;
      wSkin[i] = Math.max(wSkin[i], f);
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
  // 고른 사진 입이 벌어져 있으면 입 안(이·혀)은 고른 사진 그대로(다문 내 입술을 늘리면 세로 줄무늬가 된다)
  if (LIPS_INNER.every((i) => i < N)) {
    const inner = fillPoly(LIPS_INNER.map((i) => dp[i]), W, H);
    const ib = boxBlur(inner, W, H, Math.max(1, Math.round(fw * 0.01)));
    for (let i = 0; i < mask.length; i++) if (ib[i] > 0) mask[i] *= 1 - Math.min(1, ib[i] * 2);
  }
  // 고개 돌린 사진에서 코 뒤로 가려진 자리(삼각형이 뒤집힘)는 내 얼굴이 접혀 얼룩이 된다 → 고른 사진 그대로
  const fb = boxBlur(fold, W, H, Math.max(1, Math.round(fw * 0.015)));
  for (let i = 0; i < mask.length; i++) if (fb[i] > 0) mask[i] *= 1 - Math.min(1, fb[i] * 3);
  const fr = Math.max(2, Math.round(fw * 0.03));
  const shrink = boxBlur(mask, W, H, fr);
  for (let i = 0; i < mask.length; i++) mask[i] = Math.max(0, Math.min(1, (shrink[i] - 0.5) * 2.2));
  mask = boxBlur(mask, W, H, Math.max(1, Math.round(fr * 0.7)));
  // 눈썹은 한쪽씩 '내 눈썹 전부' 아니면 '고른 사진 눈썹 전부': 내 앞머리·잔머리가 눈썹에 걸려 바꿀 자리에 구멍이 나면
  // 휜 내 눈썹 일부와 고른 사진 눈썹 일부가 어긋나 겹쳐 두 겹 눈썹이 된다 → 다 못 바꾸는 쪽은 고른 사진 눈썹을 그대로 둔다
  for (const ids of [[...BROW_R_LOWER, ...BROW_R_UPPER], [...BROW_L_LOWER, ...BROW_L_UPPER]]) {
    if (ids.some((i) => i >= N)) continue;
    const zone = fillPoly(ids.map((i) => dp[i]), W, H);
    let sm = 0, cnt = 0;
    for (let i = 0; i < zone.length; i++) if (zone[i] > 0.5) { sm += mask[i]; cnt++; }
    if (!cnt || sm / cnt >= 0.9) continue;
    const zb = boxBlur(zone, W, H, Math.max(2, Math.round(fw * 0.04)));
    for (let i = 0; i < mask.length; i++) mask[i] *= 1 - Math.max(0, Math.min(1, (zb[i] - 0.02) / 0.25));
  }
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
  // 크게 늘린 얼굴(내 사진 얼굴이 작음)은 피부결이 없어 매끈한 인형처럼 보인다 → 고른 사진 피부의 아주 작은 결(밝기 잔무늬)만 얹는다.
  // 결은 작은 값(±5)만: 눈·입술·눈썹 가장자리처럼 큰 차이는 빼서 고른 사진 이목구비가 비치지 않게
  if (up > 1.4) {
    const k = Math.min(1, (up - 1.4) / 1.5);
    const Ld = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) Ld[i] = 0.299 * dst[i * 4] + 0.587 * dst[i * 4 + 1] + 0.114 * dst[i * 4 + 2];
    const Lb = boxBlur(Ld, W, H, 1);
    for (let i = 0; i < W * H; i++) {
      const m = mask[i] * wt[i];
      if (m <= 0.01) continue;
      const hp = Ld[i] - Lb[i], ah = Math.abs(hp);
      const g = ah < 5 ? 1 : Math.max(0, 1 - (ah - 5) / 5);
      if (g <= 0) continue;
      const add = hp * g * k * m;
      for (let c = 0; c < 3; c++) out[i * 4 + c] = Math.max(0, Math.min(255, out[i * 4 + c] + add));
    }
  }
  return out;
}
