// 사진 속에 내 얼굴: 고른 사진(모델의 머리·화장·옷·배경)은 그대로 두고 얼굴 부분만 내 얼굴로 바꾼다.
// ① 두 얼굴의 점 468개로 삼각형 망을 만들어(들로네) 내 얼굴을 고른 사진 얼굴 모양에 맞춰 휘고
// ② 얼굴 윤곽 안쪽 중 두 사진 모두 피부·눈·코·입인 자리(정밀 얼굴 분할)만 바꾸고
//    (고른 사진의 앞머리·안경, 내 사진의 앞머리는 옮기지 않는다)
// ③ 피부끼리만 크게 흐린 두 그림의 비로 조명·피부색을 고른 사진에 맞추고 ④ 가장자리를 부드럽게 섞는다.
// 순수 계산(DOM 없음).

import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_INNER, LIPS_OUTER } from './face-regions.ts';
import { boxBlur, lab2rgb, rgb2lab } from './still-hair.ts';

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
    // 분할이 못 잡은 아주 가는 머리카락(얼굴을 가로지른 1~2px 가닥)도 틈으로: 밝기의 '어두운 능선'(헤시안 한 방향만 큼)
    // 눈·눈썹·입술·콧구멍 둘레는 이목구비 선이라 뺀다
    if (N >= 468) {
      const feat = new Float32Array(W * H);
      // 눈꺼풀(눈썹 아래 ~ 눈 위)도 뺀다: 쌍꺼풀 주름이 가는 어두운 선이라 머리카락으로 잡혀 평평한 살색 조각으로 메워졌다
      for (const ids of [[...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)], [...BROW_R_LOWER, ...BROW_R_UPPER], [...BROW_L_LOWER, ...BROW_L_UPPER], [...BROW_R_LOWER, ...[...EYE_R_UPPER].reverse()], [...BROW_L_LOWER, ...[...EYE_L_UPPER].reverse()], LIPS_OUTER, [64, 98, 240, 2, 460, 327, 294, 4]]) {
        const m = fillPoly(ids.map((i) => dp[i]), W, H);
        for (let i = 0; i < m.length; i++) if (m[i]) feat[i] = 1;
      }
      const featB = boxBlur(feat, W, H, Math.max(2, Math.round(fw0 * 0.04)));
      // 얼굴을 가로지른 가닥은 머리카락 곁에만 있다: 내 피부가 아닌 곳(머리카락)이 얼굴 폭 6% 안에 없으면 주름·이목구비 선으로 보고 그대로
      const notSkin = new Float32Array(W * H);
      for (let i = 0; i < notSkin.length; i++) notSkin[i] = cover[i] && wSkin[i] < 0.3 ? 1 : 0;
      const hairNear = boxBlur(notSkin, W, H, Math.max(3, Math.round(fw0 * 0.06)));
      const Lw = new Float32Array(W * H);
      for (let i = 0; i < Lw.length; i++) Lw[i] = 0.299 * warped[i * 3] + 0.587 * warped[i * 3 + 1] + 0.114 * warped[i * 3 + 2];
      const S = boxBlur(boxBlur(Lw, W, H, 1), W, H, 1);
      const line = new Float32Array(W * H);
      for (let y = 2; y < H - 2; y++) for (let x = 2; x < W - 2; x++) {
        const i = y * W + x;
        if (!valid[i] || featB[i] > 0 || hairNear[i] < 0.02) continue;
        const hxx = S[i + 2] - 2 * S[i] + S[i - 2], hyy = S[i + 2 * W] - 2 * S[i] + S[i - 2 * W];
        const hxy = (S[i + 2 + 2 * W] - S[i + 2 - 2 * W] - S[i - 2 + 2 * W] + S[i - 2 - 2 * W]) / 4;
        const tr = hxx + hyy, dd = Math.sqrt((hxx - hyy) ** 2 + 4 * hxy * hxy);
        const l1 = (tr + dd) / 2, l2 = (tr - dd) / 2;
        if (l1 > 8 && Math.abs(l2) < l1 * 0.35) line[i] = 1;
      }
      const lineB = boxBlur(line, W, H, 1);
      for (let i = 0; i < valid.length; i++) if (lineB[i] > 0.1) valid[i] = 0;
    }
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
  // 비를 RGB 에 곱하면 피부색이 크게 다른 두 사진(어두운 피부 → 밝은 피부, 비 2.5~3)에서 명암 대비까지 그만큼 커지고
  // 밝은 곳은 하얗게 날아가 얼굴이 얼룩덜룩한 가면이 됐다 → Lab 에서 둘레 평균을 옮긴다:
  // 색(a·b)은 평균 차이만큼 더하고, 밝기는 평균을 맞추되 대비는 (L+16) 비만큼만(0.7~1.5: 빛의 세기 비를 L 로 옮긴 몫)
  const labV = new Float32Array(W * H * 3), labD = new Float32Array(W * H * 3);
  const t3 = new Float32Array(3);
  for (let i = 0; i < W * H; i++) {
    if (mask[i] <= 0.002 && wt[i] <= 0) continue;
    const v = cover[i] ? [warped[i * 3], warped[i * 3 + 1], warped[i * 3 + 2]] : [dst[i * 4], dst[i * 4 + 1], dst[i * 4 + 2]];
    rgb2lab(Math.round(Math.max(0, Math.min(255, v[0]))), Math.round(Math.max(0, Math.min(255, v[1]))), Math.round(Math.max(0, Math.min(255, v[2]))), t3);
    labV.set(t3, i * 3);
    rgb2lab(dst[i * 4], dst[i * 4 + 1], dst[i * 4 + 2], t3);
    labD.set(t3, i * 3);
  }
  const mA: Float32Array[] = [], mB: Float32Array[] = [], gA: number[] = [], gB: number[] = [];
  for (let c = 0; c < 3; c++) {
    const a = new Float32Array(W * H), b = new Float32Array(W * H);
    let sa = 0, sb = 0, sw = 0;
    for (let i = 0; i < W * H; i++) {
      if (wt[i] <= 0) continue;
      a[i] = labV[i * 3 + c] * wt[i];
      b[i] = labD[i * 3 + c] * wt[i];
      sa += a[i]; sb += b[i]; sw += wt[i];
    }
    // 피부가 거의 없는 곳(가장자리)은 얼굴 전체 평균을 쓴다
    gA.push(sw > 0 ? sa / sw : 0); gB.push(sw > 0 ? sb / sw : 0);
    mA.push(boxBlur(a, W, H, rb)); mB.push(boxBlur(b, W, H, rb));
  }
  // 중간 크기 그늘(얼굴 폭 4~12%: 콧대 옆 그림자·한쪽 볼 그늘)은 내 사진 조명이다: 고른 사진의 같은 크기 명암으로 바꾼다.
  // 내 얼굴은 이미 고른 사진 얼굴 모양으로 휘어 있어 고른 사진 명암이 제자리(코·볼)에 놓인다. 이목구비(눈·눈썹·입술·콧구멍)는
  // 원래 어두워 조명이 아니므로 흐림에서 빼고 둘레 피부 명암으로 잰다.
  // (옆에서 센 빛을 받은 내 사진을 부드러운 빛의 사진에 넣으면 콧대 옆 검은 띠·반쪽 그늘이 얼룩처럼 남았다)
  const rs = Math.max(2, Math.round(fw * 0.025));
  const featM = new Float32Array(W * H);
  if (N >= 468) {
    for (const ids of [[...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)], [...BROW_R_LOWER, ...BROW_R_UPPER], [...BROW_L_LOWER, ...BROW_L_UPPER], LIPS_OUTER, [64, 98, 240, 2, 460, 327, 294, 4]]) {
      const m = fillPoly(ids.map((k) => dp[k]), W, H);
      for (let i = 0; i < m.length; i++) if (m[i]) featM[i] = 1;
    }
  }
  const featW = boxBlur(featM, W, H, Math.max(1, Math.round(fw * 0.015)));
  const sw2 = new Float32Array(W * H), lv = new Float32Array(W * H), ld = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) {
    sw2[i] = wt[i] * (1 - Math.min(1, featW[i] * 2));
    lv[i] = labV[i * 3] * sw2[i];
    ld[i] = labD[i * 3] * sw2[i];
  }
  const sS = boxBlur(sw2, W, H, rs), sL = boxBlur(sw2, W, H, rb);
  const vS = boxBlur(lv, W, H, rs), vL = boxBlur(lv, W, H, rb), dS = boxBlur(ld, W, H, rs), dL = boxBlur(ld, W, H, rb);
  // 내 쪽 색(2.5% 흐림)이 둘레 피부(12%)와 같은 색조·채도면 그늘, 채도가 크게 낮으면(수염 자국·짙은 눈썹 둘레) 그늘이 아니다:
  // 수염 자국까지 밝혀 푸른 잿빛 얼룩이 됐다
  const av = new Float32Array(W * H), bv = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) { av[i] = labV[i * 3 + 1] * sw2[i]; bv[i] = labV[i * 3 + 2] * sw2[i]; }
  const aS = boxBlur(av, W, H, rs), bS = boxBlur(bv, W, H, rs), aL = boxBlur(av, W, H, rb), bL = boxBlur(bv, W, H, rb);
  // 내 사진 그늘이 고른 사진보다 뚜렷이 셀 때만(옆에서 센 빛: 중간 크기 명암 표준편차 비 1.6~2.2 넘게): 보통 사진끼리는
  // 밝힌 그늘의 붉은 기(콧볼 옆)가 주황빛으로 떠서 그대로 두는 편이 자연스럽다
  let shadeOn = 0;
  {
    let n = 0, s1 = 0, s2 = 0;
    for (let i = 0; i < W * H; i += 2) {
      if (sw2[i] < 0.5 || sS[i] < 0.2 || sL[i] < 0.2) continue;
      const a1 = vS[i] / sS[i] - vL[i] / sL[i], a2 = dS[i] / sS[i] - dL[i] / sL[i];
      s1 += a1 * a1; s2 += a2 * a2; n++;
    }
    if (n > 50) shadeOn = Math.max(0, Math.min(1, (Math.sqrt(s1 / n) / Math.max(0.5, Math.sqrt(s2 / n)) - 1.6) / 0.6));
  }
  const o3 = new Uint8ClampedArray(3);
  for (let i = 0; i < W * H; i++) {
    const m = mask[i];
    if (m <= 0.002) continue;
    // 두 크기 흐림 모두 피부가 충분할 때만(가장자리·가린 곳은 지금처럼)
    let ok = shadeOn * Math.min(1, sS[i] / 0.2) * Math.min(1, sL[i] / 0.2);
    if (ok > 0) {
      const as = aS[i] / sS[i], bs = bS[i] / sS[i], al = aL[i] / sL[i], bl = bL[i] / sL[i];
      const cs = Math.hypot(as, bs), cl = Math.hypot(al, bl);
      const cosH = cs > 1e-3 && cl > 1e-3 ? (as * al + bs * bl) / (cs * cl) : 1;
      // 채도 비 0.5~0.8, 색조 차 25~45°
      ok *= Math.max(0, Math.min(1, (cs / Math.max(1e-3, cl) - 0.5) / 0.3)) * Math.max(0, Math.min(1, (cosH - 0.707) / (0.906 - 0.707)));
    }
    const devMy = ok > 0 ? vS[i] / sS[i] - vL[i] / sL[i] : 0, devRef = ok > 0 ? dS[i] / sS[i] - dL[i] / sL[i] : 0;
    const t = Math.min(1, wb[i] / 0.15);
    const ma = [0, 1, 2].map((c) => t * (mA[c][i] / (wb[i] + 1e-6)) + (1 - t) * gA[c]);
    const mb = [0, 1, 2].map((c) => t * (mB[c][i] / (wb[i] + 1e-6)) + (1 - t) * gB[c]);
    const s = Math.max(0.7, Math.min(1.5, (mb[0] + 16) / (ma[0] + 16)));
    const L0 = mb[0] + (labV[i * 3] - ma[0]) * s;
    const L = L0 + (devRef - devMy * s) * ok;
    // 그늘을 밝힌 만큼 색도도 같이(그늘진 피부는 색도가 낮다): 밝기만 올리면 어두운 피부가 잿빛 얼룩이 됐다
    const kc = Math.max(0.9, Math.min(1.1, (Math.max(0, L) + 16) / (Math.max(0, L0) + 16)));
    lab2rgb(Math.max(0, Math.min(100, L)), (labV[i * 3 + 1] + mb[1] - ma[1]) * kc, (labV[i * 3 + 2] + mb[2] - ma[2]) * kc, o3, 0);
    for (let c = 0; c < 3; c++) {
      const d0 = dst[i * 4 + c];
      out[i * 4 + c] = d0 + (o3[c] - d0) * m;
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
