// 사진 모드 염색: 고른 사진의 '실제 머리카락 색 분포'를 내 머리카락에 올(픽셀)마다 옮긴다.
// 실시간 염색(셰이더)은 목표색 하나에 밝기 비율을 곱하는 방식이라 큰 밝기 변화(금발 → 흑발)에서 페인트처럼 평평해진다.
// 사진은 한 번만 계산하면 되므로:
//  ① 정밀 얼굴 분할(face-parse) 확률을 원본 해상도로 키우고 내 사진 밝기로 가이디드 필터를 걸어 머리카락 알파(잔머리까지)를 만든다
//  ② 내 머리카락 밝기의 순위(분위)를 고른 사진 머리카락의 같은 분위 밝기·색(Lab)으로 바꾼다
//     → 내 올의 명암 순서(결)는 그대로, 색·윤기·그늘의 분포는 고른 사진 그대로
// 순수 계산 함수(DOM 없음)라 단위 테스트로 확인한다.

export const Q = 64; // 분위 수
const TRIM = Q * 0.03; // 고른 사진 분포에서 버릴 양 끝 비율

export interface HairDist {
  /** 분위 0..Q 의 L(0~100), a, b */
  L: Float32Array;
  a: Float32Array;
  b: Float32Array;
  n: number;
}

const lin = (c: number): number => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const gam = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const LIN = new Float32Array(256);
for (let i = 0; i < 256; i++) LIN[i] = lin(i / 255);

export function rgb2lab(r: number, g: number, b: number, out: Float32Array): void {
  const R = LIN[r], G = LIN[g], B = LIN[b];
  const x = (0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047;
  const y = 0.2126 * R + 0.7152 * G + 0.0722 * B;
  const z = (0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883;
  const f = (t: number): number => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f(x), fy = f(y), fz = f(z);
  out[0] = 116 * fy - 16;
  out[1] = 500 * (fx - fy);
  out[2] = 200 * (fy - fz);
}

export function lab2rgb(L: number, a: number, b: number, out: Uint8ClampedArray, o: number): void {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const inv = (t: number): number => (t * t * t > 0.008856 ? t * t * t : (t - 16 / 116) / 7.787);
  const x = inv(fx) * 0.95047, y = inv(fy), z = inv(fz) * 1.08883;
  const R = 3.2406 * x - 1.5372 * y - 0.4986 * z;
  const G = -0.9689 * x + 1.8758 * y + 0.0415 * z;
  const B = 0.0557 * x - 0.204 * y + 1.057 * z;
  out[o] = 255 * gam(Math.max(0, Math.min(1, R)));
  out[o + 1] = 255 * gam(Math.max(0, Math.min(1, G)));
  out[o + 2] = 255 * gam(Math.max(0, Math.min(1, B)));
}

/** 머리카락 픽셀(알파 ≥ thr)의 밝기 분위별 L·a·b. 픽셀이 너무 적으면 null */
export function hairDist(rgba: Uint8ClampedArray, alpha: ArrayLike<number>, thr = 0.6, minN = 300): HairDist | null {
  const lab = new Float32Array(3);
  const Ls: number[] = [];
  const As: number[] = [];
  const Bs: number[] = [];
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] < thr) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    Ls.push(lab[0]);
    As.push(lab[1]);
    Bs.push(lab[2]);
  }
  const n = Ls.length;
  if (n < minN) return null;
  const idx = Array.from({ length: n }, (_, i) => i).sort((p, q) => Ls[p] - Ls[q]);
  const L = new Float32Array(Q + 1), a = new Float32Array(Q + 1), b = new Float32Array(Q + 1);
  // 분위마다: L 은 그 분위 값, a·b 는 그 분위 주변(±1/Q) 평균(한 픽셀 색 잡음에 흔들리지 않게)
  for (let k = 0; k <= Q; k++) {
    const c = Math.min(n - 1, Math.round((k / Q) * (n - 1)));
    L[k] = Ls[idx[c]];
    const lo = Math.max(0, Math.floor(((k - 1) / Q) * (n - 1))), hi = Math.min(n - 1, Math.ceil(((k + 1) / Q) * (n - 1)));
    let sa = 0, sb = 0, sc = 0;
    for (let j = lo; j <= hi; j++) {
      sa += As[idx[j]];
      sb += Bs[idx[j]];
      sc += Math.hypot(As[idx[j]], Bs[idx[j]]);
    }
    const m = hi - lo + 1;
    a[k] = sa / m;
    b[k] = sb / m;
    // 같은 밝기에 색이 다른 가닥이 섞인 머리(라임 + 청록)는 평균하면 탁한 올리브가 된다 → 색상은 평균, 채도는 픽셀 평균 채도로(최대 1.6배).
    // 색이 거의 없는 머리(은발·검정, 채도 8 아래)는 잡음 채도가 붙지 않게 그대로
    const cm = Math.hypot(a[k], b[k]), cs = sc / m;
    if (cs > 8 && cm > 1) {
      const f = Math.min(1.6, cs / cm);
      a[k] *= f;
      b[k] *= f;
    }
  }
  return { L, a, b, n };
}

/**
 * 머리카락을 위(뿌리)→아래(끝) 세 띠로 나눈 분포. 투톤·옴브레(뿌리와 끝 색이 다른 머리)를 띠마다 옮기려고.
 * y0·y1: 머리카락 세로 범위(2%·98% 자리). 띠가 비면 null 이 아니라 전체 분포로 채운다.
 */
export interface HairBands { y0: number; y1: number; bands: HairDist[]; ombre: boolean; lowSpread: number }
/** 머리카락 세로 범위(알파 ≥ thr 인 줄의 2%·98% 자리) */
export function hairSpan(alpha: ArrayLike<number>, w: number, thr = 0.6): { y0: number; y1: number } | null {
  const ys: number[] = [];
  for (let i = 0; i < alpha.length; i++) if (alpha[i] >= thr) ys.push(Math.floor(i / w));
  if (ys.length < 10) return null;
  ys.sort((p, q) => p - q);
  const y0 = ys[Math.floor(ys.length * 0.02)];
  return { y0, y1: Math.max(y0 + 1, ys[Math.floor(ys.length * 0.98)]) };
}
export function hairBands(rgba: Uint8ClampedArray, w: number, alpha: ArrayLike<number>, thr = 0.6, minN = 300): HairBands | null {
  const all = hairDist(rgba, alpha, thr, minN);
  const span = hairSpan(alpha, w, thr);
  if (!all || !span) return null;
  const { y0, y1 } = span;
  const bands: HairDist[] = [];
  for (let k = 0; k < 3; k++) {
    // 띠는 서로 조금 겹친다(경계 잡음 줄이기)
    const t0 = k / 3 - 0.05, t1 = (k + 1) / 3 + 0.05;
    const m = new Float32Array(alpha.length);
    for (let i = 0; i < alpha.length; i++) {
      const t = (Math.floor(i / w) - y0) / (y1 - y0);
      if (t >= t0 && t < t1) m[i] = alpha[i];
    }
    bands.push(hairDist(rgba, m, thr, Math.max(60, Math.floor(minN / 4))) ?? all);
  }
  // 뿌리 띠와 끝 띠의 색이 크게 다르면(ΔE > 12) 투톤·옴브레로 본다. 밝기는 30% 자리로 잰다:
  // 검은 머리 정수리의 윤기(밝은 반사)·시스루 앞머리는 뿌리 띠의 밝은 쪽만 올려 가운데 값으로 재면 옴브레로 잘못 읽힌다
  const mid = (d: HairDist) => [d.L[Math.round(Q * 0.3)], d.a[Q / 2], d.b[Q / 2]];
  const [A, B] = [mid(bands[0]), mid(bands[2])];
  const ombre = Math.hypot(A[0] - B[0], A[1] - B[1], A[2] - B[2]) > 12;
  // 하이라이트 판단용 밝기 폭: 윤기가 몰리는 뿌리 띠는 빼고 가운데·끝 띠로
  const spread = (d: HairDist) => d.L[Math.round(Q * 0.9)] - d.L[Math.round(Q * 0.1)];
  return { y0, y1, bands, ombre, lowSpread: Math.max(spread(bands[1]), spread(bands[2])) };
}

/**
 * 고른 사진 분포의 어두운 꼬리 다듬기: 머리 분할에 섞인 검은 옷·그늘이 아래 몇 %를 거의 검정으로 만들면,
 * 밝은 머리로 바꿀 때 내 머리의 어두운 곳이 검은 얼룩이 된다 → 10% 자리에서 (50%−10%)의 1/4 넘게 내려간 값은 거기서 멈추고 색도 10% 색으로.
 * 원래 어두운 바탕(하이라이트 머리)은 10% 자리 자체가 어두워 거의 안 바뀐다.
 */
export function trimDarkTail(d: HairDist): HairDist {
  const k10 = Math.round(Q * 0.1);
  const floor = d.L[k10] - (d.L[Q / 2] - d.L[k10]) * 0.25;
  const L = Float32Array.from(d.L), a = Float32Array.from(d.a), b = Float32Array.from(d.b);
  for (let k = 0; k < k10; k++) {
    if (L[k] >= floor) continue;
    const t = Math.min(1, (floor - L[k]) / Math.max(1, floor));
    L[k] = floor;
    a[k] = a[k] * (1 - t) + d.a[k10] * t;
    b[k] = b[k] * (1 - t) + d.b[k10] * t;
  }
  return { L, a, b, n: d.n };
}

/** 정렬된 분위표 qs 에서 값 v 의 분위 위치(0..Q, 실수) */
export function quantilePos(qs: Float32Array, v: number): number {
  // 같은 값이 여러 분위에 걸쳐 있으면(눌린 검정·날아간 흰색) 그 묶음의 가운데로: 맨 끝으로 몰리면 얼룩이 된다
  if (v <= qs[0] || v >= qs[Q] || qs[0] === qs[1] || qs[Q - 1] === qs[Q]) {
    let lo = 0;
    while (lo <= Q && qs[lo] < v) lo++;
    let hi = Q;
    while (hi >= 0 && qs[hi] > v) hi--;
    if (hi >= lo) return (lo + hi) / 2;
    if (v <= qs[0]) return 0;
    if (v >= qs[Q]) return Q;
  }
  let lo = 0, hi = Q;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (qs[m] <= v) lo = m;
    else hi = m;
  }
  const d = qs[hi] - qs[lo];
  return lo + (d > 1e-6 ? (v - qs[lo]) / d : 0.5);
}

const at = (q: Float32Array, t: number): number => {
  const i = Math.min(Q - 1, Math.floor(t));
  const f = t - i;
  return q[i] * (1 - f) + q[i + 1] * f;
};

/**
 * 내 사진의 머리카락을 고른 사진 분포로 바꾼 새 그림(RGBA). alpha: 0~1(픽셀마다), amount: 진하기(0~1.6, 1 이 사진 그대로).
 * 결 보존: 분위 대응만 쓰면 고른 사진 머리의 밝기 폭이 좁을 때(검은 머리) 결이 눌린다 → 내 사진의 잔결(주변과의 차)을 조금 되살린다.
 */
export function recolorHair(
  rgba: Uint8ClampedArray, w: number, h: number, alpha: Float32Array, mine: HairDist, ref: HairDist, amount = 1, seed = 1,
  bands: { ref: HairBands; y0: number; y1: number } | null = null, hlSpread: number | null = null,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba);
  const lab = new Float32Array(3);
  const tmp = new Uint8ClampedArray(3);
  ref = trimDarkTail(ref);
  if (bands) bands = { ...bands, ref: { ...bands.ref, bands: bands.ref.bands.map(trimDarkTail) } };
  // 잔결용 주변 밝기(가로·세로 3px 상자 평균): 루마 근사로 충분
  const luma = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) luma[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2];
  const box = boxBlur(luma, w, h, 3);
  const mineRange = Math.max(1, mine.L[Math.round(Q * 0.9)] - mine.L[Math.round(Q * 0.1)]);
  const refRange = Math.max(1, ref.L[Math.round(Q * 0.9)] - ref.L[Math.round(Q * 0.1)]);
  // 고른 사진 폭이 내 사진보다 좁을수록 잔결을 더 되살린다(최대 0.5)
  const detail = Math.max(0, Math.min(0.5, 0.5 * (1 - refRange / mineRange)));
  const amt = Math.max(0, Math.min(1, amount));
  // 어두운 머리를 훨씬 밝게(탈색처럼) 할 때는 내 그늘이 고른 사진의 가장 어두운 쪽(그림자)으로 몰려 얼룩진다
  // → 밝게 올릴수록 아래쪽 분위를 들어 올린다(실제 탈색 머리는 그늘도 밝다)
  const ratio = (ref.L[Q / 2] + 2) / (mine.L[Q / 2] + 2);
  const lift = Math.max(0, Math.min(1, (ratio - 1.8) / 2.2));
  const g = 1 - 0.45 * lift;
  // 크게 밝힐 때: ① 아주 어두운 곳은 JPEG 네모 무늬가 그대로 얼룩이 되므로 주변 밝기로 대응 위치를 정하고
  // ② 원래 결이 거의 없는(검게 뭉개진) 곳에는 머리 흐름을 따라 그린 가는 올 무늬를 더한다
  // 어두운 픽셀은 점처럼 흩어진 잡음이라 넓게(얼굴 크기 사진에서 5px 안팎) 흐린 밝기로 순위를 정한다
  const blur2 = lift > 0 ? boxBlur(luma, w, h, Math.max(2, Math.round(Math.min(w, h) / 150))) : luma;
  const fine = lift > 0 ? boxBlur(luma, w, h, 1) : null;
  const mineDark = mine.L[Math.round(Q * 0.15)], mineMid = mine.L[Q / 2];
  const strands = lift > 0.05 ? strandTexture(luma, w, h, alpha, seed) : null;
  // 거의 검은 머리(머리 속 루마 24 아래가 절반 넘음: 아프로 69%): 어두운 픽셀 순위를 둘레 평균으로 메우면 곱슬 결이 통째로
  // 지워져 물감처럼 평평한 얼룩이 된다 → 메우기는 정말 새까만 곳(루마 12 아래)만, 대신 중간 크기(2px/8px) 결을 비로 되살린다
  let darkFrac = 0;
  if (lift > 0) {
    let nc = 0, nd = 0;
    for (let i = 0; i < w * h; i++) if (alpha[i] >= 0.9) { nc++; if (luma[i] < 24) nd++; }
    darkFrac = nc ? nd / nc : 0;
  }
  const mostlyDark = Math.max(0, Math.min(1, (darkFrac - 0.4) / 0.2)) * (lift > 0 ? 1 : 0);
  const mid2 = mostlyDark > 0 ? boxBlur(luma, w, h, 2) : null, mid8 = mostlyDark > 0 ? boxBlur(luma, w, h, Math.max(6, Math.round(Math.min(w, h) / 100))) : null;
  const local = lift > 0.05 ? boxBlur(luma.map((v, i) => (v - box[i]) ** 2), w, h, 3) : null;
  const refSpread = ref.L[Math.round(Q * 0.9)] - ref.L[Math.round(Q * 0.1)];
  // 하이라이트·발레아쥬(어두운 바탕에 밝은 가닥): 고른 사진 밝기 폭이 넓고 내 머리는 고르면, 분위만 옮기면 한 색으로 평평해진다
  // → 머리 흐름을 따라 길게 문지른 굵은 '리본' 무늬를 순위에 섞어 밝은 분위가 가닥 모양으로 모이게 한다
  // 내 머리 밝기 폭은 가닥 사이로 비친 배경까지 섞여 넓게 잡히므로 쓰지 않는다: 고른 사진 폭만 본다(17번 하이라이트 사진 폭 ≈ 48, 단색 염색 ≈ 25~35)
  // hlSpread: 뿌리 띠(정수리 윤기)를 뺀 밝기 폭(있으면). 전체 폭과 둘 중 작은 쪽:
  // 검은 머리 윤기는 전체 폭만, 사진 위 글씨·검은 옷은 아래 띠 폭만 키운다(진짜 하이라이트는 둘 다 넓다)
  const hl = Math.max(0, Math.min(1, (Math.min(refSpread, hlSpread ?? refSpread) - 32) / 14));
  // 리본 굵기·길이는 머리 크기(알파 상자 큰 변)에 맞춘다: 사진 크기로 정하면 큰 사진의 짧은 단발에서 리본이 머리에 비해
  // 너무 굵고 길어 정수리에 하이라이트 색의 넓고 밋밋한 띠가 됐다
  let hx0 = w, hx1 = 0, hy0 = h, hy1 = 0;
  if (hl > 0.05) for (let i = 0; i < w * h; i++) if (alpha[i] > 0.5) { const x = i % w, y = (i / w) | 0; if (x < hx0) hx0 = x; if (x > hx1) hx1 = x; if (y < hy0) hy0 = y; if (y > hy1) hy1 = y; }
  const hairSz = Math.min(Math.min(w, h), Math.max(hx1 - hx0, hy1 - hy0, 1));
  const ribbons = hl > 0.05 ? strandTexture(luma, w, h, alpha, seed + 7, { width: Math.max(1, Math.round(hairSz / 150)), length: hairSz / 4 }) : null;
  const wr = 0.85 * hl;
  const blackRef = Math.max(0, Math.min(1, (28 - ref.L[Q / 2]) / 6));
  // ① 픽셀마다 순위(0~1)
  const U = new Float32Array(w * h).fill(-1);
  // 내 머리 밝기 분포 위쪽(97%)보다 훨씬 밝은 픽셀은 머리카락이 아니라 가르마 두피·배경일 때가 많다(알파 줄일 몫, 0~1)
  const over = new Float32Array(w * h);
  const qHi = mine.L[Math.round(Q * 0.97)], qMid = mine.L[Q / 2];
  for (let i = 0; i < w * h; i++) {
    if (alpha[i] * amt < 0.004) continue;
    const o = i * 4;
    rgb2lab(rgba[o], rgba[o + 1], rgba[o + 2], lab);
    let Lsrc = lab[0];
    if (lift > 0) {
      // 어두울수록 주변(5px) 밝기를 더 섞는다: 루마 차이만큼 L 을 옮기는 근사
      const dark = 1 - Math.max(0, Math.min(1, (Lsrc - mineDark) / Math.max(1, mineMid - mineDark)));
      Lsrc += ((blur2[i] - luma[i]) / 255) * 100 * dark * lift;
    }
    U[i] = quantilePos(mine.L, Lsrc) / Q;
    if (lift > 0) over[i] = Math.max(0, Math.min(1, ((lab[0] - qHi) / Math.max(4, qHi - qMid) - 0.5) / 1));
    if (ribbons) {
      // 리본 위쪽 일부만 밝은 가닥으로(위로만 민다: 어둡게 누르면 단색 머리에 검은 얼룩이 생긴다)
      const r = Math.max(0, Math.min(1, (ribbons[i] - 0.15) / 0.4));
      const sm = r * r * (3 - 2 * r);
      // 리본 안을 가는 올 무늬로 끊는다: 한 덩어리로 밝히면 물감으로 칠한 띠처럼 평평하다(실제 하이라이트는 밝은 올 묶음)
      const tex = strands ? Math.max(0.3, Math.min(1, 0.65 + strands[i] * 0.7)) : 1;
      U[i] += wr * sm * tex * (1 - U[i]);
    }
  }
  // 배경에 닿은 가장자리(알파 0.3 아래가 2px 안)는 배경이 섞여 제 밝기 순위가 부풀어 있다(알파는 높아도):
  // 둘레 머리 속 순위 +0.1 까지만(검은 짧은 머리를 밝게 염색하면 머리 둘레가 가장 밝은 머리색 = 하얀 점선이 됐다).
  // 아래 새까만 픽셀 메우기보다 먼저: 메우기가 부푼 가장자리 순위를 빌려 가면 테두리 안쪽 검정까지 하얘진다
  {
    const bgm = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) if (alpha[i] < 0.3) bgm[i] = 1;
    const nearBg = boxBlur(bgm, w, h, 2);
    const cm = new Float32Array(w * h), cu = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) if (U[i] >= 0 && nearBg[i] === 0 && luma[i] >= 24) { cm[i] = 1; cu[i] = U[i]; }
    const r = Math.max(3, Math.round(Math.min(w, h) / 80));
    const cd = boxBlur(cm, w, h, r), cn = boxBlur(cu, w, h, r);
    for (let i = 0; i < w * h; i++) {
      if (U[i] < 0 || nearBg[i] === 0 || cd[i] < 0.02) continue;
      const cap = cn[i] / cd[i] + 0.1;
      if (U[i] > cap) U[i] = cap;
    }
  }
  // ①' 아주 어두운 곳(원본 루마 24 아래: 정보가 거의 없는 검정)은 가장 어두운 분위로 몰려 밝게 바꿀 때 검은 점·얼룩이 된다
  //     → 둘레의 밝기 정보가 있는 머리카락(루마 24 이상) 순위를 넓게 가져와, 어두울수록 더 섞는다(루마 8 이하는 전부).
  //     딱 잘라 바꾸면 경계 바로 위 픽셀들이 맨 아래 순위가 되어 또 점이 된다
  {
    const ok = new Float32Array(w * h), uv = new Float32Array(w * h);
    let dark = 0;
    for (let i = 0; i < w * h; i++) {
      if (U[i] < 0) continue;
      if (luma[i] < 24) dark++;
      else { ok[i] = 1; uv[i] = U[i]; }
    }
    if (dark > 0) {
      const r = Math.max(4, Math.round(Math.min(w, h) / 60));
      const num = boxBlur(uv, w, h, r), den = boxBlur(ok, w, h, r);
      // 넓게 다 어두운 곳은 4배 넓은 둘레에서, 그래도 없으면 가운데 순위
      const num2 = boxBlur(uv, w, h, r * 4), den2 = boxBlur(ok, w, h, r * 4);
      const th = 24 - 12 * mostlyDark;
      for (let i = 0; i < w * h; i++) {
        if (U[i] < 0 || luma[i] >= th) continue;
        const k = Math.max(0, Math.min(1, (th - luma[i]) / (th * 2 / 3)));
        const near = den[i] >= 0.02 ? num[i] / den[i] : den2[i] >= 0.005 ? num2[i] / den2[i] : 0.5;
        const fill = Math.max(U[i], near * 0.85);
        U[i] += (fill - U[i]) * k;
      }
    }
  }
  // (리본은 순위를 위로만 미므로 다시 고르게 펴지 않는다: 펴면 가장 어두운 쪽이 다른 픽셀로 바뀌어 검은 점이 생긴다)
  // 머리카락 안쪽(둘레 알파가 꽉 찬 곳)의 어두운 픽셀은 알파가 덜 차도 원래 검정이 비쳐 얼룩이 된다 → 둘레 알파로 채운다
  const aBlur = boxBlur(alpha, w, h, 3);
  // 가장자리(알파 1 아래)는 머리카락과 배경이 섞인 픽셀: 픽셀 전체를 새 색 쪽으로 알파만큼 당기면 배경 몫까지 덜어져
  // 원래 머리색이 c(1−c)(배경−머리)만큼 남는다(검은 머리를 밝게 하면 둘레가 검은 선) → 머리 몫만 바꾼다:
  // 원본 + 알파 × (새 색 − 둘레 속머리 색). 속머리 색은 알파 0.8 이상 픽셀의 둘레 평균
  const coreW = new Float32Array(w * h), cR = new Float32Array(w * h), cG = new Float32Array(w * h), cB = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) if (alpha[i] >= 0.8) { coreW[i] = 1; cR[i] = rgba[i * 4]; cG[i] = rgba[i * 4 + 1]; cB[i] = rgba[i * 4 + 2]; }
  const er = Math.max(3, Math.round(Math.min(w, h) / 80));
  const cWb = boxBlur(coreW, w, h, er), cRb = boxBlur(cR, w, h, er), cGb = boxBlur(cG, w, h, er), cBb = boxBlur(cB, w, h, er);
  for (let i = 0; i < w * h; i++) {
    const a0 = luma[i] < 24 && aBlur[i] > 0.85 ? Math.max(alpha[i], aBlur[i]) : alpha[i];
    // 크게 밝힐 때 가르마 두피처럼 내 머리보다 훨씬 밝은 곳은 가장 밝은 머리색으로 바뀌어 하얀 솜털 얼룩이 된다 → 덜 물들인다
    // 배경이 섞인 가장자리(ew)는 밝은 게 배경 몫이라 덜 물들이지 않는다(아래에서 머리 몫만 바꾸고 새 색 안으로 묶는다)
    const ew = cWb[i] >= 0.02 ? Math.max(0, Math.min(1, (1 - a0) / 0.5)) : 0;
    const al = a0 * amt * (1 - 0.8 * over[i] * (1 - ew));
    if (al < 0.004 || U[i] < 0) continue;
    const o = i * 4;
    // 양 끝 분위(그림자·배경·반사가 섞인 곳)는 버리고 3%~97% 사이로 대응시킨다(검은 점·흰 점 방지)
    const t = TRIM + Math.pow(Math.max(0, Math.min(1, U[i])), g) * (Q - 2 * TRIM);
    // 투톤·옴브레: 내 머리 세로 위치(뿌리 0 → 끝 1)에 맞는 띠 분포를 섞어 쓴다(띠 가운데 1/6·1/2·5/6)
    let rL: Float32Array = ref.L, ra: Float32Array = ref.a, rb: Float32Array = ref.b, mix: [HairDist, HairDist, number] | null = null;
    if (bands) {
      const ty = Math.max(0, Math.min(1, (Math.floor(i / w) - bands.y0) / Math.max(1, bands.y1 - bands.y0)));
      const f = Math.max(0, Math.min(2, (ty - 1 / 6) * 3));
      const k = Math.min(1, Math.floor(f));
      mix = [bands.ref.bands[k], bands.ref.bands[k + 1], f - k];
      rL = mix[0].L;
    }
    const bandAt = (key: 'L' | 'a' | 'b'): number => (mix ? at(mix[0][key], t) * (1 - mix[2]) + at(mix[1][key], t) * mix[2] : at(key === 'L' ? rL : key === 'a' ? ra : rb, t));
    let L = bandAt('L');
    if (strands && local) {
      // 원래 잔결이 약한 곳일수록(표준편차 루마 3 아래) 그린 올 무늬를 세게
      const weak = 1 - Math.max(0, Math.min(1, (Math.sqrt(local[i]) - 1.5) / 4));
      // 그린 올 무늬도 어두운 쪽은 반만(검은 점처럼 보이지 않게)
      const sv = strands[i] * refSpread * 0.28 * lift * weak;
      L += sv < 0 ? sv * 0.5 : sv;
    }
    // 잔결: 내 픽셀이 주변보다 밝거나 어두운 정도(L 단위 근사)
    // 크게 밝힐 때 원본 어두운 픽셀의 잡음(JPEG)이 그대로 검은 점이 되므로 살짝 흐린 밝기로 재고, 어둡게 하는 쪽은 4 까지만
    const dl = (((lift > 0 ? blur2[i] : luma[i]) - box[i]) / 255) * 100 * detail;
    L += lift > 0 ? Math.max(-4, dl) : dl;
    if (fine) {
      // 크게 밝힐 때 올 결: 어두운 머리의 올 대비는 밝기 '차'로는 아주 작아 밝힌 뒤 평평한 물감처럼 된다
      // → 1px 흐린 밝기 / 3px 상자 평균의 '비'로 올 결을 되살린다(어둡게 하는 쪽은 60%: 검은 점 방지). 정보 없는 새까만 곳(루마 16 아래)은 빼고
      const rr = Math.max(-0.2, Math.min(0.12, Math.log((fine[i] + 6) / (box[i] + 6))));
      const info = Math.max(0, Math.min(1, (luma[i] - 16) / 16));
      L *= Math.exp((rr < 0 ? rr * 0.6 : rr) * 0.6 * lift * info);
    }
    if (mid2 && mid8) {
      const r2 = Math.max(-0.25, Math.min(0.2, Math.log((mid2[i] + 6) / (mid8[i] + 6))));
      const info2 = Math.max(0, Math.min(1, (luma[i] - 6) / 12));
      L *= Math.exp(r2 * 0.45 * lift * info2 * mostlyDark);
    }
    // 흑발 사진(가운데 밝기 L 25 아래)으로 바꿀 때는 원래보다 밝게 하지 않는다(검은 염색은 머리를 밝히지 못한다):
    // 검은 배경에 번진 알파까지 내 머리 분포에 섞이면 내 검은 머리가 분포 위쪽 = 고른 사진 윤기 밝기로 대응돼 회색 덩어리가 됐다
    if (blackRef > 0) {
      rgb2lab(rgba[o], rgba[o + 1], rgba[o + 2], lab);
      if (L > lab[0] + 4) L -= (L - lab[0] - 4) * blackRef;
    }
    lab2rgb(Math.max(0, Math.min(100, L)), bandAt('a'), bandAt('b'), tmp, 0);
    // 알파 0.5 아래는 둘레 속머리 색을 전부, 1 에 가까울수록 제 색을(안쪽은 지금과 같다).
    // 결과는 원본과 새 색 사이로 묶는다(알파가 부푼 배경 픽셀이 새 색보다 밝아져 하얀 테가 되지 않게)
    for (let c = 0; c < 3; c++) {
      const hc = ew > 0 ? rgba[o + c] + ((c === 0 ? cRb[i] : c === 1 ? cGb[i] : cBb[i]) / cWb[i] - rgba[o + c]) * ew : rgba[o + c];
      const v = rgba[o + c] + (tmp[c] - hc) * al;
      out[o + c] = Math.max(Math.min(rgba[o + c], tmp[c]), Math.min(Math.max(rgba[o + c], tmp[c]), v));
    }
  }
  return out;
}

/** 적분 영상 상자 평균(반지름 r) */
export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) {
      row += src[y * w + x];
      I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row;
    }
  }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const s = I[y1 * (w + 1) + x1] - I[y0 * (w + 1) + x1] - I[y1 * (w + 1) + x0] + I[y0 * (w + 1) + x0];
      out[y * w + x] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

/**
 * 가이디드 필터(흑백 안내 영상): 거친 확률 p 를 내 사진의 경계(머리카락 올)에 맞춰 다듬는다. guide·p 는 0~1.
 */
export function guidedFilter(guide: Float32Array, p: Float32Array, w: number, h: number, r: number, eps: number): Float32Array {
  const n = w * h;
  const Ip = new Float32Array(n), II = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    Ip[i] = guide[i] * p[i];
    II[i] = guide[i] * guide[i];
  }
  const mI = boxBlur(guide, w, h, r), mp = boxBlur(p, w, h, r), mIp = boxBlur(Ip, w, h, r), mII = boxBlur(II, w, h, r);
  const A = new Float32Array(n), B = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = mII[i] - mI[i] * mI[i];
    A[i] = (mIp[i] - mI[i] * mp[i]) / (v + eps);
    B[i] = mp[i] - A[i] * mI[i];
  }
  const mA = boxBlur(A, w, h, r), mB = boxBlur(B, w, h, r);
  const q = new Float32Array(n);
  for (let i = 0; i < n; i++) q[i] = Math.max(0, Math.min(1, mA[i] * guide[i] + mB[i]));
  return q;
}

/** 회색조 닫기(팽창 후 침식, 반지름 r): 알파 안의 작은 구멍(아주 어두운 머리카락 덩어리)을 메운다 */
/**
 * 머리 속(둘레 반지름 r 안이 거의 다 머리카락: 알파 0.35 이상 92~99.5%)은 알파를 1 쪽으로 채운다(제자리 수정).
 * 머리를 뒤로 넘긴 머리처럼 분할이 덜 확신하면 머리 속 알파가 0.85 에 머물고 정수리 한쪽은 0.5 까지 떨어져,
 * 밝게 염색하면 원래 갈색이 비쳐 전체가 탁하고 가르마 옆이 회색 얼룩이 됐다. 가장자리(둘레에 배경)는 그대로
 */
export function fillInterior(a: Float32Array, w: number, h: number, r: number): void {
  const m = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) m[i] = a[i] >= 0.35 ? 1 : 0;
  const d = boxBlur(m, w, h, r);
  for (let i = 0; i < a.length; i++) {
    const k = Math.max(0, Math.min(1, (d[i] - 0.92) / 0.075));
    if (k > 0 && a[i] >= 0.35) a[i] += (1 - a[i]) * k;
  }
}

export function closeHoles(a: Float32Array, w: number, h: number, r: number): Float32Array {
  const pass = (src: Float32Array, horiz: boolean, max: boolean): Float32Array => {
    const out = new Float32Array(src.length);
    const n = horiz ? w : h;
    const m = horiz ? h : w;
    for (let j = 0; j < m; j++) {
      for (let i = 0; i < n; i++) {
        let v = max ? 0 : 1;
        for (let k = Math.max(0, i - r); k <= Math.min(n - 1, i + r); k++) {
          const s = horiz ? src[j * w + k] : src[k * w + j];
          v = max ? Math.max(v, s) : Math.min(v, s);
        }
        out[horiz ? j * w + i : i * w + j] = v;
      }
    }
    return out;
  };
  const d = pass(pass(a, true, true), false, true);
  return pass(pass(d, true, false), false, false);
}


/**
 * 머리 흐름을 따라 늘인 가는 올 무늬(−1~1 쯤, 평균 0): 잡음을 머리카락 방향으로 평균 내는 선적분 컨볼루션(LIC).
 * 방향은 밝기 기울기의 구조 텐서로 구하고(올과 수직인 기울기 → 올 방향), 결이 없는 곳(방향이 흐린 곳)은
 * 정수리에서 바깥으로 퍼지는 방향을 쓴다. alpha 가 있는 곳만 계산한다.
 */
export function strandTexture(
  luma: Float32Array, w: number, h: number, alpha: Float32Array, seed = 1,
  opt: { width?: number; length?: number } = {},
): Float32Array {
  const n = w * h;
  const gx = new Float32Array(n), gy = new Float32Array(n);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    const i = y * w + x;
    gx[i] = luma[i + 1] - luma[i - 1];
    gy[i] = luma[i + w] - luma[i - w];
  }
  const r = Math.max(3, Math.round(Math.min(w, h) / 120));
  const j11 = boxBlur(gx.map((v) => v * v), w, h, r);
  const j22 = boxBlur(gy.map((v) => v * v), w, h, r);
  const j12 = boxBlur(gx.map((v, i) => v * gy[i]), w, h, r);
  // 정수리: 머리카락 상자 위쪽 가운데
  let x0 = w, x1 = 0, y0 = h;
  for (let i = 0; i < n; i++) if (alpha[i] > 0.5) { const x = i % w, y = (i / w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; }
  const crownX = (x0 + x1) / 2, crownY = y0 + (x1 - x0) * 0.15;
  const cohOf = (i: number): number => {
    const a = j11[i], b = j22[i], c = j12[i], tr = a + b;
    return tr > 1e-3 ? Math.sqrt((a - b) * (a - b) + 4 * c * c) / tr : 0;
  };
  // 곱슬(아프로·컬): 머리 속 결 방향이 흩어진(일관성 0.2 아래) 픽셀이 많다(아프로 46%, 생머리·검은 짧은 머리 1~10%)
  // → 정수리에서 뻗는 방향을 대신 쓰면 사방으로 뻗은 직선 줄무늬가 된다. 제 자리 방향으로 짧게만 문지른다
  let nCore = 0, nLow = 0;
  for (let i = 0; i < n; i++) if (alpha[i] >= 0.9) { nCore++; if (cohOf(i) < 0.2) nLow++; }
  const coily = nCore > 0 ? Math.max(0, Math.min(1, (nLow / nCore - 0.2) / 0.15)) : 0;
  const dx = new Float32Array(n), dy = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (alpha[i] < 0.01) continue;
    const a = j11[i], b = j22[i], c = j12[i];
    const coh = cohOf(i);
    // 기울기 주방향 θ, 올 방향은 그에 수직
    const th = 0.5 * Math.atan2(2 * c, a - b) + Math.PI / 2;
    let ux = Math.cos(th), uy = Math.sin(th);
    const px = (i % w) - crownX, py = ((i / w) | 0) - crownY;
    const pl = Math.hypot(px, py) || 1;
    const fx = px / pl, fy = py / pl;
    if (ux * fx + uy * fy < 0) { ux = -ux; uy = -uy; }
    const k = Math.max(coily, Math.min(1, (coh - 0.2) / 0.4));
    const vx = ux * k + fx * (1 - k), vy = uy * k + fy * (1 - k);
    const vl = Math.hypot(vx, vy) || 1;
    dx[i] = vx / vl;
    dy[i] = vy / vl;
  }
  // 잡음(가는 올 굵기 1~2px) → 방향을 따라 앞뒤로 L 걸음 평균
  let sd = seed * 2654435761 >>> 0;
  const noise = new Float32Array(n);
  for (let i = 0; i < n; i++) { sd = (sd * 1664525 + 1013904223) >>> 0; noise[i] = sd / 4294967296 - 0.5; }
  // width: 잡음 덩어리 반지름(가는 올 1px, 하이라이트 리본은 더 굵게), length: 흐름을 따라 문지르는 걸음 수
  const nz = boxBlur(noise, w, h, opt.width ?? 1);
  const L = Math.max(6, Math.round((opt.length ?? Math.min(w, h) / 60) * (1 - 0.75 * coily)));
  const out = new Float32Array(n);
  let sum = 0, sum2 = 0, cnt = 0;
  for (let i = 0; i < n; i++) {
    if (alpha[i] < 0.01) continue;
    let acc = nz[i], m = 1;
    for (const sgn of [1, -1]) {
      let x = i % w, y = (i / w) | 0;
      for (let k = 0; k < L; k++) {
        const j = (Math.round(y) * w + Math.round(x));
        if (j < 0 || j >= n) break;
        const ddx = dx[j] || dx[i], ddy = dy[j] || dy[i];
        x += sgn * ddx;
        y += sgn * ddy;
        const xi = Math.round(x), yi = Math.round(y);
        if (xi < 0 || yi < 0 || xi >= w || yi >= h) break;
        acc += nz[yi * w + xi];
        m++;
      }
    }
    out[i] = acc / m;
    sum += out[i]; sum2 += out[i] * out[i]; cnt++;
  }
  const mean = sum / Math.max(1, cnt);
  const sdv = Math.sqrt(Math.max(1e-9, sum2 / Math.max(1, cnt) - mean * mean));
  for (let i = 0; i < n; i++) if (alpha[i] >= 0.01) out[i] = Math.max(-2.5, Math.min(2.5, (out[i] - mean) / sdv)) / 2.5;
  return out;
}

/**
 * 머리카락 알파의 가장자리(분할이 확실하지 않은 곳)에서 '내 머리카락 색'과 다른 픽셀을 뺀다(제자리 수정).
 * 어두운 머리 + 어두운 배경(나뭇잎·벽)에서 알파가 배경으로 번져, 밝게 염색하면 배경까지 금발이 되던 것을 막는다.
 * sure: 확실한 머리카락(분할 확률)이 이 값 이상인 곳은 건드리지 않는다.
 */
export function colorGate(rgba: Uint8ClampedArray, alpha: Float32Array, coarse: Float32Array, sure = 0.85): void {
  const lab = new Float32Array(3);
  let n = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sl = 0;
  for (let i = 0; i < alpha.length; i += 2) {
    if (alpha[i] < 0.95 || coarse[i] < sure) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    n++; sa += lab[1]; sb += lab[2]; saa += lab[1] * lab[1]; sbb += lab[2] * lab[2]; sl += lab[0];
  }
  if (n < 100) return;
  const ma = sa / n, mb = sb / n, ml = Math.max(1, sl / n);
  const da = Math.max(3, Math.sqrt(Math.max(0, saa / n - ma * ma)));
  const db = Math.max(3, Math.sqrt(Math.max(0, sbb / n - mb * mb)));
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] <= 0 || coarse[i] >= sure) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    // 한쪽만 본다: 머리보다 초록(a↓)·파랑(b↓) 쪽으로 벗어난 것만 뺀다(나뭇잎·하늘·푸른 벽).
    // 피부와 섞인 이마 잔머리는 빨강·노랑(a·b↑) 쪽이고, 빛 받은 머리 끝은 밝기만 달라서 남는다
    // 어두운 곳(가닥 사이 그늘)은 색도 옅어진다: 기대 색을 밝기 비율만큼 줄여 비교한다
    // (안 하면 갈색 머리 그늘이 '덜 노란 픽셀'로 빠져, 검게 염색하면 붉은 갈색 줄로 남았다)
    const k = Math.max(0.25, Math.min(1, lab[0] / ml));
    const d = Math.hypot(Math.max(0, (ma * k - lab[1]) / da), Math.max(0, (mb * k - lab[2]) / db));
    const keep = Math.max(0, Math.min(1, 1 - (d - 2.5) / 2));
    alpha[i] *= keep;
  }
}

/**
 * 머리 둘레의 잔머리(1~2px 날린 가닥)를 알파에 더한다(제자리 수정).
 * 분할은 굵은 덩어리만 잡아서, 밖으로 뻗은 가는 가닥은 원래 색으로 남는다 → 밝게 염색하면 머리 둘레에 검은 테두리처럼 보인다.
 * 가닥은 가늘어서 둘레 배경 평균 밝기와 머리 밝기 사이 어디쯤이다: 배경 → 머리 쪽으로 간 비율을 알파로 쓴다(어두운 머리·밝은 배경, 반대도).
 * 덩어리 머리에서 reach 안쪽만, block(얼굴·목 확률 0~1)이 높은 곳은 건드리지 않는다(눈썹·눈가 그늘).
 */
export function flyaways(rgba: Uint8ClampedArray, w: number, h: number, alpha: Float32Array, reach: number, block: Float32Array | null = null): void {
  const n = w * h;
  const luma = new Float32Array(n);
  for (let i = 0; i < n; i++) luma[i] = 0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2];
  const hs: number[] = [];
  for (let i = 0; i < n; i += 3) if (alpha[i] >= 0.9) hs.push(luma[i]);
  if (hs.length < 100) return;
  hs.sort((a, b) => a - b);
  const hL = hs[hs.length >> 1];
  const sol = new Float32Array(n), bgw = new Float32Array(n), bgv = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (alpha[i] >= 0.5) sol[i] = 1;
    else if (alpha[i] < 0.05 && !(block && block[i] > 0.4)) { bgw[i] = 1; bgv[i] = luma[i]; }
  }
  const near = boxBlur(sol, w, h, Math.max(2, Math.round(reach)));
  const r = Math.max(3, Math.round(reach / 3));
  const num = boxBlur(bgv, w, h, r), den = boxBlur(bgw, w, h, r);
  // ② 반대로 머리 가장자리 안쪽인데 배경 밝기 그대로인 곳(가닥 사이로 보이는 벽)은 알파를 뺀다
  //    안 빼면 가장 밝은 분위로 대응돼 머리 둘레에 하얀 줄이 생긴다. 배경이 둘레에 충분히 있는 곳(가장자리)만
  const r2 = Math.max(4, Math.round(reach / 1.5));
  const num2 = boxBlur(bgv, w, h, r2), den2 = boxBlur(bgw, w, h, r2);
  for (let i = 0; i < n; i++) {
    if (alpha[i] < 0.05 || den2[i] < 0.05 || (block && block[i] > 0.4)) continue;
    const bg = num2[i] / den2[i];
    const c = bg - hL;
    if (Math.abs(c) < 25) continue;
    const s = (bg - luma[i]) / c;
    alpha[i] *= Math.max(0, Math.min(1, (s - 0.1) / 0.35));
  }
  // 색 검사: 잔머리는 배경과 머리의 섞임이라 색(a·b)도 그 사이에 있다. 머리 옆 가는 어두운 선이라도 색이 동떨어지면
  // (머리 끝에 닿은 보라색 옷깃) 잔머리가 아니다 — 옷깃이 머리색 줄로 칠해졌다
  const lab = new Float32Array(3);
  const ha: number[] = [], hb: number[] = [];
  for (let i = 0; i < n; i += 3) if (alpha[i] >= 0.9) { rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab); ha.push(lab[1]); hb.push(lab[2]); }
  ha.sort((a, b) => a - b); hb.sort((a, b) => a - b);
  const hA = ha[ha.length >> 1], hB = hb[hb.length >> 1];
  const bga = new Float32Array(n), bgb = new Float32Array(n);
  for (let i = 0; i < n; i++) if (bgw[i]) { rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab); bga[i] = lab[1]; bgb[i] = lab[2]; }
  const numA = boxBlur(bga, w, h, r), numB = boxBlur(bgb, w, h, r);
  for (let i = 0; i < n; i++) {
    if (alpha[i] >= 0.5 || near[i] <= 0 || den[i] < 0.15 || (block && block[i] > 0.4)) continue;
    const bg = num[i] / den[i];
    const c = bg - hL;
    // 배경과 머리 밝기가 비슷하면 가닥과 배경 무늬를 구별할 수 없다
    if (Math.abs(c) < 25) continue;
    const t = Math.max(0, Math.min(1, (bg - luma[i]) / c));
    let s = Math.max(0, Math.min(1, (t - 0.25) / 0.55));
    if (s <= 0) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    const ea = numA[i] / den[i] + t * (hA - numA[i] / den[i]), eb = numB[i] / den[i] + t * (hB - numB[i] / den[i]);
    s *= 1 - Math.max(0, Math.min(1, (Math.hypot(lab[1] - ea, lab[2] - eb) - 10) / 8));
    if (s <= 0) continue;
    const prox = Math.min(1, near[i] * 20);
    alpha[i] = Math.max(alpha[i], s * prox);
  }
}

/**
 * 머리카락 알파의 '속'만 남긴다(둘레 r 칸 안이 모두 머리카락인 곳). 고른 사진 색 분포용:
 * 시스루 앞머리 사이로 보이는 이마, 날린 가닥 사이 배경이 머리색 분포의 밝은 쪽에 섞이면
 * 검은 머리 사진이 '하이라이트 머리'로 읽혀 내 머리에 피부·배경색 줄이 생긴다. 속이 너무 적으면(짧은·가는 머리) 그대로.
 */
export function hairCore(alpha: Float32Array, w: number, h: number, thr: number, r: number, minN = 200): Float32Array {
  const m = new Float32Array(alpha.length);
  for (let i = 0; i < m.length; i++) m[i] = alpha[i] >= thr ? 1 : 0;
  const b = boxBlur(m, w, h, r);
  const out = new Float32Array(alpha.length);
  let n = 0;
  for (let i = 0; i < m.length; i++) if (b[i] > 0.97) { out[i] = alpha[i]; n++; }
  return n >= minN * 2 ? out : alpha;
}

/**
 * 머리카락 알파에서 색이 머리보다 다른 부류(피부·배경)에 가까운 픽셀을 뺀 새 알파.
 * 시스루 앞머리 사이 이마·가닥 사이 배경은 분할이 머리카락으로 보지만 색은 피부·배경이다.
 * 부류마다 확실한 곳(확률 0.8 이상)의 가운데 Lab 을 기준으로, 머리 가운데 색과 ΔE 20 넘게 다른 부류만 쓴다(금발·피부처럼 비슷하면 안 뺌).
 */
export function pureHair(rgba: Uint8ClampedArray, alpha: Float32Array, others: Float32Array[], thr = 0.7): Float32Array {
  const lab = new Float32Array(3);
  const med = (sel: (i: number) => boolean): [number, number, number] | null => {
    const L: number[] = [], A: number[] = [], B: number[] = [];
    for (let i = 0; i < alpha.length; i += 2) {
      if (!sel(i)) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      L.push(lab[0]); A.push(lab[1]); B.push(lab[2]);
    }
    if (L.length < 50) return null;
    const m = (v: number[]) => v.sort((p, q) => p - q)[v.length >> 1];
    return [m(L), m(A), m(B)];
  };
  const hairC = med((i) => alpha[i] >= 0.9);
  if (!hairC) return alpha;
  const d2 = (c: [number, number, number]) => (lab[0] - c[0]) ** 2 + (lab[1] - c[1]) ** 2 + (lab[2] - c[2]) ** 2;
  const cls: [number, number, number][] = [];
  for (const o of others) {
    const c = med((i) => o[i] >= 0.8);
    if (c && Math.sqrt((c[0] - hairC[0]) ** 2 + (c[1] - hairC[1]) ** 2 + (c[2] - hairC[2]) ** 2) > 20) cls.push(c);
  }
  if (!cls.length) return alpha;
  const out = new Float32Array(alpha);
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] < thr) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    const dh = d2(hairC);
    if (cls.some((c) => d2(c) < dh)) out[i] = 0;
  }
  return out;
}

/**
 * 내 머리 알파에서 색이 얼굴 피부 쪽인 픽셀(가르마 두피, 시스루 앞머리 사이 이마)을 덜 물들인다(제자리 수정).
 * 머리 가운데 색 → 피부 가운데 색 선 위에서 피부 쪽(55%~85%)일수록 알파를 줄인다. 머리와 피부색이 비슷하면(금발, ΔE 25 미만) 안 함.
 * faceP: 얼굴·목 확률(0~1). 피부 기준색은 그 확률 0.8 넘는 곳의 가운데 값.
 */
export function skinGate(rgba: Uint8ClampedArray, alpha: Float32Array, faceP: Float32Array): void {
  const lab = new Float32Array(3);
  const med = (sel: (i: number) => boolean): [number, number, number] | null => {
    const L: number[] = [], A: number[] = [], B: number[] = [];
    for (let i = 0; i < alpha.length; i += 3) {
      if (!sel(i)) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      L.push(lab[0]); A.push(lab[1]); B.push(lab[2]);
    }
    if (L.length < 50) return null;
    const m = (v: number[]) => v.sort((p, q) => p - q)[v.length >> 1];
    return [m(L), m(A), m(B)];
  };
  const hc = med((i) => alpha[i] >= 0.95);
  const sc = med((i) => faceP[i] >= 0.8 && alpha[i] < 0.05);
  if (!hc || !sc) return;
  const d = [sc[0] - hc[0], sc[1] - hc[1], sc[2] - hc[2]];
  const dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  if (dd < 25 * 25) return;
  const chromaSep = d[1] * d[1] + d[2] * d[2] > 12 * 12 && Math.hypot(hc[1], hc[2]) < 6;
  for (let i = 0; i < alpha.length; i++) {
    if (alpha[i] <= 0) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    const t = ((lab[0] - hc[0]) * d[0] + (lab[1] - hc[1]) * d[1] + (lab[2] - hc[2]) * d[2]) / dd;
    // 색도(a·b)까지 피부 쪽이어야: 빛 받아 밝기만 오른 머리(윤기)는 a·b 가 머리색이라 덜 걸린다
    const tc = (d[1] * d[1] + d[2] * d[2]) > 16 ? ((lab[1] - hc[1]) * d[1] + (lab[2] - hc[2]) * d[2]) / (d[1] * d[1] + d[2] * d[2]) : t;
    let s = Math.max(0, Math.min(1, (Math.min(t, tc) - 0.55) / 0.3));
    // 머리가 거의 무채색(흑발)이고 피부와 색도가 뚜렷이 다르면(a·b 거리 12 넘게), 색도가 피부 쪽(75~100%)이고 아주 어둡지 않은
    // (전체로 25~40% 넘게 피부 쪽) 픽셀은 그늘진 피부다: 밝기만 어두워 위 기준(둘 다 55%)에 안 걸려, 흑발로 염색할 때 그늘진
    // 이마·관자놀이가 색이 빠진 잿빛 보라 얼룩이 됐다(갈색 머리는 피부와 색도가 비슷해 여기 안 걸린다)
    if (chromaSep) s = Math.max(s, Math.max(0, Math.min(1, (tc - 0.75) / 0.25)) * Math.max(0, Math.min(1, (t - 0.25) / 0.15)));
    if (s > 0) alpha[i] *= 1 - s;
  }
}
