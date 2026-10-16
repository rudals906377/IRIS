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

function rgb2lab(r: number, g: number, b: number, out: Float32Array): void {
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

function lab2rgb(L: number, a: number, b: number, out: Uint8ClampedArray, o: number): void {
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
    let sa = 0, sb = 0;
    for (let j = lo; j <= hi; j++) {
      sa += As[idx[j]];
      sb += Bs[idx[j]];
    }
    a[k] = sa / (hi - lo + 1);
    b[k] = sb / (hi - lo + 1);
  }
  return { L, a, b, n };
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
export function recolorHair(rgba: Uint8ClampedArray, w: number, h: number, alpha: Float32Array, mine: HairDist, ref: HairDist, amount = 1): Uint8ClampedArray {
  const out = new Uint8ClampedArray(rgba);
  const lab = new Float32Array(3);
  const tmp = new Uint8ClampedArray(3);
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
  for (let i = 0; i < w * h; i++) {
    const al = alpha[i] * amt;
    if (al < 0.004) continue;
    const o = i * 4;
    rgb2lab(rgba[o], rgba[o + 1], rgba[o + 2], lab);
    // 양 끝 분위(그림자·배경·반사가 섞인 곳)는 버리고 3%~97% 사이로 대응시킨다(검은 점·흰 점 방지)
    const t = TRIM + Math.pow(quantilePos(mine.L, lab[0]) / Q, g) * (Q - 2 * TRIM);
    let L = at(ref.L, t);
    // 잔결: 내 픽셀이 주변보다 밝거나 어두운 정도(L 단위 근사)
    L += ((luma[i] - box[i]) / 255) * 100 * detail;
    lab2rgb(Math.max(0, Math.min(100, L)), at(ref.a, t), at(ref.b, t), tmp, 0);
    out[o] = rgba[o] + (tmp[0] - rgba[o]) * al;
    out[o + 1] = rgba[o + 1] + (tmp[1] - rgba[o + 1]) * al;
    out[o + 2] = rgba[o + 2] + (tmp[2] - rgba[o + 2]) * al;
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
