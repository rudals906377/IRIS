// 사진 모드 메이크업: 고른 사진 얼굴을 내 얼굴 모양에 맞춰 휘고(face-swap.ts 와 같은 삼각형 휘기),
// '맨 피부에서 얼마나 달라졌는가'(화장)만 옮긴다. 두 사진 각각 넓게 흐린 피부 바탕을 빼서 비교하므로
// 두 사진의 조명·피부색 차이는 옮기지 않고, 블러셔 모양·립 그라데이션·섀도 번짐·아이라인을 고른 사진 그대로 옮긴다.
//  - 색(a·b): 얼굴 전체(블러셔·윤곽·베이스), 입술, 눈가에서 내 화장 차이를 고른 사진 화장 차이로 바꾼다
//  - 밝기(L): 입술·눈가는 그대로, 피부는 조금만(조명까지 옮기지 않게). 아이라인·속눈썹(눈가의 어두운 잔무늬)과 입술 광은 따로 더한다
//  - 눈동자·흰자·눈썹·앞머리·얼굴 가린 손은 건드리지 않는다
// 순수 계산(DOM 없음).

import { fillPoly, warpTriangles, type P } from './face-swap.ts';
import { boxBlur, lab2rgb, rgb2lab } from './still-hair.ts';
import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_INNER, LIPS_OUTER } from './face-regions.ts';

export interface MakeupAmounts {
  /** 입술 */
  lip: number;
  /** 눈가(섀도·아이라인) */
  eye: number;
  /** 피부(블러셔·윤곽·베이스) */
  skin: number;
}

/** 두 사진 피부 바탕색 차이 중 옮기는 몫(나머지는 내 피부색 유지) */
const BASE_SHIFT = 0.4;
/** 화장 차이 범위(L, a, b) */
const LIMIT: [number, number][] = [[-40, 18], [-5, 30], [-10, 14]];

const ramp = (v: number, a: number, b: number) => Math.max(0, Math.min(1, (v - a) / (b - a)));

/** 가중 흐림: blur(v·w)/blur(w) */
function wblur(v: Float32Array, w: Float32Array, W: number, H: number, r: number): Float32Array {
  const vw = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) vw[i] = v[i] * w[i];
  const a = boxBlur(vw, W, H, r), b = boxBlur(w, W, H, r);
  for (let i = 0; i < a.length; i++) a[i] = b[i] > 1e-4 ? a[i] / b[i] : 0;
  return a;
}

/**
 * my(내 사진 RGBA, 점 myPts)에 ref(고른 사진, 점 refPts)의 화장을 옮긴 새 그림.
 * oval: 얼굴 윤곽 점 번호. mySkin/refSkin: 각 사진 크기의 얼굴 피부·이목구비 확률(앞머리 뺀 것, 없으면 윤곽만).
 * refHands: 고른 사진 크기의 몸 피부(얼굴 가린 손) 확률.
 */
export function transferMakeup(
  my: Uint8ClampedArray, W: number, H: number, myPts: P[],
  ref: Uint8ClampedArray, rw: number, rh: number, refPts: P[],
  oval: number[], amt: MakeupAmounts,
  mySkin: Float32Array | null = null, refSkin: Float32Array | null = null, refHands: Float32Array | null = null,
  dbg?: Record<string, unknown>,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(my);
  const fw = Math.hypot(myPts[454].x - myPts[234].x, myPts[454].y - myPts[234].y);
  // 얼굴 둘레 상자 안만 계산(빠르게)
  let bx0 = W, by0 = H, bx1 = 0, by1 = 0;
  for (const i of oval) { const p = myPts[i]; bx0 = Math.min(bx0, p.x); by0 = Math.min(by0, p.y); bx1 = Math.max(bx1, p.x); by1 = Math.max(by1, p.y); }
  const m = fw * 0.15;
  const x0 = Math.max(0, Math.floor(bx0 - m)), y0 = Math.max(0, Math.floor(by0 - m));
  const x1 = Math.min(W, Math.ceil(bx1 + m)), y1 = Math.min(H, Math.ceil(by1 + m));
  const w = x1 - x0, h = y1 - y0;
  if (w < 16 || h < 16) return out;
  const loc = (pts: P[]) => pts.map((p) => ({ x: p.x - x0, y: p.y - y0 }));
  const dp = loc(myPts.slice(0, 468));
  const sub = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) sub.set(my.subarray(((y + y0) * W + x0) * 4, ((y + y0) * W + x1) * 4), y * w * 4);
  // ① 고른 사진 얼굴을 내 얼굴 모양으로
  const extras: Float32Array[] = [];
  if (refSkin) extras.push(refSkin);
  if (refHands) extras.push(refHands);
  const wr = warpTriangles(dp, w, h, ref, rw, rh, refPts.slice(0, 468), extras);
  const rSkin = refSkin ? wr.extra[0] : null;
  const rHands = refHands ? wr.extra[refSkin ? 1 : 0] : null;
  // ② 자리(내 얼굴 모양 기준)
  const poly = (ids: number[]) => fillPoly(ids.map((i) => dp[i]), w, h);
  const face = poly(oval);
  const lipO = poly(LIPS_OUTER), lipI = poly(LIPS_INNER);
  const eyeO = poly([...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)]);
  const eyeO2 = poly([...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)]);
  const brow = poly([...BROW_R_LOWER, ...BROW_R_UPPER]), brow2 = poly([...BROW_L_LOWER, ...BROW_L_UPPER]);
  const n = w * h;
  const eyeOpen = new Float32Array(n), browM0 = new Float32Array(n);
  for (let i = 0; i < n; i++) { eyeOpen[i] = Math.max(eyeO[i], eyeO2[i]); browM0[i] = Math.max(brow[i], brow2[i]); }
  const rf = (k: number) => Math.max(1, Math.round(fw * k));
  // 눈가: 눈 둘레를 넓힌 곳(섀도가 눈썹 아래까지) − 눈 안(살짝 줄여서: 속눈썹 선은 눈 경계에 걸친다)
  const zoneB = boxBlur(eyeOpen, w, h, rf(0.07));
  const openB = boxBlur(eyeOpen, w, h, rf(0.008));
  const browB = boxBlur(browM0, w, h, rf(0.015));
  const lipM = new Float32Array(n), eyeM = new Float32Array(n), browM = new Float32Array(n), faceM = new Float32Array(n);
  const lipB = boxBlur(lipO, w, h, rf(0.008)), innB = boxBlur(lipI, w, h, rf(0.006));
  const faceB = boxBlur(face, w, h, rf(0.03));
  for (let i = 0; i < n; i++) {
    lipM[i] = lipB[i] * (1 - ramp(innB[i], 0.3, 0.8));
    eyeM[i] = ramp(zoneB[i], 0.02, 0.15) * (1 - ramp(openB[i], 0.55, 0.9));
    browM[i] = ramp(browB[i], 0.05, 0.4);
    faceM[i] = ramp(faceB[i], 0.5, 0.95);
  }
  // ③ 믿을 수 있는 곳: 휜 곳 ∩ 두 사진 모두 얼굴 피부(앞머리 아님) ∩ 고른 사진 손 아님
  const V = new Float32Array(n);
  // 손가락 가장자리·손톱은 몸 피부로 덜 잡혀 손가락 색이 새므로 넓혀서 뺀다
  const handsW = rHands ? boxBlur(rHands, w, h, rf(0.02)) : null;
  for (let i = 0; i < n; i++) {
    const X = (i % w) + x0, Y = Math.floor(i / w) + y0;
    V[i] = wr.cover[i] * (rSkin ? ramp(rSkin[i], 0.3, 0.7) : 1) * (mySkin ? ramp(mySkin[Y * W + X], 0.3, 0.7) : 1) * (handsW ? 1 - ramp(handsW[i], 0.1, 0.3) : 1);
  }
  // ④ Lab 과 바탕(넓게 흐린 맨 피부) · 저주파(화장 번짐)
  const myL = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const rfL = [new Float32Array(n), new Float32Array(n), new Float32Array(n)];
  const t = new Float32Array(3);
  for (let i = 0; i < n; i++) {
    rgb2lab(sub[i * 4], sub[i * 4 + 1], sub[i * 4 + 2], t);
    myL[0][i] = t[0]; myL[1][i] = t[1]; myL[2][i] = t[2];
    const r = Math.round(wr.rgb[i * 3]), g = Math.round(wr.rgb[i * 3 + 1]), b = Math.round(wr.rgb[i * 3 + 2]);
    rgb2lab(Math.max(0, Math.min(255, r)), Math.max(0, Math.min(255, g)), Math.max(0, Math.min(255, b)), t);
    rfL[0][i] = t[0]; rfL[1][i] = t[1]; rfL[2][i] = t[2];
  }
  const wBase = new Float32Array(n);
  for (let i = 0; i < n; i++) wBase[i] = V[i] * faceM[i] * (1 - ramp(lipB[i], 0, 0.2)) * (1 - ramp(zoneB[i], 0, 0.1)) * (1 - browM[i]);
  const rB = rf(0.22), rL = rf(0.015);
  // 밝기 바탕은 넓게 흐린 피부(조명 기울기를 따라가게). 색 바탕은 맨 피부 값 하나: 화장은 대개 붉은 기를 더하므로
  // a 는 피부 중 덜 붉은 쪽(5% 자리), b 는 가운데 값. 넓게 흐린 색 바탕은 볼 전체 블러셔를 바탕으로 삼켜 버리고,
  // 이마·콧등 같은 고정 자리는 코까지 칠한 블러셔에 틀린다
  const myBase = [wblur(myL[0], wBase, w, h, rB)];
  const rfBase = [wblur(rfL[0], wBase, w, h, rB)];
  const pick: number[] = [];
  for (let i = 0; i < n; i += 2) if (wBase[i] > 0.5) pick.push(i);
  const pct = (ch: Float32Array, q: number) => {
    const v = pick.map((i) => ch[i]).sort((x, y) => x - y);
    return new Float32Array(n).fill(v.length ? v[Math.floor((v.length - 1) * q)] : 0);
  };
  myBase[1] = pct(myL[1], 0.05); rfBase[1] = pct(rfL[1], 0.05);
  myBase[2] = pct(myL[2], 0.5); rfBase[2] = pct(rfL[2], 0.5);
  const myLow = myL.map((c) => wblur(c, V, w, h, rL));
  const rfLow = rfL.map((c) => wblur(c, V, w, h, rL));
  // 화장 차이. 손·앞머리로 가려 고른 사진 화장을 모르는 곳(입술 위 손가락 등)은 둘레 값으로 메운다
  const D = [0, 1, 2].map((c) => {
    const d = new Float32Array(n);
    // 색은 바탕 차이(베이스 화장·톤)도 일부(BASE_SHIFT) 옮긴다. 다만 바탕이 없는 곳(입술·눈 안)은 그 자리 차이만
    const g = c === 0 ? 0 : BASE_SHIFT * amt.skin;
    // 내 원래 화색은 피부에선 절반만 빼고(내 혈색 + 고른 사진 블러셔), 입술·눈가는 모두 빼서 그 색으로 바꾼다.
    // 화장은 피부를 파랗게·크게 어둡게 만들지 않으므로 차이를 그럴 만한 범위로 자른다(가장자리·머리카락 대응 실수 막기)
    const [lo, hi] = LIMIT[c];
    for (let i = 0; i < n; i++) {
      const wm = 0.5 + 0.5 * Math.max(lipM[i], eyeM[i]);
      const v = (rfLow[c][i] - rfBase[c][i]) - wm * (myLow[c][i] - myBase[c][i]) + g * (rfBase[c][i] - myBase[c][i]);
      d[i] = Math.max(lo, Math.min(hi, v));
    }
    return d;
  });
  // 가까운 둘레(얼굴 폭 6%)로 먼저, 손이 넓게 가린 곳은 먼 둘레(18%)로
  const Df = D.map((d) => wblur(d, V, w, h, rf(0.06)));
  const Dg = D.map((d) => wblur(d, V, w, h, rf(0.18)));
  const C1 = boxBlur(V, w, h, rf(0.06));
  const Cf = boxBlur(V, w, h, rf(0.18));
  const Vb = boxBlur(V, w, h, rf(0.012));
  for (let i = 0; i < n; i++) {
    const v = ramp(Vb[i], 0.3, 0.9), c1 = ramp(C1[i], 0.05, 0.3);
    for (let c = 0; c < 3; c++) D[c][i] = D[c][i] * v + (Df[c][i] * c1 + Dg[c][i] * (1 - c1)) * (1 - v);
    Vb[i] = v;
  }
  if (dbg) {
    const at = (k: number) => { const p = dp[k]; const i = Math.round(p.y) * w + Math.round(p.x); return { V: +V[i].toFixed(2), Cf: +Cf[i].toFixed(2), my: [0, 1, 2].map((c) => +myLow[c][i].toFixed(1)), rf: [0, 1, 2].map((c) => +rfLow[c][i].toFixed(1)), D: [0, 1, 2].map((c) => +D[c][i].toFixed(1)) }; };
    Object.assign(dbg, { amt, base: { my: [myBase[1][0], myBase[2][0]], rf: [rfBase[1][0], rfBase[2][0]] }, cheekR: at(205), cheekL: at(425), lip: at(14), nose: at(6), npick: pick.length });
  }
  // ⑤ 옮기기
  const o = new Uint8ClampedArray(4);
  for (let i = 0; i < n; i++) {
    // 눈 안(흰자·눈동자)은 어떤 화장도 옮기지 않는다
    const keep = (1 - browM[i]) * ramp(Cf[i], 0.03, 0.15) * faceM[i] * (1 - ramp(openB[i], 0.5, 0.85));
    if (keep <= 0.002) continue;
    const mAB = keep * Math.max(amt.skin * faceM[i], amt.lip * lipM[i], amt.eye * eyeM[i]);
    const mL = keep * Math.min(1, amt.lip * lipM[i] + amt.eye * eyeM[i] + amt.skin * 0.2 * faceM[i] * (1 - lipM[i]) * (1 - eyeM[i]));
    if (mAB <= 0.002 && mL <= 0.002) continue;
    let L = myL[0][i] + D[0][i] * mL;
    const a = myL[1][i] + D[1][i] * mAB, b = myL[2][i] + D[2][i] * mAB;
    // 아이라인·속눈썹(눈가의 어두운 잔무늬), 입술 광(입술의 밝은 잔무늬)
    const det = (rfL[0][i] - rfLow[0][i]) * keep * Vb[i];
    L += amt.eye * eyeM[i] * Math.min(0, det) * 0.9 + amt.lip * lipM[i] * Math.max(0, det) * 0.6;
    lab2rgb(Math.max(0, Math.min(100, L)), a, b, o, 0);
    const X = (i % w) + x0, Y = Math.floor(i / w) + y0;
    const q = (Y * W + X) * 4;
    out[q] = o[0]; out[q + 1] = o[1]; out[q + 2] = o[2];
  }
  return out;
}
