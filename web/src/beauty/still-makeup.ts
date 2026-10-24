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
/** 어두운 피부의 입술에서 고른 사진 입술 밝기를 그대로 쓰는 몫(색 a·b 는 늘 립스틱 색 자체를 씀) */
const LIP_ABS_L = 0.35;

const ramp = (v: number, a: number, b: number) => Math.max(0, Math.min(1, (v - a) / (b - a)));

/** 사진 가장자리(짧은 변 4% 띠)의 색도(Lab a·b 크기) 가운데 값: 색 배경·색 조명 사진인지 */
export function borderChroma(rgba: Uint8ClampedArray, w: number, h: number): number {
  const bw = Math.max(2, Math.round(Math.min(w, h) * 0.04));
  const lab = new Float32Array(3);
  const v: number[] = [];
  for (let y = 0; y < h; y += 2) for (let x = 0; x < w; x += 2) {
    if (x >= bw && x < w - bw && y >= bw && y < h - bw) continue;
    const o = (y * w + x) * 4;
    rgb2lab(rgba[o], rgba[o + 1], rgba[o + 2], lab);
    v.push(Math.hypot(lab[1], lab[2]));
  }
  if (!v.length) return 0;
  v.sort((p, q) => p - q);
  return v[v.length >> 1];
}

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
  /** 피부에서 내 원래 화색을 빼는 몫(0.5: 내 혈색 + 고른 사진 블러셔). 얼굴 바꾸기는 1(바뀐 얼굴의 내 홍조가 겹쳐 블러셔가 두 배가 됐다) */
  skinOwn = 0.5,
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
  // 얼굴 가장자리는 넓게(얼굴 폭 6%) 녹인다: 좁으면 턱선·이마 둘레에 화장한 얼굴과 귀·목의 경계가 선처럼 보인다
  const faceB = boxBlur(boxBlur(face, w, h, rf(0.03)), w, h, rf(0.03));
  for (let i = 0; i < n; i++) {
    lipM[i] = lipB[i] * (1 - ramp(innB[i], 0.3, 0.8));
    eyeM[i] = ramp(zoneB[i], 0.02, 0.15) * (1 - ramp(openB[i], 0.55, 0.9));
    browM[i] = ramp(browB[i], 0.05, 0.4);
    faceM[i] = ramp(faceB[i], 0.35, 0.95);
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
  // 색 조명: 고른 사진 바탕(사진 가장자리)이 진한 색이면(빨간 배경 16번: 색도 60) 그 빛이 얼굴 전체에 비쳐 피부가 고르게 붉게 찍힌다.
  // 덜 붉은 5% 피부를 맨 피부로 삼으면 그 빛까지 화장이 되어 내 얼굴 아래쪽 전체·턱까지 분홍이 됐다(볼~턱 a +7.7).
  // → 바탕 색도가 진할수록, 블러셔가 잘 가지 않는 이마(눈썹 위)·턱(입술 아래) 피부의 가운데 색을 맨 피부 색으로 쓴다
  const cast = ramp(borderChroma(ref, rw, rh), 30, 50);
  const browTop = Math.min(...[...BROW_R_UPPER, ...BROW_L_UPPER].map((k) => dp[k].y));
  const browIds = [...BROW_R_UPPER, ...BROW_L_UPPER, ...BROW_R_LOWER, ...BROW_L_LOWER];
  const browMid = browIds.reduce((t, k) => t + dp[k].y, 0) / browIds.length;
  if (cast > 0) {
    const lipBot = Math.max(...LIPS_OUTER.map((k) => dp[k].y));
    const va: number[] = [];
    for (const i of pick) {
      const y = (i / w) | 0;
      if (y < browTop - fw * 0.03 || y > lipBot + fw * 0.06) va.push(rfL[1][i]);
    }
    if (va.length > 50) {
      va.sort((p, q) => p - q);
      const a0 = rfBase[1][0];
      // 붉은 기(a)만, 그것도 70%: b 까지 옮기면 얼굴이 잿빛으로 떴고, 다 빼면 고른 사진의 진한 과즙 블러셔까지 거의 사라졌다
      rfBase[1].fill(a0 + Math.max(0, va[va.length >> 1] - a0) * cast * 0.7);
    }
  }
  if (dbg) dbg.cast = +cast.toFixed(2);
  // 고른 사진에서 피부 바탕보다 L 25~40 넘게 어두운 곳(눈가·입술·눈썹 밖)은 화장이 아니다: 볼에 댄 손의 검은 손톱,
  // 이마에 흘러내린 머리카락, 손가락 그늘. 손 지도에서 반쯤만 빠진 검은 손톱이 둘레 메우기에 섞여 볼에 회색 얼룩이 됐다
  // (눈 둘레 전체·입술·눈썹·콧구멍은 원래 어둡다: 속눈썹·아이라인·입 안 선은 그대로 믿는다)
  {
    const nb = [64, 98, 240, 2, 460, 327, 294].map((k) => dp[k]);
    const mg = fw * 0.03;
    const nx0 = Math.min(...nb.map((p) => p.x)) - mg, nx1 = Math.max(...nb.map((p) => p.x)) + mg;
    const ny0 = Math.min(...nb.map((p) => p.y)) - mg, ny1 = Math.max(...nb.map((p) => p.y)) + mg;
    for (let i = 0; i < n; i++) {
      if (V[i] <= 0) continue;
      const x = i % w, y = (i / w) | 0;
      if (x >= nx0 && x <= nx1 && y >= ny0 && y <= ny1) continue;
      const zone = Math.max(lipB[i], ramp(zoneB[i], 0.02, 0.15), browM[i]);
      V[i] *= 1 - ramp(rfBase[0][i] - rfL[0][i], 25, 40) * (1 - zone);
    }
  }
  const myLow = myL.map((c) => wblur(c, V, w, h, rL));
  const rfLow = rfL.map((c) => wblur(c, V, w, h, rL));
  // 화장 차이. 손·앞머리로 가려 고른 사진 화장을 모르는 곳(입술 위 손가락 등)은 둘레 값으로 메운다
  const mouthNear = boxBlur(lipO, w, h, rf(0.06));
  // 콧방울·콧구멍 둘레도 같다: 고른 사진의 콧구멍 테·코 밑 그늘(L −6~−12, a +10)이 바뀐 얼굴에 붉은 멍처럼 옮겨졌다
  const noseNear = boxBlur(poly([64, 98, 240, 2, 460, 327, 294]), w, h, rf(0.06));
  for (let i = 0; i < n; i++) mouthNear[i] = Math.max(ramp(mouthNear[i], 0, 0.05), ramp(noseNear[i], 0, 0.05));
  const D = [0, 1, 2].map((c) => {
    const d = new Float32Array(n);
    // 색은 바탕 차이(베이스 화장·톤)도 일부(BASE_SHIFT) 옮긴다. 다만 바탕이 없는 곳(입술·눈 안)은 그 자리 차이만
    const g = c === 0 ? 0 : BASE_SHIFT * amt.skin;
    // 내 원래 화색은 피부에선 절반만 빼고(내 혈색 + 고른 사진 블러셔), 입술·눈가는 모두 빼서 그 색으로 바꾼다.
    // 화장은 피부를 파랗게·크게 어둡게 만들지 않으므로 차이를 그럴 만한 범위로 자른다(가장자리·머리카락 대응 실수 막기)
    const [lo, hi] = LIMIT[c];
    for (let i = 0; i < n; i++) {
      const zm = Math.max(lipM[i], eyeM[i]);
      // 그늘(피부 바탕보다 L 2~8 넘게 어두움: 콧대 옆·턱 밑)의 따뜻한 색은 화장이 아니라 내 얼굴 굴곡의 그늘이다:
      // 피부에서는 그 몫을 빼지 않는다(다 빼면 얼굴 바꾸기에서 콧대 옆이 누런 회색 띠가 됐다)
      const shade = ramp(myBase[0][i] - myLow[0][i], 2, 8) * (1 - zm);
      const wm = (skinOwn + (1 - skinOwn) * zm) * (1 - shade);
      // 바탕색 차이 몫은 ±3 까지만: 얼굴에만 입히므로 크면 귀·목과 갈려 가면처럼 보인다(빨간 조명 사진은 피부가 붉게 찍힌다)
      // 고른 사진 쪽 그늘도 같다: 입술 밖 입 둘레(얼굴 폭 6% 안)에서 고른 피부 바탕보다 L 4~12 넘게 어두운 곳
      // (아랫입술 밑·입꼬리 그늘)의 붉은 기는 화장이 아니라 그늘색이다(밝은 분홍빛 피부는 그늘이 붉다): 색 차이를 70%까지 덜 옮긴다.
      // (아랫입술 밑 그늘(L −18, a +25)이 내 턱에 그대로 옮겨져 립이 번진 듯한 분홍 얼룩이 됐다. 볼 블러셔는 입 둘레 밖이라 그대로)
      // 더 깊은 그늘(L 10~20 넘게 어두움: 입술에 댄 손가락과 아랫입술 사이 주름, 18번)은 밝기·색 모두 화장이 아니다:
      // 밝기 차이도 80%, 색은 모두 뺀다(L −24·a +5 가 내 턱에 주황 점으로 남았음)
      const shDark = rfBase[0][i] - rfLow[0][i], shNear = mouthNear[i] * (1 - ramp(lipB[i], 0.05, 0.5));
      const rShade = (c === 0 ? 0.8 * ramp(shDark, 10, 20) : Math.max(0.7 * ramp(shDark, 4, 12), ramp(shDark, 10, 20))) * shNear;
      let v = (rfLow[c][i] - rfBase[c][i]) * (1 - rShade) - wm * (myLow[c][i] - myBase[c][i]) + Math.max(-3, Math.min(3, g * (rfBase[c][i] - myBase[c][i])));
      // 이마(눈썹 위)는 블러셔·립·섀도가 가지 않는 자리: 그곳의 색 차이는 화장보다 피부 상태·조명이다. 눈썹 왁싱 직후 사진의
      // 붉어진 이마(a +10)가 내 이마 전체에 분홍 띠로 옮겨졌다 → 색(a·b)은 25% 만(밝기·하이라이트는 그대로).
      // 눈썹 높이 가운데부터 위로(미간·눈썹 바로 위도: 가장 높은 눈썹 끝부터 재면 그 아래 미간에 분홍이 남았다), 눈썹·눈가는 빼고
      if (c > 0) v *= 1 - 0.75 * ramp(browMid - ((i / w) | 0), 0, fw * 0.05) * (1 - zm) * (1 - browM[i]);
      // 눈가에서 내 그늘(눈 안쪽 구석·눈 밑, 바탕보다 L 15~25 어두움)을 다 빼면 고른 사진에 그 그늘이 없는 자리가 +10~13 밝아진다:
      // 깊은 눈·어두운 피부(will1, 바탕 L 54)에서는 그늘이 사라진 주황 얼룩이 됐다. 컨실러도 그늘을 다 못 지우므로
      // 눈가의 순 밝힘은 고른 사진 자체의 밝힘(하이라이트) + 4 까지만
      if (c === 0 && eyeM[i] > 0) {
        const cap = 4 + Math.max(0, rfLow[0][i] - rfBase[0][i]);
        if (v > cap) v = cap + (v - cap) * (1 - eyeM[i]);
      }
      // 피부(입술·눈가 밖)는 노란 기를 3 넘게 빼지 않는다: 화장은 피부를 파랗게 만들지 않는데, 고른 사진 조명
      // (밝게 비친 콧등·이마는 하얗게, 즉 b 가 낮게 찍힌다)이 차이로 옮겨져 얼굴이 잿빛으로 떴다. 섀도·립은 차가운 색도 그대로
      // 입술은 내 립스틱을 지우고 고른 사진 색으로 바꿔야 하므로 붉은 기(a)를 피부 한도(−5)보다 훨씬 더 뺄 수 있어야 한다:
      // 진한 빨간 립(a 50)에 고른 사진의 연한 장밋빛 립(a 33)을 입혀도 −5 에 걸려 a 45 의 빨간 립 그대로였다 → 입술 안은 −30 까지
      const lo2 = c === 2 ? lo * zm - 3 * (1 - zm) : c === 1 ? lo - 25 * lipM[i] : lo;
      // 입술: 립스틱은 입술 위에 얹는 색이라, 내 피부가 고른 사진보다 훨씬 어두우면 '고른 사진 피부와의 차이'를 옮길 때
      // 내 피부의 누런 기와 큰 명암 차이가 그대로 더해진다(어두운 피부에 장밋빛 립 → 어둡고 누런 벽돌색).
      // 그때만(피부 바탕 L 차이 10~22) 고른 사진 입술 색 그대로(내 잔무늬는 남김)를 밝기 35%·색 60% 섞는다.
      // 피부 밝기가 비슷하면 차이 방식 그대로: 그대로 쓰면 고른 사진 조명(차가운 화이트밸런스)까지 옮겨져 입술이 회보라로 탁해진다
      const kAbs = lipM[i] > 0 ? lipM[i] * ramp(rfBase[0][i] - myBase[0][i], 10, 22) : 0;
      // 입술 색(a·b)은 '피부와의 차이'가 아니라 립스틱 색 자체: 차이로 옮기면 고른 사진 입술이 그 피부보다 얼마나 붉은지·노란지가
      // 내 피부 위에 더해져, 차가운 흰 피부의 장밋빛 립(b 9, 피부 b 8)이 따뜻한 피부(b 27)에서 코랄(b 24)이 됐다.
      // 고른 사진 입술 색 그대로에 두 사진 피부 바탕색 차이는 ±5 까지만 더한다(화이트밸런스 차이는 그쯤, 피부색 차이는 그보다 커도 립에 안 묻는다)
      if (c > 0 && lipM[i] > 0) {
        const wb = Math.max(-5, Math.min(5, myBase[c][i] - rfBase[c][i]));
        v += (rfLow[c][i] - myLow[c][i] + wb - v) * lipM[i];
      }
      let va = kAbs > 0 && c === 0 ? v + (rfLow[c][i] - myLow[c][i] - v) * LIP_ABS_L * kAbs : v;
      // 그때 밝기는 고른 사진 입술 자체보다 어둡게 하지 않는다(내 입술보다 밝은 립이면 어둡게 하지 않음): 고른 사진의
      // '피부보다 31 어두운 립'을 차이로 옮기면 어두운 피부의 입술이 L 52 → 37 의 검붉은 립이 됐다(립스틱은 얹는 색이라
      // 고른 사진에서 보이는 밝기(L 45)보다 어두워질 까닭이 없다)
      if (c === 0 && kAbs > 0) {
        const floor = Math.min(0, rfLow[0][i] - myLow[0][i]);
        if (va < floor) va += (floor - va) * kAbs;
      }
      d[i] = Math.max(lo2, Math.min(hi, va));
    }
    return d;
  });
  // 입술 밝기 차이는 입술 가운데 값 ±8 안으로: 립스틱은 입술을 고르게 바꾼다. 고른 사진이 입을 살짝 벌려
  // 입꼬리·입술 사이가 그늘로 새까마면 그 그늘(화장 아님)이 다문 내 입꼬리에 검은 점·줄로 옮겨졌다
  // 색(a·b)도 가운데 값 ±3 안으로: 립스틱 색은 윗입술·아랫입술이 같다. 고른 사진의 아랫입술만 빛을 받아 더 붉으면(01: a 43 vs 34)
  // 내 윗입술은 분홍(a +14)·아랫입술은 진한 빨강(a +23)이 되어 아랫입술에만 립을 바른 듯 보였다
  for (const [c, lim] of [[0, 8], [1, 3], [2, 3]] as const) {
    const v: number[] = [];
    for (let i = 0; i < n; i += 2) if (lipM[i] > 0.5 && V[i] > 0.5) v.push(D[c][i]);
    if (v.length > 20) {
      v.sort((p, q) => p - q);
      const med = v[v.length >> 1];
      for (let i = 0; i < n; i++) {
        if (lipM[i] <= 0) continue;
        const k = Math.max(med - lim, Math.min(med + lim, D[c][i]));
        D[c][i] += (k - D[c][i]) * lipM[i];
      }
    }
  }
  // 가까운 둘레(얼굴 폭 6%)로 먼저, 손이 넓게 가린 곳은 먼 둘레(18%)로.
  // 입술 밖은 입술 값을 빼고 메운다: 섞으면 턱에 댄 손가락으로 가린 아랫입술 밑이 립 색(a +15)으로 메워져
  // 턱에 분홍 얼룩이 번졌다. 입술 자리는 지금처럼 둘레 전체로(눈가는 빼지 않는다: 눈 밑 블러셔가 볼 메우기에 필요)
  // 눈 안(뜬 눈: 흰자·눈동자)은 어디도 메우지 않는다: 두 사진 눈 모양이 달라 흰자 ↔ 눈동자가 맞물리면 D 가 L −40 이라,
  // 내 쪽 피부 분할이 빠진 눈두덩이 그 값으로 메워져 윗눈꺼풀 전체가 짙은 갈색 띠가 됐다
  const zoneW = new Float32Array(n), Vs = new Float32Array(n), Vd = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    zoneW[i] = ramp(lipB[i], 0, 0.2);
    Vd[i] = V[i] * (1 - ramp(openB[i], 0.2, 0.5));
    Vs[i] = Vd[i] * (1 - zoneW[i]);
  }
  const Df = D.map((d) => wblur(d, Vd, w, h, rf(0.06)));
  const Dg = D.map((d) => wblur(d, Vd, w, h, rf(0.18)));
  const Dfs = D.map((d) => wblur(d, Vs, w, h, rf(0.06)));
  const Dgs = D.map((d) => wblur(d, Vs, w, h, rf(0.18)));
  const C1 = boxBlur(Vd, w, h, rf(0.06)), C1s = boxBlur(Vs, w, h, rf(0.06));
  const Cf = boxBlur(V, w, h, rf(0.18));
  const Vb = boxBlur(V, w, h, rf(0.012));
  for (let i = 0; i < n; i++) {
    const v = ramp(Vb[i], 0.3, 0.9), c1 = ramp(C1[i], 0.05, 0.3), c1s = ramp(C1s[i], 0.05, 0.3), z = zoneW[i];
    for (let c = 0; c < 3; c++) {
      const all = Df[c][i] * c1 + Dg[c][i] * (1 - c1), skin = Dfs[c][i] * c1s + Dgs[c][i] * (1 - c1s);
      D[c][i] = D[c][i] * v + (all * z + skin * (1 - z)) * (1 - v);
    }
    Vb[i] = v;
  }
  if (dbg) {
    const at = (k: number) => { const p = dp[k]; const i = Math.round(p.y) * w + Math.round(p.x); return { V: +V[i].toFixed(2), Cf: +Cf[i].toFixed(2), my: [0, 1, 2].map((c) => +myLow[c][i].toFixed(1)), rf: [0, 1, 2].map((c) => +rfLow[c][i].toFixed(1)), D: [0, 1, 2].map((c) => +D[c][i].toFixed(1)) }; };
    Object.assign(dbg, { amt, base: { my: [myBase[1][0], myBase[2][0]], rf: [rfBase[1][0], rfBase[2][0]] }, cheekR: at(205), cheekL: at(425), lip: at(14), nose: at(6), npick: pick.length });
  }
  // 아이라인·속눈썹: 고른 사진에서 눈두덩 평균(얼굴 폭 5%)보다 어두운 정도를 내 눈두덩 평균 위에 놓는다(내 속눈썹이 더 어두우면 그대로)
  // 주근깨: 피부 평균(얼굴 폭 2%)보다 뚜렷이(L 2.5 넘게) 어두운 작은 점만(옮기면서 흐려지므로 2배)
  const rfLid = wblur(rfL[0], V, w, h, rf(0.05)), myLid = wblur(myL[0], V, w, h, rf(0.05));
  const rfSkinMean = wblur(rfL[0], V, w, h, rf(0.02));
  // 주근깨는 '둥근 점'만: 밝기 곡률(헤시안)의 두 고윳값이 모두 크면 점, 하나만 크면 선(가닥 머리카락·손가락 가장자리·주름)
  const dots = dotMask(rfL[0], w, h);
  // 콧구멍도 둥글고 어두워 점으로 잡힌다: 코 아래(콧방울·인중 위) 둘레는 뺀다
  {
    const nb = [64, 98, 240, 2, 460, 327, 294].map((k) => dp[k]);
    let nx0 = Infinity, ny0 = Infinity, nx1 = -Infinity, ny1 = -Infinity;
    for (const p of nb) { nx0 = Math.min(nx0, p.x); ny0 = Math.min(ny0, p.y); nx1 = Math.max(nx1, p.x); ny1 = Math.max(ny1, p.y); }
    const mg = fw * 0.06;
    for (let y = Math.max(0, Math.floor(ny0 - mg)); y <= Math.min(h - 1, Math.ceil(ny1 + mg)); y++) {
      for (let x = Math.max(0, Math.floor(nx0 - mg)); x <= Math.min(w - 1, Math.ceil(nx1 + mg)); x++) dots[y * w + x] = 0;
    }
  }
  // 아래 눈가 아이라인은 내 눈 선에 붙은 가는 띠만: 고른 사진 눈이 더 둥글면 그 아래 속눈썹 선·애교살 그늘이
  // 내 아래 눈꺼풀보다 아래로 와서 따로 떨어진 어두운 띠(다크서클처럼)가 된다. 위쪽(아이라인·날개)은 그대로
  const nearB = boxBlur(eyeOpen, w, h, rf(0.022));
  const lowerOK = new Float32Array(n).fill(1);
  for (const [ia, ib] of [[33, 133], [362, 263]]) {
    const A = dp[ia], B = dp[ib];
    const xa = Math.min(A.x, B.x) - fw * 0.06, xb = Math.max(A.x, B.x) + fw * 0.06;
    for (let y = 0; y < h; y++) for (let x = Math.max(0, Math.floor(xa)); x < Math.min(w, Math.ceil(xb)); x++) {
      const t = Math.max(0, Math.min(1, (x - A.x) / (B.x - A.x || 1)));
      const below = ramp(y - (A.y + (B.y - A.y) * t), 0, fw * 0.015);
      if (below <= 0) continue;
      const i = y * w + x;
      lowerOK[i] = Math.min(lowerOK[i], 1 - below * (1 - ramp(nearB[i], 0.02, 0.12)));
    }
  }
  // 그 아래는 가는 선(그린 아래 속눈썹)만: 얼굴 폭 1.2% 흐림보다 어두운 정도
  const rfThin = wblur(rfL[0], V, w, h, rf(0.012));
  // 입꼬리·입술 선 그늘도 둥글고 어두워 주근깨로 잡힌다(입술 칸 바로 밖이라 입술 제외에 안 걸림): 입술 둘레(얼굴 폭 3%)는 뺀다
  const mouthB = boxBlur(lipO, w, h, rf(0.03));
  // 주근깨: 고른 사진 손 바로 곁(얼굴 폭 4% 안 손 몫 6~9% 넘음)인 점은 뺀다. 칠한 손톱은 몸 피부로 안 잡히고 얼굴로 읽혀
  // 끝이 둥근 어두운 점이 턱·볼에 검은 점으로 옮겨졌다(손 지도는 얼굴 가장자리에도 조금 번져서 문턱을 낮추면 진짜 주근깨가 빠진다)
  const handsFar = rHands ? boxBlur(rHands, w, h, rf(0.04)) : null;
  // 주근깨 화장은 볼·코에 걸친 많은 점이다. 점이 몇 개뿐이면(맨얼굴의 점·여드름·수염 자국) 옮기지 않는다
  // (얼굴에 떨어진 몇 개의 검은 점·턱의 거뭇한 얼룩이 됐다): 얼굴 피부에서 뚜렷한 점 덩어리를 세어 40~80개 사이에서 켠다(주근깨 화장 사진 130개 남짓, 맨얼굴·다른 화장은 30개 아래: 모공·잡티)
  let nDots = 0;
  {
    const seed = new Uint8Array(n);
    for (let i = 0; i < n; i++) seed[i] = dots[i] > 0.5 && V[i] > 0.5 && faceM[i] > 0.5 && eyeM[i] < 0.2 && lipM[i] < 0.2 && rfL[0][i] - rfSkinMean[i] + 2.5 < 0 ? 1 : 0;
    const seen = new Uint8Array(n), st: number[] = [];
    for (let i0 = 0; i0 < n; i0++) {
      if (!seed[i0] || seen[i0]) continue;
      nDots++;
      seen[i0] = 1;
      st.push(i0);
      while (st.length) {
        const i = st.pop()!, x = i % w, y = (i / w) | 0;
        for (const j of [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, y > 0 ? i - w : -1, y < h - 1 ? i + w : -1]) if (j >= 0 && seed[j] && !seen[j]) { seen[j] = 1; st.push(j); }
      }
    }
  }
  const frOn = ramp(nDots, 40, 80);
  // 치아: 웃는 사진은 입 안 선이 보이는 이를 다 못 감싸 입술·피부 화장 자리에 이가 들어온다 → 입 둘레에서 내 입술 가운데
  // 밝기보다 12~24 넘게 밝고 색이 옅은(채도 14~24 아래) 픽셀은 화장 색을 입히지 않는다(이가 분홍으로 물들었다)
  let lipMedL = 0;
  {
    const v: number[] = [];
    for (let i = 0; i < n; i += 2) if (lipM[i] > 0.5) v.push(myL[0][i]);
    if (v.length) { v.sort((p, q) => p - q); lipMedL = v[v.length >> 1]; }
  }
  if (dbg) dbg.nDots = nDots;
  // ⑤ 옮기기
  const o = new Uint8ClampedArray(4);
  // 이: 입 둘레의 입술보다 L 12~24 밝고 색기 없는 픽셀. 크게 웃어 입술 안쪽 윤곽 점이 실제 입 벌림보다 작게 잡히면
  // 이 사이 틈·잇몸 선(어둡고 붉은)이 입술로 읽혀 립 색이 칠해져 이 둘레가 피 묻은 듯 빨간 선이 됐다(00121 × 07)
  // → 이 둘레(얼굴 폭 1%, 2px 이상)도 같이 지킨다(입술 안쪽 가장자리가 조금 원래 색으로 남는 것은 자연스럽다)
  const teethM = new Float32Array(n);
  if (lipMedL > 0) for (let i = 0; i < n; i++) {
    if (mouthB[i] <= 0) continue;
    teethM[i] = ramp(mouthB[i], 0.2, 0.5) * ramp(myL[0][i] - lipMedL, 12, 24) * (1 - ramp(Math.hypot(myL[1][i], myL[2][i]), 14, 24));
  }
  const teethNear = boxBlur(teethM, w, h, Math.max(2, rf(0.01)));
  for (let i = 0; i < n; i++) {
    // 눈 안(흰자·눈동자)은 어떤 화장도 옮기지 않는다
    const teeth = Math.max(teethM[i], ramp(teethNear[i], 0.04, 0.15) * ramp(mouthB[i], 0.2, 0.5));
    const keep = (1 - browM[i]) * ramp(Cf[i], 0.03, 0.15) * faceM[i] * (1 - ramp(openB[i], 0.5, 0.85)) * (1 - teeth);
    if (keep <= 0.002) continue;
    // 얼굴 위로 내려온 머리카락(피부 바탕보다 훨씬 어두운 픽셀)에는 화장 색을 입히지 않는다: 검은 앞머리가 붉게 물들었다
    // (눈가는 속눈썹·아이라인이라 빼지 않는다. 아이라인은 아래에서 따로 옮긴다)
    const hairK = 1 - ramp(myBase[0][i] - myL[0][i], 18, 38) * (1 - lipM[i]) * (1 - ramp(openB[i], 0.02, 0.2));
    const mAB = keep * hairK * Math.max(amt.skin * faceM[i], amt.lip * lipM[i], amt.eye * eyeM[i]);
    const mL = keep * hairK * Math.min(1, amt.lip * lipM[i] + amt.eye * eyeM[i] + amt.skin * 0.2 * faceM[i] * (1 - lipM[i]) * (1 - eyeM[i]));
    if (mAB <= 0.002 && mL <= 0.002) continue;
    let L = myL[0][i] + D[0][i] * mL;
    const a = myL[1][i] + D[1][i] * mAB, b = myL[2][i] + D[2][i] * mAB;
    // 아이라인·속눈썹(눈가의 어두운 잔무늬), 입술 광(입술의 밝은 잔무늬)
    const det = (rfL[0][i] - rfLow[0][i]) * keep * Vb[i];
    L += amt.lip * lipM[i] * Math.max(0, det) * 0.6;
    // 아이라인: 고른 사진의 '눈두덩보다 어두운 정도'를 내 눈두덩에(어둡게만)
    const liner = myLid[i] + Math.min(0, rfL[0][i] - rfLid[i]);
    const we = amt.eye * eyeM[i] * keep * Vb[i] * lowerOK[i];
    if (we > 0.002 && liner < L) L += (liner - L) * we;
    if (lowerOK[i] < 1) L += Math.min(0, rfL[0][i] - rfThin[i]) * amt.eye * eyeM[i] * keep * Vb[i] * (1 - lowerOK[i]);
    // 주근깨·점: 피부(눈가·입술 밖)에서 뚜렷한 작은 어두운 점
    const fr = rfL[0][i] - rfSkinMean[i] + 2.5;
    if (fr < 0 && dots[i] > 0 && frOn > 0) L += fr * 2 * dots[i] * frOn * amt.skin * keep * Vb[i] * (handsFar ? 1 - ramp(handsFar[i], 0.06, 0.09) : 1) * (1 - eyeM[i]) * (1 - ramp(mouthB[i], 0, 0.08));
    lab2rgb(Math.max(0, Math.min(100, L)), a, b, o, 0);
    const X = (i % w) + x0, Y = Math.floor(i / w) + y0;
    const q = (Y * W + X) * 4;
    out[q] = o[0]; out[q + 1] = o[1]; out[q + 2] = o[2];
  }
  return out;
}

/** 어두운 둥근 점(주근깨) 가중치 0~1: 살짝 흐린 밝기의 헤시안이 양의 정부호(둘레보다 오목)이고 두 방향 곡률이 비슷한 곳, 점 크기만큼 넓힘 */
export function dotMask(L: Float32Array, w: number, h: number): Float32Array {
  const S = boxBlur(boxBlur(L, w, h, 1), w, h, 1);
  const m = new Float32Array(w * h);
  for (let y = 2; y < h - 2; y++) for (let x = 2; x < w - 2; x++) {
    const i = y * w + x;
    const hxx = S[i + 2] - 2 * S[i] + S[i - 2], hyy = S[i + 2 * w] - 2 * S[i] + S[i - 2 * w];
    const hxy = (S[i + 2 + 2 * w] - S[i + 2 - 2 * w] - S[i - 2 + 2 * w] + S[i - 2 - 2 * w]) / 4;
    const tr = hxx + hyy, dd = Math.sqrt((hxx - hyy) ** 2 + 4 * hxy * hxy);
    const lmin = (tr - dd) / 2, lmax = (tr + dd) / 2;
    // 오목(둘레보다 어두운 점) + 둥근 정도(작은 곡률이 큰 곡률의 40% 넘게) + 뚜렷함
    if (lmin > 0.6 && lmin > 0.4 * lmax) m[i] = Math.min(1, lmin / 2);
  }
  const g = boxBlur(m, w, h, 2);
  for (let i = 0; i < g.length; i++) g[i] = Math.min(1, g[i] * 4);
  return g;
}
