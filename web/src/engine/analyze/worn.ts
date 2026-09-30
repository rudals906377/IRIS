// 모델 착용 사진 분석: 사진 속 모델의 관절점(자세 추정)과 '옷' 분할 영역에서
// 상의만 잘라 내고, 착용 엔진이 쓰는 기준점·부위 라벨을 만든다.
//
// - 상의와 하의가 모두 '옷'으로 분할되므로, 엉덩이 근처에서 색이 크게 바뀌는 줄을 밑단으로 본다.
// - 소매: 팔(어깨→팔꿈치→손목) 선에 가깝고 몸판 옆선(어깨→엉덩이) 바깥에 있는 옷 픽셀
// - 기준점: 어깨·겨드랑이는 관절점에서, 목둘레·밑단·소매 끝은 옷 윤곽에서 찾는다.
// 좌표는 모두 분석 이미지 픽셀. L/R은 착용자 기준(정면 사진에서 착용자 왼쪽 = 이미지 오른쪽)이며
// MediaPipe의 left_* 관절점과 같은 규칙이다.

import type { Vec2 } from '../math.ts';
import { largestComponent } from './background.ts';
import type { TopAnalysis } from './top.ts';

export interface Landmark {
  x: number;
  y: number;
  visibility?: number;
}

/** MediaPipe Pose 관절점 번호 */
const LM = { nose: 0, lShoulder: 11, rShoulder: 12, lElbow: 13, rElbow: 14, lWrist: 15, rWrist: 16, lHip: 23, rHip: 24 } as const;

function distToSeg(p: Vec2, a: Vec2, b: Vec2): { d: number; t: number } {
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const l2 = vx * vx + vy * vy || 1;
  const t = Math.max(0, Math.min(1, ((p.x - a.x) * vx + (p.y - a.y) * vy) / l2));
  return { d: Math.hypot(p.x - a.x - vx * t, p.y - a.y - vy * t), t };
}

/**
 * @param clothes 옷 확률(0~1) 또는 0/1 마스크
 * @param rgb     분석 이미지 RGBA
 * @param lm      관절점(분석 이미지 픽셀 좌표)
 */
export function analyzeWorn(clothes: ArrayLike<number>, rgb: Uint8ClampedArray, w: number, h: number, lm: Landmark[]): TopAnalysis | null {
  const warnings: string[] = ['모델 착용 사진에서 옷을 잘라 냈습니다'];
  const vis = (i: number): number => lm[i]?.visibility ?? 1;
  for (const i of [LM.lShoulder, LM.rShoulder, LM.lHip, LM.rHip]) {
    if (!lm[i] || vis(i) < 0.5) return null;
  }
  const sL = lm[LM.lShoulder];
  const sR = lm[LM.rShoulder];
  const hL = lm[LM.lHip];
  const hR = lm[LM.rHip];
  // 정면 사진만(착용자 왼쪽 어깨가 이미지 오른쪽)
  if (sL.x <= sR.x) return null;
  const SW = sL.x - sR.x;
  const shoulderY = (sL.y + sR.y) / 2;
  const hipY = (hL.y + hR.y) / 2;
  const T = hipY - shoulderY;
  if (T < SW * 0.6) return null;
  const cx = (sL.x + sR.x + hL.x + hR.x) / 4;

  let mask: Uint8Array = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = clothes[i] > 0.5 ? 1 : 0;

  // 팔 선(없으면 늘어뜨린 팔로 가정)
  const arm = (s: Landmark, e: number, wr: number, side: 1 | -1): Vec2[] => {
    const elbow = lm[e] && vis(e) > 0.3 ? lm[e] : { x: s.x + side * SW * 0.12, y: s.y + T * 0.55 };
    const wrist = lm[wr] && vis(wr) > 0.3 ? lm[wr] : { x: elbow.x, y: elbow.y + T * 0.5 };
    return [s, elbow, wrist];
  };
  const armL = arm(sL, LM.lElbow, LM.lWrist, 1);
  const armR = arm(sR, LM.rElbow, LM.rWrist, -1);
  const armRadius = SW * 0.24;
  const nearArm = (p: Vec2, a: Vec2[]): { d: number; s: number } => {
    const u = distToSeg(p, a[0], a[1]);
    const f = distToSeg(p, a[1], a[2]);
    return u.d <= f.d ? { d: u.d, s: u.t } : { d: f.d, s: 1 + f.t };
  };

  // 밑단: 엉덩이 근처 가운데 띠에서 위아래 색 차이가 가장 큰 줄
  const band = SW * 0.25;
  const rowColor = (y: number): [number, number, number, number] => {
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let x = Math.round(cx - band); x <= Math.round(cx + band); x++) {
      if (x < 0 || x >= w) continue;
      const i = y * w + x;
      if (!mask[i]) continue;
      r += rgb[i * 4];
      g += rgb[i * 4 + 1];
      b += rgb[i * 4 + 2];
      n++;
    }
    return n ? [r / n, g / n, b / n, n / (band * 2 + 1)] : [0, 0, 0, 0];
  };
  const y0 = Math.max(0, Math.round(hipY - T * 0.45));
  const y1 = Math.min(h - 1, Math.round(hipY + T * 0.5));
  const rows: [number, number, number, number][] = [];
  for (let y = y0; y <= y1; y++) rows.push(rowColor(y));
  const win = Math.max(3, Math.round(T * 0.04));
  let hemY = -1;
  let bestStep = 0;
  for (let k = win; k < rows.length - win; k++) {
    let a = [0, 0, 0];
    let b = [0, 0, 0];
    let na = 0;
    let nb = 0;
    for (let j = k - win; j < k; j++) if (rows[j][3] > 0.5) (a = a.map((v, c) => v + rows[j][c])), na++;
    for (let j = k + 1; j <= k + win; j++) if (rows[j][3] > 0.5) (b = b.map((v, c) => v + rows[j][c])), nb++;
    // 아래쪽에 옷이 거의 없으면(하의가 옷으로 안 잡힘·맨살) 그곳이 밑단
    if (na >= win * 0.7 && nb < win * 0.3 && bestStep < 18) {
      hemY = y0 + k;
      bestStep = Infinity;
      break;
    }
    if (na < win * 0.7 || nb < win * 0.7) continue;
    const step = Math.hypot(a[0] / na - b[0] / nb, a[1] / na - b[1] / nb, a[2] / na - b[2] / nb);
    if (step > bestStep) {
      bestStep = step;
      hemY = y0 + k;
    }
  }
  if (hemY < 0 || bestStep < 18) {
    // 상반신 사진처럼 옷이 사진 아래 끝까지 이어지면 자르지 않는다.
    const bottomCovered = rowColor(h - 1)[3] > 0.5 && h - 1 < hipY + T * 0.5;
    hemY = bottomCovered ? h - 1 : Math.round(hipY + T * 0.12);
    if (!bottomCovered) warnings.push('밑단을 뚜렷이 찾지 못해 엉덩이 높이로 추정했습니다');
  }
  // 소매 길이: 팔 선을 따라가다 옷이 끝나고 맨살이 나오는 곳(팔 선 매개변수 0~2: 어깨→팔꿈치→손목)
  const pointOnArm = (a: Vec2[], t: number): Vec2 => {
    const [p, q] = t <= 1 ? [a[0], a[1]] : [a[1], a[2]];
    const u = t <= 1 ? t : t - 1;
    return { x: p.x + (q.x - p.x) * u, y: p.y + (q.y - p.y) * u };
  };
  // 몸통 대표 색(가운데 띠의 중앙값)
  const torsoColor: number[] = (() => {
    const cs: number[][] = [[], [], []];
    for (let y = Math.round(shoulderY + T * 0.25); y < Math.round(shoulderY + T * 0.6); y += 2) {
      for (let x = Math.round(cx - SW * 0.2); x <= Math.round(cx + SW * 0.2); x += 2) {
        if (x < 0 || x >= w || y < 0 || y >= h || !mask[y * w + x]) continue;
        const i = y * w + x;
        for (let c = 0; c < 3; c++) cs[c].push(rgb[i * 4 + c]);
      }
    }
    return cs.map((a) => (a.length ? a.sort((p, q) => p - q)[a.length >> 1] : 128));
  })();
  // 겹쳐 입은 속옷(반팔 안의 긴팔 등)도 '옷'으로 분할되므로, 소매 윗부분 색에서 크게 달라지는 곳도 소매 끝으로 본다.
  const sleeveEnd = (a: Vec2[]): number => {
    let miss = 0;
    let changed = 0;
    let ref: [number, number, number] | null = null;
    const acc = [0, 0, 0, 0];
    for (let t = 0.15; t <= 2; t += 0.02) {
      const p = pointOnArm(a, t);
      const x = Math.round(p.x);
      const y = Math.round(p.y);
      let n = 0;
      let on = 0;
      const c = [0, 0, 0];
      for (let dy = -2; dy <= 2; dy++) {
        for (let dx = -2; dx <= 2; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue;
          n++;
          const i = yy * w + xx;
          if (!mask[i]) continue;
          on++;
          c[0] += rgb[i * 4];
          c[1] += rgb[i * 4 + 1];
          c[2] += rgb[i * 4 + 2];
        }
      }
      if (!n || on / n < 0.4) {
        if (++miss >= 3) return t - 0.04;
        continue;
      }
      miss = 0;
      const col = [c[0] / on, c[1] / on, c[2] / on];
      if (t < 0.4) {
        acc[0] += col[0];
        acc[1] += col[1];
        acc[2] += col[2];
        acc[3]++;
        continue;
      }
      ref ??= [acc[0] / acc[3], acc[1] / acc[3], acc[2] / acc[3]];
      // 그림자는 밝기만 바꾸고, 겹쳐 입은 다른 옷은 색조가 바뀐다: 밝기를 뺀 색조 차이 또는 아주 큰 색 차이
      const chroma = (v: number[]): number[] => {
        const m = (v[0] + v[1] + v[2]) / 3;
        return [v[0] - m, v[1] - m, v[2] - m];
      };
      const ca = chroma(col);
      const cb = chroma(ref);
      const hueDiff = Math.hypot(ca[0] - cb[0], ca[1] - cb[1], ca[2] - cb[2]);
      const ct = chroma(torsoColor);
      const torsoHue = Math.hypot(ca[0] - ct[0], ca[1] - ct[1], ca[2] - ct[2]);
      const torsoDiff = Math.hypot(col[0] - torsoColor[0], col[1] - torsoColor[1], col[2] - torsoColor[2]);
      // 같은 옷의 그림자·주름이면 몸통 색과도 비슷하다. 몸통 색과도 달라야 다른 옷으로 본다.
      const otherGarment = torsoHue > 16 || torsoDiff > 70;
      if (otherGarment && (hueDiff > 16 || Math.hypot(col[0] - ref[0], col[1] - ref[1], col[2] - ref[2]) > 70)) {
        if (++changed >= 4) return t - 0.06;
      } else changed = 0;
    }
    return 2;
  };
  const endL = sleeveEnd(armL);
  const endR = sleeveEnd(armR);
  // 밑단 위(상의)·아래(하의)의 대표 색
  const meanColor = (ya: number, yb: number): [number, number, number] | null => {
    let r = 0;
    let g = 0;
    let b = 0;
    let n = 0;
    for (let y = Math.max(0, ya); y <= Math.min(h - 1, yb); y++) {
      const c = rowColor(y);
      if (c[3] < 0.3) continue;
      r += c[0];
      g += c[1];
      b += c[2];
      n++;
    }
    return n ? [r / n, g / n, b / n] : null;
  };
  const topColor = meanColor(Math.round(hemY - T * 0.25), hemY - 2);
  const bottomColor = meanColor(hemY + 2, Math.round(hemY + T * 0.25));
  // 밑단 아래는 지우되, 늘어뜨린 소매(팔 근처이면서 하의보다 상의 색에 가까운 것)는 남긴다.
  for (let y = hemY + 1; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const p = { x, y };
      const d = Math.min(nearArm(p, armL).d, nearArm(p, armR).d);
      const oa = nearArm(p, armL);
      const ob = nearArm(p, armR);
      let keep = d <= armRadius * 0.8 && ((oa.d <= ob.d && oa.s <= endL + 0.03) || (ob.d < oa.d && ob.s <= endR + 0.03));
      if (keep && topColor && bottomColor) {
        const c = [rgb[i * 4], rgb[i * 4 + 1], rgb[i * 4 + 2]];
        const dt = Math.hypot(c[0] - topColor[0], c[1] - topColor[1], c[2] - topColor[2]);
        const db = Math.hypot(c[0] - bottomColor[0], c[1] - bottomColor[1], c[2] - bottomColor[2]);
        keep = dt < db;
      }
      if (!keep) mask[i] = 0;
    }
  }
  // 어깨보다 한참 위(머리 주변 옷 오검출)도 지운다.
  const headCut = shoulderY - SW * 0.35;
  for (let y = 0; y < Math.max(0, headCut); y++) mask.fill(0, y * w, (y + 1) * w);
  mask = largestComponent(mask, w, h);



  // 몸판 옆선: 어깨 관절 → 엉덩이 관절
  const sideX = (s: Landmark, hp: Landmark, y: number): number => s.x + ((hp.x - s.x) * (y - s.y)) / (hp.y - s.y || 1);
  // 소매 = 팔 선 둘레의 띠(팔 두께 정도)에서 소매 끝까지. 팔 바깥쪽은 조금 더 넓게(풍성한 소매).
  const bandIn = SW * 0.17;
  const bandOut = SW * 0.3;
  // 윤곽 윗선
  const ytop = (x: number): number => {
    const xi = Math.round(x);
    if (xi < 0 || xi >= w) return NaN;
    for (let y = 0; y < h; y++) if (mask[y * w + xi]) return y;
    return NaN;
  };
  const clampY = (y: number, lo: number, hi: number, fallback: number): number => (Number.isFinite(y) && y >= lo && y <= hi ? y : fallback);
  const shoulderAt = (s: Landmark, side: 1 | -1): Vec2 => {
    // 어깨 봉제선: 관절보다 조금 바깥의 옷 윗선(소매 곡면의 윗 모서리)
    const x = s.x + side * SW * 0.1;
    return { x, y: clampY(ytop(x), s.y - SW * 0.25, s.y + SW * 0.1, s.y - SW * 0.05) };
  };
  const shoulderL = shoulderAt(sL, 1);
  const shoulderR = shoulderAt(sR, -1);
  // 겨드랑이: 팔 안쪽 선 위. 팔을 내린 정면 사진에서는 진동 둘레가 팔과 나란해
  // 실제 높이에 두면 소매 곡면이 납작해지므로 조금 높게 둔다.
  const armpitY = shoulderY + SW * 0.3;
  const armpitL = { x: sL.x - SW * 0.1, y: armpitY };
  const armpitR = { x: sR.x + SW * 0.1, y: armpitY };
  // 목둘레: 가운데 양쪽의 옷 윗선(깃), 앞 목선: 가운데 옷 윗선의 가장 낮은 곳
  const neckAt = (side: 1 | -1): Vec2 => {
    const x = cx + side * SW * 0.2;
    return { x, y: clampY(ytop(x), shoulderY - SW * 0.4, shoulderY + SW * 0.15, shoulderY - SW * 0.08) };
  };
  const neckL = neckAt(1);
  const neckR = neckAt(-1);
  let neckFront: Vec2 = { x: cx, y: (neckL.y + neckR.y) / 2 + SW * 0.12 };
  {
    let lowest = -1;
    for (let x = Math.round(cx - SW * 0.08); x <= Math.round(cx + SW * 0.08); x++) {
      const y = ytop(x);
      if (Number.isFinite(y) && y > lowest) lowest = y;
    }
    if (lowest > 0 && lowest < shoulderY + SW * 0.45) neckFront = { x: cx, y: lowest };
  }
  const labels = new Uint8Array(w * h);
  let nL = 0;
  let nR = 0;
  const inSleeve = (p: Vec2, a: Vec2[], end: number, side: 1 | -1, s: Landmark, hp: Landmark, sh: Vec2, pit: Vec2): boolean => {
    // 겨드랑이 위: 진동 둘레선(어깨점 → 겨드랑이) 바깥이 소매(평평한 상품 사진 분석과 같은 규칙)
    if (p.y <= pit.y) {
      const t = Math.max(0, Math.min(1, (p.y - sh.y) / (pit.y - sh.y || 1)));
      return (p.x - (sh.x + (pit.x - sh.x) * t)) * side > 0 && p.y >= sh.y - SW * 0.1;
    }
    const na = nearArm(p, a);
    if (na.s < 0.08 || na.s > end + 0.03) return false;
    const outward = (p.x - sideX(s, hp, p.y)) * side > 0;
    // 몸판 옆선 바깥이면서 팔 선 근처(넓은 반소매 포함)는 소매
    if (outward && na.d < SW * 0.5) return true;
    if (!outward && na.s < 0.3) return false; // 어깨 안쪽은 몸판
    const q = pointOnArm(a, na.s);
    const outer = (p.x - q.x) * side > 0;
    return na.d < (outer ? bandOut : bandIn);
  };
  // 소매 끝을 지나 팔 위에 남은 옷(안에 겹쳐 입은 긴팔 소매)은 이 상품이 아니므로 지운다.
  const beyondSleeve = (p: Vec2, a: Vec2[], end: number, side: 1 | -1, s: Landmark, hp: Landmark): boolean => {
    if (end >= 1.95) return false;
    const na = nearArm(p, a);
    return na.s > end + 0.06 && na.d < bandOut && (p.x - sideX(s, hp, p.y)) * side > 0;
  };
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    const p = { x: i % w, y: (i / w) | 0 };
    if (beyondSleeve(p, armL, endL, 1, sL, hL) || beyondSleeve(p, armR, endR, -1, sR, hR)) {
      mask[i] = 0;
      continue;
    }
    let lab = 1;
    if (endL > 0.25 && inSleeve(p, armL, endL, 1, sL, hL, shoulderL, armpitL)) lab = 2;
    else if (endR > 0.25 && inSleeve(p, armR, endR, -1, sR, hR, shoulderR, armpitR)) lab = 3;
    labels[i] = lab;
    if (lab === 2) nL++;
    else if (lab === 3) nR++;
  }

  // 밑단 양 끝: 밑단 줄에서 몸판(라벨 1) 구간
  const hemRow = Math.max(0, hemY - 2);
  let hx0 = -1;
  let hx1 = -1;
  for (let x = 0; x < w; x++) {
    if (labels[hemRow * w + x] === 1) {
      if (hx0 < 0) hx0 = x;
      hx1 = x;
    }
  }
  if (hx0 < 0) return null;
  // 사진 아래 끝에서 잘린 옷(상반신 사진): 실제 밑단을 모른다.
  if (hemRow >= h - 4) {
    warnings.push('밑단이 사진 밖으로 잘렸습니다');
  }
  const hemL = { x: hx1, y: hemRow };
  const hemR = { x: hx0, y: hemRow };

  // 소매 끝: 팔 선을 따라 가장 먼 소매 픽셀들의 바깥·안쪽
  const cuff = (id: number, a: Vec2[], side: 1 | -1): { outer: Vec2; inner: Vec2; s: number } | null => {
    let maxS = -1;
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== id) continue;
      const s = nearArm({ x: i % w, y: (i / w) | 0 }, a).s;
      if (s > maxS) maxS = s;
    }
    if (maxS < 0.1) return null;
    const seg = maxS > 1 ? [a[1], a[2]] : [a[0], a[1]];
    const dx = seg[1].x - seg[0].x;
    const dy = seg[1].y - seg[0].y;
    const dl = Math.hypot(dx, dy) || 1;
    // 바깥쪽 법선: 몸 중심에서 멀어지는 쪽
    let nx = -dy / dl;
    let ny = dx / dl;
    if (nx * side < 0) {
      nx = -nx;
      ny = -ny;
    }
    let outer = { x: 0, y: 0, t: -Infinity };
    let inner = { x: 0, y: 0, t: Infinity };
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== id) continue;
      const p = { x: i % w, y: (i / w) | 0 };
      if (nearArm(p, a).s < maxS - 0.08) continue;
      const t = p.x * nx + p.y * ny;
      if (t > outer.t) outer = { ...p, t };
      if (t < inner.t) inner = { ...p, t };
    }
    return { outer: { x: outer.x, y: outer.y }, inner: { x: inner.x, y: inner.y }, s: maxS };
  };
  const minSleeve = SW * SW * 0.02;
  const cL = nL > minSleeve ? cuff(2, armL, 1) : null;
  const cR = nR > minSleeve ? cuff(3, armR, -1) : null;
  const keypoints: Record<string, Vec2> = { neckL, neckR, neckFront, shoulderL, shoulderR, armpitL, armpitR, hemL, hemR };
  let sleeve: TopAnalysis['sleeve'] = 'none';
  if (cL && cR) {
    keypoints.sleeveOuterL = cL.outer;
    keypoints.sleeveInnerL = cL.inner;
    keypoints.sleeveOuterR = cR.outer;
    keypoints.sleeveInnerR = cR.inner;
    sleeve = (endL + endR) / 2 > 1.3 ? 'long' : 'short';
  } else {
    for (let i = 0; i < labels.length; i++) if (labels[i] === 2 || labels[i] === 3) labels[i] = 1;
    warnings.push('소매를 찾지 못해 민소매로 처리했습니다');
  }

  // 신뢰도: 관절점 확신도 × 몸통 영역이 옷으로 덮인 비율
  let inTorso = 0;
  let covered = 0;
  for (let y = Math.round(shoulderY + T * 0.15); y < Math.min(hemY, Math.round(hipY)); y += 2) {
    const x0 = Math.round(sideX(sR, hR, y) + SW * 0.1);
    const x1 = Math.round(sideX(sL, hL, y) - SW * 0.1);
    for (let x = x0; x <= x1; x += 2) {
      if (x < 0 || x >= w) continue;
      inTorso++;
      if (labels[y * w + x]) covered++;
    }
  }
  const coverage = inTorso ? covered / inTorso : 0;
  const lmConf = Math.min(vis(LM.lShoulder), vis(LM.rShoulder), vis(LM.lHip), vis(LM.rHip));
  let confidence = Math.min(1, coverage / 0.8) * Math.min(1, lmConf / 0.8);
  if (coverage < 0.5) warnings.push('몸통의 옷 영역이 작습니다(팔·머리카락에 가려졌을 수 있음)');
  confidence = Math.max(0, Math.min(1, confidence));
  // 착용 사진은 몸 앞면 폭만 보이므로(옆으로 감긴 부분이 안 보임) 평평하게 놓은 옷보다 좁다.
  // 어깨점 폭(관절 폭의 1.2배)을 기준으로 조금 넉넉하게 펴서 입힌다.
  return { keypoints, labels, sleeve, confidence, warnings, widthScale: 1.3 };
}
