// 상의 상품 사진 자동 분석: 옷 윤곽(마스크)에서 착용 엔진이 쓰는 기준점과 부위 라벨을 찾는다.
//
// 기준점(착용자 기준 L/R — 앞면 사진에서 착용자 왼쪽은 이미지 오른쪽):
//   neckL/R      목둘레 양 끝(윤곽 윗선에서 중심 양쪽의 가장 높은 곳)
//   neckFront    앞 목선 가장 낮은 곳(파여 있지 않으면 목 폭으로 추정)
//   armpitL/R    소매 아랫선과 몸판 옆선이 만나는 곳(몸판과 소매 사이 틈의 꼭대기)
//   shoulderL/R  어깨 봉제선 끝(겨드랑이 바로 위의 윤곽 윗선)
//   sleeveOuter/Inner  소매 끝단 바깥·안쪽 모서리
//   hemL/R       밑단 양 끝
// 부위 라벨: 1 몸판, 2 착용자 왼쪽 소매, 3 오른쪽 소매, 4 목 안쪽

import type { Vec2 } from '../math.ts';

export interface TopAnalysis {
  keypoints: Record<string, Vec2>;
  labels: Uint8Array;
  sleeve: 'short' | 'long' | 'none';
  /** 0~1, 낮으면 사용자에게 기준점 확인을 권한다. */
  confidence: number;
  warnings: string[];
}

interface Run {
  x0: number;
  x1: number;
}

function rowRuns(mask: Uint8Array, w: number, y: number): Run[] {
  const runs: Run[] = [];
  let x = 0;
  const row = y * w;
  while (x < w) {
    while (x < w && !mask[row + x]) x++;
    if (x >= w) break;
    const x0 = x;
    while (x < w && mask[row + x]) x++;
    runs.push({ x0, x1: x - 1 });
  }
  return runs;
}

function runAt(runs: Run[], x: number): Run | undefined {
  return runs.find((r) => r.x0 <= x && x <= r.x1);
}

export function analyzeTop(mask: Uint8Array, w: number, h: number): TopAnalysis | null {
  const warnings: string[] = [];
  // 경계 상자
  let top = h;
  let bottom = -1;
  let left = w;
  let right = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (!mask[y * w + x]) continue;
      if (y < top) top = y;
      if (y > bottom) bottom = y;
      if (x < left) left = x;
      if (x > right) right = x;
    }
  }
  if (bottom < 0) return null;
  const H = bottom - top + 1;
  const W = right - left + 1;
  if (H < 40 || W < 40) return null;

  // 폭 2px 이하 조각은 가장자리 잡음으로 보고 무시한다(소매로 오인 방지).
  const minRun = Math.max(3, Math.round(W * 0.012));
  const runs: Run[][] = [];
  for (let y = 0; y < h; y++) runs.push(rowRuns(mask, w, y).filter((r) => r.x1 - r.x0 + 1 >= minRun));

  // 몸판 중심: 아래쪽 40% 구간에서 가장 넓은 구간의 중점들의 중앙값
  const mids: number[] = [];
  for (let y = Math.round(top + H * 0.6); y <= bottom; y++) {
    const rs = runs[y];
    if (!rs.length) continue;
    const widest = rs.reduce((a, b) => (b.x1 - b.x0 > a.x1 - a.x0 ? b : a));
    mids.push((widest.x0 + widest.x1) / 2);
  }
  mids.sort((a, b) => a - b);
  const cx = mids.length ? mids[mids.length >> 1] : (left + right) / 2;

  // 밑단: 맨 아래 2% 위치의 몸판 구간
  const hemY = Math.round(bottom - H * 0.02);
  const hemRun = runAt(runs[hemY], Math.round(cx)) ?? runs[hemY][0];
  if (!hemRun) return null;
  const hemL: Vec2 = { x: hemRun.x1, y: hemY };
  const hemR: Vec2 = { x: hemRun.x0, y: hemY };
  const hemHalf = (hemRun.x1 - hemRun.x0) / 2;

  // 몸판 가장자리 추적(아래 → 위): 각 행에서 중심을 포함한 구간
  const torsoEdge: { y: number; x0: number; x1: number; right: boolean; leftSide: boolean }[] = [];
  for (let y = bottom; y >= top; y--) {
    const r = runAt(runs[y], Math.round(cx));
    if (!r) {
      torsoEdge.push({ y, x0: NaN, x1: NaN, right: false, leftSide: false });
      continue;
    }
    const rs = runs[y];
    torsoEdge.push({
      y,
      x0: r.x0,
      x1: r.x1,
      right: rs.some((q) => q.x0 > r.x1), // 오른쪽(착용자 왼쪽)에 떨어진 조각 = 소매
      leftSide: rs.some((q) => q.x1 < r.x0),
    });
  }

  // 겨드랑이: 몸판 옆에 소매 조각이 떨어져 있는 구간의 꼭대기. 틈이 없으면 옆선이 급히 벌어지는 곳.
  const findArmpit = (side: 1 | -1): Vec2 | null => {
    const hasGap = (e: (typeof torsoEdge)[number]): boolean => (side === 1 ? e.right : e.leftSide);
    const edgeX = (e: (typeof torsoEdge)[number]): number => (side === 1 ? e.x1 : e.x0);
    // 1) 틈 구간(아래에서 위로 올라가며 틈이 있다가 사라지는 곳)
    let gapTop: number | null = null;
    for (let i = 0; i < torsoEdge.length; i++) {
      const e = torsoEdge[i];
      if (e.y > top + H * 0.85 || e.y < top + H * 0.08) continue;
      if (hasGap(e)) gapTop = i;
    }
    if (gapTop !== null) {
      const e = torsoEdge[gapTop];
      return { x: edgeX(e), y: e.y };
    }
    // 2) 민소매: 옆선을 따라 올라가다 진동 둘레가 급히 안쪽으로 굽는 곳
    {
      const win = Math.max(3, Math.round(H * 0.04));
      for (let i = win; i < torsoEdge.length - win; i++) {
        const e = torsoEdge[i];
        if (e.y > top + H * 0.75) continue;
        if (e.y < top + H * 0.12) break;
        const a = edgeX(torsoEdge[i - win]);
        const b = edgeX(torsoEdge[i + win]);
        if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
        // 위(i+win)로 갈수록 몸 중심 쪽으로 들어가는 기울기
        const inward = (side === 1 ? a - b : b - a) / (2 * win);
        if (inward > 0.45) return { x: edgeX(e), y: e.y };
      }
    }
    // 3) 옆선 급변: 아래에서 위로 올라가며 폭이 갑자기 커지는 곳
    let prev = NaN;
    for (let i = 0; i < torsoEdge.length; i++) {
      const e = torsoEdge[i];
      if (e.y > top + H * 0.9) {
        prev = edgeX(e);
        continue;
      }
      const x = edgeX(e);
      if (Number.isFinite(prev) && Math.abs(x - prev) > W * 0.025) {
        return { x: prev, y: e.y + 1 };
      }
      prev = x;
      if (e.y < top + H * 0.1) break;
    }
    return null;
  };

  const armpitL = findArmpit(1);
  const armpitR = findArmpit(-1);
  if (!armpitL || !armpitR) {
    warnings.push('겨드랑이 위치를 찾지 못했습니다');
    return null;
  }

  // 윤곽 윗선
  const ytop = new Float32Array(w).fill(NaN);
  for (let x = left; x <= right; x++) {
    for (let y = top; y <= bottom; y++) {
      if (mask[y * w + x]) {
        ytop[x] = y;
        break;
      }
    }
  }
  const torsoHalf = Math.max(8, (armpitL.x - armpitR.x) / 2);

  // 목둘레 양 끝: 중심에서 몸판 반폭의 5~45% 떨어진 곳 중 가장 높은 점
  const highest = (x0: number, x1: number): Vec2 => {
    let best = { x: Math.round((x0 + x1) / 2), y: Infinity };
    const a = Math.round(Math.min(x0, x1));
    const b = Math.round(Math.max(x0, x1));
    for (let x = a; x <= b; x++) {
      if (Number.isFinite(ytop[x]) && ytop[x] < best.y) best = { x, y: ytop[x] };
    }
    return best;
  };
  const neckL = highest(cx + torsoHalf * 0.05, cx + torsoHalf * 0.45);
  const neckR = highest(cx - torsoHalf * 0.45, cx - torsoHalf * 0.05);
  const neckW = Math.max(4, neckL.x - neckR.x);
  const neckY = (neckL.y + neckR.y) / 2;
  const dipY = ytop[Math.round(cx)];
  let neckFront: Vec2;
  let hasOpenNeck = false;
  if (Number.isFinite(dipY) && dipY - neckY > neckW * 0.12) {
    neckFront = { x: cx, y: dipY };
    hasOpenNeck = true;
  } else {
    // 마네킹 사진처럼 목 안쪽(뒤판)이 보이면 앞 목선을 목 폭으로 추정한다.
    neckFront = { x: cx, y: neckY + neckW * 0.42 };
  }

  // 어깨 봉제선 끝: 겨드랑이보다 조금 안쪽 x의 윤곽 윗선
  const shoulderAt = (armpit: Vec2, side: 1 | -1): Vec2 => {
    const x = Math.round(armpit.x - side * torsoHalf * 0.08);
    const y = Number.isFinite(ytop[x]) ? ytop[x] : neckY + (armpit.y - neckY) * 0.25;
    return { x, y };
  };
  const shoulderL = shoulderAt(armpitL, 1);
  const shoulderR = shoulderAt(armpitR, -1);

  // 부위 라벨
  const labels = new Uint8Array(w * h);
  // 진동 둘레선(어깨 → 겨드랑이) 바깥쪽이 소매
  const armholeX = (s: Vec2, a: Vec2, y: number): number => {
    if (a.y === s.y) return a.x;
    const t = Math.min(1, Math.max(0, (y - s.y) / (a.y - s.y)));
    return s.x + (a.x - s.x) * t;
  };
  let sleeveCountL = 0;
  let sleeveCountR = 0;
  for (let y = top; y <= bottom; y++) {
    const rs = runs[y];
    const torso = runAt(rs, Math.round(cx));
    for (const r of rs) {
      for (let x = r.x0; x <= r.x1; x++) {
        const i = y * w + x;
        let lab = 1;
        if (y <= armpitL.y && x > armholeX(shoulderL, armpitL, y)) lab = 2;
        else if (y > armpitL.y && x > cx && r !== torso) lab = 2;
        if (y <= armpitR.y && x < armholeX(shoulderR, armpitR, y)) lab = 3;
        else if (y > armpitR.y && x < cx && r !== torso) lab = 3;
        labels[i] = lab;
        if (lab === 2) sleeveCountL++;
        if (lab === 3) sleeveCountR++;
      }
    }
  }

  // 소매는 진동 둘레에 붙은 가장 큰 덩어리만 인정하고, 나머지 조각은 몸판으로 돌린다.
  for (const partId of [2, 3]) {
    const comp = largestLabelComponent(labels, w, h, partId);
    for (let i = 0; i < labels.length; i++) if (labels[i] === partId && !comp[i]) labels[i] = 1;
  }
  sleeveCountL = 0;
  sleeveCountR = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] === 2) sleeveCountL++;
    else if (labels[i] === 3) sleeveCountR++;
  }

  // 목 안쪽: 목둘레 양 끝과 앞 목선을 지나는 포물선 위쪽
  if (!hasOpenNeck) {
    for (let x = Math.ceil(neckR.x); x <= Math.floor(neckL.x); x++) {
      const t = (x - cx) / (neckW / 2);
      const curveY = neckFront.y - (neckFront.y - neckY) * t * t;
      for (let y = top; y < curveY && y <= bottom; y++) {
        const i = y * w + x;
        if (labels[i] === 1) labels[i] = 4;
      }
    }
  }

  // 소매 끝단: 소매 픽셀을 소매 축에 투영해 가장 먼 쪽의 바깥·안쪽 모서리
  const sleeveEnds = (partId: number, shoulder: Vec2, armpit: Vec2): { outer: Vec2; inner: Vec2; length: number } | null => {
    const root = { x: (shoulder.x + armpit.x) / 2, y: (shoulder.y + armpit.y) / 2 };
    let sx = 0;
    let sy = 0;
    let n = 0;
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== partId) continue;
      sx += i % w;
      sy += Math.floor(i / w);
      n++;
    }
    if (n < W * H * 0.004) return null;
    const c = { x: sx / n, y: sy / n };
    let ax = c.x - root.x;
    let ay = c.y - root.y;
    const al = Math.hypot(ax, ay) || 1;
    ax /= al;
    ay /= al;
    // 바깥쪽 법선: 어깨 쪽
    let nx = -ay;
    let ny = ax;
    if (nx * (shoulder.x - root.x) + ny * (shoulder.y - root.y) < 0) {
      nx = -nx;
      ny = -ny;
    }
    let maxS = -Infinity;
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== partId) continue;
      const s = (i % w - root.x) * ax + (Math.floor(i / w) - root.y) * ay;
      if (s > maxS) maxS = s;
    }
    let outer = { x: root.x, y: root.y, t: -Infinity };
    let inner = { x: root.x, y: root.y, t: Infinity };
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== partId) continue;
      const px = i % w;
      const py = Math.floor(i / w);
      const s = (px - root.x) * ax + (py - root.y) * ay;
      if (s < maxS * 0.9) continue;
      const t = (px - root.x) * nx + (py - root.y) * ny;
      if (t > outer.t) outer = { x: px, y: py, t };
      if (t < inner.t) inner = { x: px, y: py, t };
    }
    return { outer: { x: outer.x, y: outer.y }, inner: { x: inner.x, y: inner.y }, length: maxS };
  };

  const endL = sleeveCountL > 0 ? sleeveEnds(2, shoulderL, armpitL) : null;
  const endR = sleeveCountR > 0 ? sleeveEnds(3, shoulderR, armpitR) : null;
  let sleeve: TopAnalysis['sleeve'] = 'none';
  const keypoints: Record<string, Vec2> = { neckL, neckR, neckFront, shoulderL, shoulderR, armpitL, armpitR, hemL, hemR };
  if (endL && endR) {
    keypoints.sleeveOuterL = endL.outer;
    keypoints.sleeveInnerL = endL.inner;
    keypoints.sleeveOuterR = endR.outer;
    keypoints.sleeveInnerR = endR.inner;
    const len = (endL.length + endR.length) / 2;
    sleeve = len > H * 0.45 ? 'long' : 'short';
  } else {
    // 민소매: 소매 픽셀은 몸판으로 되돌린다
    for (let i = 0; i < labels.length; i++) if (labels[i] === 2 || labels[i] === 3) labels[i] = 1;
    warnings.push('소매를 찾지 못해 민소매로 처리했습니다');
  }

  // 신뢰도: 좌우 대칭성과 비율로 간단히 평가
  const symArm = 1 - Math.min(1, Math.abs(armpitL.y - armpitR.y) / (H * 0.1));
  const symCenter = 1 - Math.min(1, Math.abs((armpitL.x + armpitR.x) / 2 - cx) / (W * 0.08));
  const ratio = hemHalf * 2 / Math.max(1, armpitL.x - armpitR.x);
  const ratioOk = ratio > 0.7 && ratio < 1.5 ? 1 : 0.4;
  const confidence = Math.max(0, Math.min(1, symArm * 0.4 + symCenter * 0.4 + ratioOk * 0.2));
  if (confidence < 0.6) warnings.push('자동 분석 신뢰도가 낮습니다. 기준점을 확인해 주세요');

  return { keypoints, labels, sleeve, confidence, warnings };
}

function largestLabelComponent(labels: Uint8Array, w: number, h: number, id: number): Uint8Array {
  const seen = new Int32Array(w * h);
  const queue = new Int32Array(w * h);
  let best = 0;
  let bestSize = 0;
  let next = 1;
  for (let s = 0; s < w * h; s++) {
    if (labels[s] !== id || seen[s]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = s;
    seen[s] = next;
    while (head < tail) {
      const i = queue[head++];
      const x = i % w;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i >= w ? i - w : -1, i < w * (h - 1) ? i + w : -1];
      for (const j of nb) {
        if (j >= 0 && labels[j] === id && !seen[j]) {
          seen[j] = next;
          queue[tail++] = j;
        }
      }
    }
    if (tail > bestSize) {
      bestSize = tail;
      best = next;
    }
    next++;
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = seen[i] === best && best > 0 ? 1 : 0;
  return out;
}
