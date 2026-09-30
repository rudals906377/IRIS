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
  /** 가로 배율 보정(ProductInfo.widthScale) */
  widthScale?: number;
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

/**
 * @param rgba 사진 픽셀(있으면 소매를 몸판 옆에 접어 붙인 사진에서 안쪽 접힘선을 찾는 데 쓴다)
 */
export function analyzeTop(mask0: Uint8Array, w: number, h: number, rgba?: Uint8ClampedArray): TopAnalysis | null {
  const warnings: string[] = [];
  // 앞이 열린 지퍼 재킷·카디건: 가운데의 세로 틈을 메운 윤곽으로 형태를 분석한다(라벨은 원래 윤곽에만).
  const mask = closeCenterGap(mask0, w, h);
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

  // 밑단: 몸판 중심선이 끝나는 곳(옆에 늘어진 끈·리본은 제외)에서 위로 25% 구간 중
  // 중심을 포함한 몸판 구간이 가장 넓은 곳의 폭을 쓴다(셔츠의 둥근 밑단 모서리 대비).
  let torsoBottom = bottom;
  while (torsoBottom > top && !runAt(runs[torsoBottom], Math.round(cx))) torsoBottom--;
  let hemRun: Run | undefined;
  for (let y = Math.round(torsoBottom - H * 0.25); y <= torsoBottom; y++) {
    const r = runAt(runs[y], Math.round(cx));
    if (r && (!hemRun || r.x1 - r.x0 >= hemRun.x1 - hemRun.x0)) hemRun = r;
  }
  if (!hemRun) return null;
  const hemY = torsoBottom - Math.round(H * 0.01);
  const hemL0: Vec2 = { x: hemRun.x1, y: hemY };
  const hemR0: Vec2 = { x: hemRun.x0, y: hemY };

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

  // 겨드랑이 후보 두 가지: (가) 볼록 껍질의 오목한 틈(소매와 몸판 사이 골)의 가장 깊은 곳,
  // (나) 행 단위 옆선 추적. 둘 다 끝까지 분석해 비율 검증 신뢰도가 높은 쪽을 쓴다.
  const pockets = hullPockets(mask, w, h, top, bottom);
  // 입구보다 훨씬 깊은 좁은 틈 = 몸판 옆에 늘어뜨린 소매 아래의 틈(꼭대기가 실제 겨드랑이보다 낮다)
  let slit = false;
  const hullArmpit = (side: 1 | -1): Vec2 | null => {
    let best: Pocket | null = null;
    for (const p of pockets) {
      const dx = (p.deep.x - cx) * side;
      if (dx < W * 0.08) continue; // 목 파임·가운데 틈 제외
      if (p.deep.y < top + H * 0.1 || p.deep.y > top + H * 0.8) continue;
      if (p.openingY < p.deep.y - H * 0.02) continue; // 입구가 위에 있는 틈(목 쪽) 제외
      if (p.depth < W * 0.03) continue;
      if (!best || p.depth > best.depth) best = p;
    }
    if (best && best.depth > best.openingLen * 1.6) slit = true;
    return best ? { x: best.deep.x, y: best.deep.y } : null;
  };
  const pairs: [Vec2, Vec2][] = [];
  const hl = hullArmpit(1);
  const hr = hullArmpit(-1);
  const sl = findArmpit(1);
  const sr = findArmpit(-1);
  if (hl && hr) pairs.push([hl, hr]);
  if (sl && sr) pairs.push([sl, sr]);
  if (hl && sr && !hr) pairs.push([hl, sr]);
  if (sl && hr && !hl) pairs.push([sl, hr]);
  let bestResult: TopAnalysis | null = null;
  const consider = (r: TopAnalysis): void => {
    if (!bestResult || r.confidence > bestResult.confidence + 0.02) bestResult = r;
  };
  // 소매를 몸판 옆에 접어 붙인 사진(재킷·셔츠·니트 상품 컷에 흔함): 윤곽에 틈이 없으므로
  // 옷 안쪽의 세로 접힘선(소매 안쪽 가장자리의 그늘)을 몸판 옆선으로 보고, 겨드랑이 높이는 비율로 추정한다.
  const fold = rgba ? foldLines(mask, rgba, w, cx, top, H, left, right) : null;
  if (fold) {
    const chest = fold.xL - fold.xR;
    const topAt = (x: number): number => {
      const xi = Math.round(x);
      for (let y = top; y <= bottom; y++) if (mask[y * w + xi]) return y;
      return top;
    };
    const shoulderY = (topAt(fold.xL - chest * 0.04) + topAt(fold.xR + chest * 0.04)) / 2;
    const y = shoulderY + chest * 0.45;
    // 어깨점: 윤곽의 바깥 위 모서리(접힌 소매가 시작되는 곳). 소매 곡면의 윗 모서리가 바깥 윤곽을 따라 내려가게 한다.
    const corner = (side: 1 | -1): Vec2 => {
      let best = { x: cx, y: top, v: -Infinity };
      for (let yy = top; yy <= top + H * 0.5; yy++) {
        const rs = runs[yy];
        if (!rs.length) continue;
        const xx = side === 1 ? rs[rs.length - 1].x1 : rs[0].x0;
        const v = (xx - cx) * side - (yy - top) * 0.8;
        if (v > best.v) best = { x: xx, y: yy, v };
      }
      return { x: best.x, y: best.y };
    };
    const r = finish({ x: fold.xL, y }, { x: fold.xR, y }, [...warnings, '소매가 몸판 옆에 접혀 있어 접힘선으로 몸판을 나눴습니다'], {
      L: { x: fold.xL, y: hemY },
      R: { x: fold.xR, y: hemY },
    }, { L: corner(1), R: corner(-1) });
    r.confidence *= 0.92;
    consider(r);
  }
  if (!pairs.length) return bestResult;
  for (const [aL, aR] of pairs) {
    const r = finish(aL, aR, [...warnings]);
    consider(r);
    // 늘어뜨린 소매가 몸판에 겹쳐 틈이 아래쪽에만 보이면, 틈 꼭대기는 실제 겨드랑이보다 낮다.
    // 틈 꼭대기의 x(몸판 옆선)는 믿고, 높이는 흔한 옷 비율(진동 깊이 ≈ 가슴 폭의 0.45)로 추정한다.
    const chest = aL.x - aR.x;
    const topAt = (x: number): number => {
      const xi = Math.round(x);
      for (let y = top; y <= bottom; y++) if (mask[y * w + xi]) return y;
      return top;
    };
    const shoulderY = (topAt(aL.x - chest * 0.04) + topAt(aR.x + chest * 0.04)) / 2;
    if (chest > 0 && (Math.min(aL.y, aR.y) - shoulderY) / chest > 0.7) {
      const y = shoulderY + chest * 0.45;
      const est = finish({ x: aL.x, y }, { x: aR.x, y }, [...warnings, '소매가 몸판에 겹쳐 겨드랑이를 비율로 추정했습니다']);
      // 좁은 틈이면 추정이 더 맞고, 아니면(어깨가 떨어진 오버핏 등) 찾은 위치가 더 맞다.
      if (slit) {
        if (est.confidence >= bestResult!.confidence - 0.15) bestResult = est;
      } else {
        est.confidence *= 0.85;
        consider(est);
      }
    }
  }
  return bestResult;

  function finish(armpitL: Vec2, armpitR: Vec2, warnings: string[], hemO?: { L: Vec2; R: Vec2 }, shoulderO?: { L: Vec2; R: Vec2 }): TopAnalysis {
  const hemL = hemO?.L ?? hemL0;
  const hemR = hemO?.R ?? hemR0;
  const hemHalf = (hemL.x - hemR.x) / 2;

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
  // 후드: 모자가 어깨선보다 높이 솟아 있고, 모자와 어깨가 만나는 곳에 위가 열린 오목한 골이 있다.
  // 그 골의 가장 깊은 점이 목둘레 양 끝이며, 그 위쪽(모자)은 목 뒤로 넘어가는 부분(라벨 4)으로 둔다.
  const hoodNotch = (side: 1 | -1): Vec2 | null => {
    let best: Pocket | null = null;
    for (const p of pockets) {
      const dx = (p.deep.x - cx) * side;
      if (dx < torsoHalf * 0.1 || dx > torsoHalf * 0.75) continue;
      if (p.openingY > p.deep.y) continue; // 입구가 위쪽
      if (p.deep.y > Math.min(armpitL.y, armpitR.y)) continue;
      if (p.depth < H * 0.025) continue;
      if (!best || p.depth > best.depth) best = p;
    }
    return best ? { ...best.deep } : null;
  };
  const notchL = hoodNotch(1);
  const notchR = hoodNotch(-1);
  const centerTop = ytop[Math.round(cx)];
  const hood =
    !!notchL && !!notchR && Number.isFinite(centerTop) && Math.min(notchL.y, notchR.y) - centerTop > H * 0.08 && notchL.x - notchR.x > torsoHalf * 0.3;
  if (hood) warnings.push('후드: 모자 부분은 목 뒤로 넘깁니다');
  const neckL = hood ? notchL! : highest(cx + torsoHalf * 0.05, cx + torsoHalf * 0.45);
  const neckR = hood ? notchR! : highest(cx - torsoHalf * 0.45, cx - torsoHalf * 0.05);
  const neckW = Math.max(4, neckL.x - neckR.x);
  const neckY = (neckL.y + neckR.y) / 2;
  const dipY = ytop[Math.round(cx)];
  let neckFront: Vec2;
  let hasOpenNeck = false;
  if (!hood && Number.isFinite(dipY) && dipY - neckY > neckW * 0.12) {
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
  const shoulderL = shoulderO?.L ?? shoulderAt(armpitL, 1);
  const shoulderR = shoulderO?.R ?? shoulderAt(armpitR, -1);

  // 부위 라벨
  const labels = new Uint8Array(w * h);
  // 진동 둘레선(어깨 → 겨드랑이) 바깥쪽이 소매
  const armholeX = (s: Vec2, a: Vec2, y: number): number => {
    if (a.y === s.y) return a.x;
    const t = Math.min(1, Math.max(0, (y - s.y) / (a.y - s.y)));
    return s.x + (a.x - s.x) * t;
  };
  // 겨드랑이 아래 몸판 옆선(겨드랑이 → 밑단 끝). 이 선 바깥에 붙어 있는 것은 몸판에 겹친 소매다.
  const sideX = (a: Vec2, hm: Vec2, y: number): number => (hm.y === a.y ? a.x : a.x + ((hm.x - a.x) * (y - a.y)) / (hm.y - a.y));
  const sideTol = W * 0.03;
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
        else if (y > armpitL.y && x > cx && (r !== torso || x > sideX(armpitL, hemL, y) + sideTol)) lab = 2;
        if (y <= armpitR.y && x < armholeX(shoulderR, armpitR, y)) lab = 3;
        else if (y > armpitR.y && x < cx && (r !== torso || x < sideX(armpitR, hemR, y) - sideTol)) lab = 3;
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

  // 후드 모자: 목둘레 곡선과 목 → 어깨 선보다 위에 있는 몸판 픽셀
  if (hood) {
    const upper = (x: number): number => {
      if (x >= neckR.x && x <= neckL.x) {
        const t = (x - cx) / (neckW / 2);
        return neckFront.y - (neckFront.y - neckY) * t * t;
      }
      const [n, sh] = x > neckL.x ? [neckL, shoulderL] : [neckR, shoulderR];
      const t = sh.x === n.x ? 1 : Math.min(1, Math.max(0, (x - n.x) / (sh.x - n.x)));
      return n.y + (sh.y - n.y) * t - 2;
    };
    for (let y = top; y <= bottom; y++) {
      for (let x = left; x <= right; x++) {
        const i = y * w + x;
        if (labels[i] === 1 && y < upper(x)) labels[i] = 4;
      }
    }
  }
  // 목 안쪽: 목둘레 양 끝과 앞 목선을 지나는 포물선 위쪽
  if (!hasOpenNeck && !hood) {
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

  // 신뢰도: 좌우 대칭성 + 실제 옷으로 가능한 비율인지(어깨·가슴·밑단·목·기장·진동 깊이)
  const symArm = 1 - Math.min(1, Math.abs(armpitL.y - armpitR.y) / (H * 0.1));
  const symCenter = 1 - Math.min(1, Math.abs((armpitL.x + armpitR.x) / 2 - cx) / (W * 0.08));
  const chest = Math.max(1, armpitL.x - armpitR.x);
  const shoulderW = shoulderL.x - shoulderR.x;
  const armpitDrop = (armpitL.y + armpitR.y) / 2 - (shoulderL.y + shoulderR.y) / 2;
  const checks: [string, number, number, number][] = [
    // 접힌 소매 사진은 어깨점이 윤곽 바깥 모서리라 넓다
    ['어깨/가슴', shoulderW / chest, 0.65, shoulderO ? 1.8 : 1.35],
    ['밑단/가슴', (hemHalf * 2) / chest, 0.6, 1.7],
    ['목/어깨', neckW / Math.max(1, shoulderW), shoulderO ? 0.06 : 0.18, 0.75],
    ['기장/가슴', (hemY - (shoulderL.y + shoulderR.y) / 2) / chest, 0.7, 2.4],
    ['진동 깊이/가슴', armpitDrop / chest, 0.18, 1.1],
  ];
  let plaus = 1;
  for (const [name, value, lo, hi] of checks) {
    if (value < lo || value > hi) {
      plaus *= 0.55;
      warnings.push(`비율 이상(${name} ${value.toFixed(2)})`);
    }
  }
  const confidence = Math.max(0, Math.min(1, (symArm * 0.5 + symCenter * 0.5) * plaus));
  if (confidence < 0.6) warnings.push('자동 분석 신뢰도가 낮습니다. 기준점을 확인해 주세요');

  for (let i = 0; i < labels.length; i++) if (!mask0[i]) labels[i] = 0;
  return { keypoints, labels, sleeve, confidence, warnings };
  }
}

/** 가운데 세로선 양쪽의 구간 사이 좁은 틈(열린 앞판)을 메운다. 틈이 없으면 원래 마스크를 그대로 돌려준다. */
function closeCenterGap(mask: Uint8Array, w: number, h: number): Uint8Array {
  let left = w;
  let right = -1;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    const x = i % w;
    if (x < left) left = x;
    if (x > right) right = x;
  }
  if (right < 0) return mask;
  const mid = (left + right) / 2;
  const maxGap = (right - left) * 0.14;
  let out: Uint8Array | null = null;
  let filledRows = 0;
  for (let y = 0; y < h; y++) {
    const runs = rowRuns(mask, w, y);
    for (let k = 0; k + 1 < runs.length; k++) {
      const a = runs[k];
      const b = runs[k + 1];
      const g0 = a.x1 + 1;
      const g1 = b.x0 - 1;
      if (g1 - g0 + 1 > maxGap) continue;
      // 틈이 가운데 근처에 있고, 양쪽 구간이 충분히 넓을 때만(소매-몸판 틈과 구분)
      if (Math.abs((g0 + g1) / 2 - mid) > (right - left) * 0.08) continue;
      if (a.x1 - a.x0 < (right - left) * 0.12 || b.x1 - b.x0 < (right - left) * 0.12) continue;
      out ??= mask.slice();
      out.fill(1, y * w + g0, y * w + g1 + 1);
      filledRows++;
    }
  }
  return out && filledRows > 3 ? out : mask;
}

interface Pocket {
  /** 틈 안에서 입구(껍질 변)로부터 가장 먼 점 */
  deep: Vec2;
  depth: number;
  /** 입구 변 중점의 y */
  openingY: number;
  /** 입구 변 길이 */
  openingLen: number;
}

/**
 * 볼록 껍질 안쪽이지만 옷이 아닌 영역(오목한 틈)을 찾고, 각 틈의 입구 변과 가장 깊은 점을 구한다.
 * 티셔츠의 소매 밑, 늘어뜨린 긴소매와 몸판 사이의 좁은 틈에서 가장 깊은 곳이 겨드랑이다.
 */
export function hullPockets(mask: Uint8Array, w: number, h: number, top: number, bottom: number): Pocket[] {
  const pts: Vec2[] = [];
  const spanL = new Int32Array(h).fill(-1);
  const spanR = new Int32Array(h).fill(-1);
  for (let y = top; y <= bottom; y++) {
    let a = -1;
    let b = -1;
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) {
        if (a < 0) a = x;
        b = x;
      }
    }
    if (a >= 0) {
      pts.push({ x: a, y }, { x: b, y });
    }
  }
  if (pts.length < 6) return [];
  // 단조 사슬로 볼록 껍질
  pts.sort((p, q) => p.x - q.x || p.y - q.y);
  const cross = (o: Vec2, a: Vec2, b: Vec2): number => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower: Vec2[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: Vec2[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  if (hull.length < 3) return [];
  // 각 행의 껍질 안쪽 범위
  for (let y = top; y <= bottom; y++) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < hull.length; k++) {
      const p = hull[k];
      const q = hull[(k + 1) % hull.length];
      if ((p.y - y) * (q.y - y) > 0) continue;
      if (p.y === q.y) {
        lo = Math.min(lo, p.x, q.x);
        hi = Math.max(hi, p.x, q.x);
        continue;
      }
      const x = p.x + ((y - p.y) * (q.x - p.x)) / (q.y - p.y);
      lo = Math.min(lo, x);
      hi = Math.max(hi, x);
    }
    if (lo <= hi) {
      spanL[y] = Math.ceil(lo);
      spanR[y] = Math.floor(hi);
    }
  }
  // 틈 픽셀 연결 요소
  const comp = new Int32Array(w * h);
  const queue = new Int32Array(w * h);
  const inPocket = (i: number): boolean => {
    const y = (i / w) | 0;
    const x = i - y * w;
    return y >= top && y <= bottom && spanL[y] >= 0 && x >= spanL[y] && x <= spanR[y] && !mask[i];
  };
  const members: number[][] = [];
  for (let y = top; y <= bottom; y++) {
    if (spanL[y] < 0) continue;
    for (let x = spanL[y]; x <= spanR[y]; x++) {
      const s = y * w + x;
      if (mask[s] || comp[s]) continue;
      const id = members.length + 1;
      const list: number[] = [];
      let head = 0;
      let tail = 0;
      queue[tail++] = s;
      comp[s] = id;
      while (head < tail) {
        const i = queue[head++];
        list.push(i);
        const xx = i % w;
        const nb = [xx > 0 ? i - 1 : -1, xx < w - 1 ? i + 1 : -1, i >= w ? i - w : -1, i < w * (h - 1) ? i + w : -1];
        for (const j of nb) {
          if (j >= 0 && !comp[j] && inPocket(j)) {
            comp[j] = id;
            queue[tail++] = j;
          }
        }
      }
      members.push(list);
    }
  }
  const area = pts.length; // 대략적인 규모 기준(행 수 × 2)
  const out: Pocket[] = [];
  members.forEach((list, idx) => {
    if (list.length < Math.max(12, area * 0.05)) return;
    const id = idx + 1;
    // 입구 변: 껍질 변을 따라가며 이 틈과 맞닿은 표본이 가장 많은 변
    let bestEdge = -1;
    let bestCount = 0;
    for (let k = 0; k < hull.length; k++) {
      const p = hull[k];
      const q = hull[(k + 1) % hull.length];
      const len = Math.hypot(q.x - p.x, q.y - p.y);
      let count = 0;
      for (let t = 0; t <= len; t += 1) {
        const x = p.x + ((q.x - p.x) * t) / (len || 1);
        const y = p.y + ((q.y - p.y) * t) / (len || 1);
        let hit = false;
        for (let dy = -1; dy <= 1 && !hit; dy++) {
          for (let dx = -1; dx <= 1 && !hit; dx++) {
            const xx = Math.round(x) + dx;
            const yy = Math.round(y) + dy;
            if (xx >= 0 && xx < w && yy >= 0 && yy < h && comp[yy * w + xx] === id) hit = true;
          }
        }
        if (hit) count++;
      }
      if (count > bestCount) {
        bestCount = count;
        bestEdge = k;
      }
    }
    if (bestEdge < 0) return;
    const p = hull[bestEdge];
    const q = hull[(bestEdge + 1) % hull.length];
    const len = Math.hypot(q.x - p.x, q.y - p.y) || 1;
    let deep = { x: 0, y: 0 };
    let depth = -1;
    for (const i of list) {
      const x = i % w;
      const y = (i / w) | 0;
      const d = Math.abs((q.x - p.x) * (p.y - y) - (p.x - x) * (q.y - p.y)) / len;
      if (d > depth) {
        depth = d;
        deep = { x, y };
      }
    }
    out.push({ deep, depth, openingY: (p.y + q.y) / 2, openingLen: len });
  });
  return out;
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

/**
 * 옷 안쪽의 세로 접힘선: 몸판 가운데 양쪽에서, 옷 안쪽(가장자리 제외) 픽셀의 가로 밝기 변화가
 * 여러 줄에 걸쳐 같은 x에 모이는 곳. 소매가 몸판 위·옆에 겹친 가장자리가 여기에 해당한다.
 */
function foldLines(mask: Uint8Array, rgba: Uint8ClampedArray, w: number, cx: number, top: number, H: number, left: number, right: number): { xL: number; xR: number } | null {
  const W = right - left + 1;
  const gray = (i: number): number => rgba[i * 4] * 0.299 + rgba[i * 4 + 1] * 0.587 + rgba[i * 4 + 2] * 0.114;
  const score = new Float32Array(w);
  const count = new Float32Array(w);
  const inner = 4;
  for (let y = Math.round(top + H * 0.4); y < Math.round(top + H * 0.88); y++) {
    for (let x = left + inner; x <= right - inner; x++) {
      const i = y * w + x;
      if (!mask[i] || !mask[i - inner] || !mask[i + inner] || !mask[i - w * inner] || !mask[i + w * inner]) continue;
      // 3줄 평균 가로 기울기
      let g = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const j = i + dy * w;
        g += gray(j + 2) + gray(j + 1) - gray(j - 1) - gray(j - 2);
      }
      score[x] += Math.abs(g) / 6;
      count[x]++;
    }
  }
  const prof = new Float32Array(w);
  for (let x = 0; x < w; x++) prof[x] = count[x] > H * 0.2 ? score[x] / count[x] : 0;
  // 3px 이동 평균
  const sm = new Float32Array(w);
  for (let x = 1; x < w - 1; x++) sm[x] = (prof[x - 1] + prof[x] + prof[x + 1]) / 3;
  const vals = Array.from(sm.slice(left, right + 1)).filter((v) => v > 0).sort((a, b) => a - b);
  const med = vals[vals.length >> 1] ?? 0;
  const peak = (a: number, b: number): number | null => {
    let best = -1;
    let bx = -1;
    for (let x = Math.round(Math.min(a, b)); x <= Math.round(Math.max(a, b)); x++) {
      if (x < 1 || x >= w - 1) continue;
      if (sm[x] > best) {
        best = sm[x];
        bx = x;
      }
    }
    return bx >= 0 && best > Math.max(2.5, med * 2.2) ? bx : null;
  };
  const xL = peak(cx + W * 0.16, cx + W * 0.42);
  const xR = peak(cx - W * 0.42, cx - W * 0.16);
  if (xL === null || xR === null) return null;
  // 좌우 대칭이어야 한다
  if (Math.abs(xL - cx - (cx - xR)) > W * 0.08) return null;
  return { xL, xR };
}
