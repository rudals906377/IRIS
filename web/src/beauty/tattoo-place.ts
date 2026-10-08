// 타투 위치 계산(순수 계산, 단위 테스트 대상): 몸 관절점 → 도안을 붙일 격자(화면 좌표 + 도안 좌표 + 보임 정도).
// 팔은 원기둥으로 보고 도안을 둘레 방향으로 감는다: 둘레 각 θ의 점은 화면에서 축으로부터 R·sinθ 떨어지고,
// 옆으로 돌아갈수록(cosθ↓) 흐려지다 뒤쪽은 보이지 않는다.

import { add, dist, lerp, norm, perp, scale, sub, type Vec2 } from '../engine/math.ts';

export type TattooPlace = 'forearmL' | 'forearmR' | 'upperArmL' | 'upperArmR' | 'neckL' | 'neckR' | 'chest';

/** L/R은 사람 기준(정면 영상에서 사람 왼쪽 = 이미지 오른쪽) */
export const PLACE_LABELS: Record<TattooPlace, string> = {
  forearmL: '왼팔 아래',
  forearmR: '오른팔 아래',
  upperArmL: '왼팔 위',
  upperArmR: '오른팔 위',
  neckL: '목 왼쪽',
  neckR: '목 오른쪽',
  chest: '쇄골 아래',
};

/** MediaPipe Pose 번호. 100번대는 얼굴 점에서 가져와 넣는 보조점(턱 끝, 턱선 양쪽) */
export const POSE = {
  nose: 0, earL: 7, earR: 8, shoulderL: 11, shoulderR: 12, elbowL: 13, elbowR: 14, wristL: 15, wristR: 16,
  chin: 100, jawL: 101, jawR: 102,
} as const;
/** 보조점에 쓰는 얼굴 점 번호(MediaPipe 얼굴 메쉬): 턱 끝, 사람 왼쪽 턱선, 오른쪽 턱선 */
export const FACE_FOR_POSE: Record<number, number> = { 100: 152, 101: 397, 102: 172 };

export interface PosePoints {
  /** 픽셀 좌표(필터 거친 값). 번호 → 점 */
  p: Record<number, Vec2>;
  /** 번호 → 보임 정도(0~1) */
  vis: Record<number, number>;
}

export interface TattooMesh {
  /** 정점마다 x, y(화면 px), u, v(도안), 보임(0~1) */
  data: Float32Array;
  /** 삼각형 정점 번호 */
  index: Uint16Array;
  /** 전체 보임 정도(관절이 잘 안 보이면 줄어듦) */
  confidence: number;
}

export interface PlaceParams {
  /** 0~1: 부위 길이 대비 도안 크기 */
  size: number;
  /** 도안 가로/세로 */
  aspect: number;
  /** 부위 굵기(px, 지름). 모르면 null → 길이 비율로 추정 */
  width: number | null;
  /** 부위 축을 따라 도안 가운데 위치(0 = 몸 쪽 끝, 1 = 먼 쪽 끝). 기본 0.5 */
  along?: number;
  /** 축에서 옆으로 벗어난 정도(-1~1, 부위 반폭 기준. 팔은 둘레 방향으로 돌아간다). 기본 0 */
  across?: number;
}

/** 부위의 축(시작 = 몸 쪽, 끝 = 먼 쪽)과 기준 굵기 비율. 도안의 위쪽은 몸 쪽을 향한다. */
export function placeAxis(place: TattooPlace, pose: PosePoints): { a: Vec2; b: Vec2; widthRatio: number; cylinder: boolean; conf: number } | null {
  const P = pose.p;
  const V = (...ids: number[]): number => Math.min(...ids.map((i) => pose.vis[i] ?? 0));
  const has = (...ids: number[]): boolean => ids.every((i) => P[i] !== undefined);
  switch (place) {
    case 'forearmL':
    case 'forearmR': {
      const [e, w] = place === 'forearmL' ? [POSE.elbowL, POSE.wristL] : [POSE.elbowR, POSE.wristR];
      if (!has(e, w)) return null;
      return { a: P[e], b: P[w], widthRatio: 0.3, cylinder: true, conf: V(e, w) };
    }
    case 'upperArmL':
    case 'upperArmR': {
      const [s, e] = place === 'upperArmL' ? [POSE.shoulderL, POSE.elbowL] : [POSE.shoulderR, POSE.elbowR];
      if (!has(s, e)) return null;
      // 어깨점은 관절 중심이라 도안이 어깨 위로 올라가지 않게 조금 아래에서 시작
      return { a: lerp(P[s], P[e], 0.12), b: P[e], widthRatio: 0.4, cylinder: true, conf: V(s, e) };
    }
    case 'neckL':
    case 'neckR': {
      const jaw = place === 'neckL' ? POSE.jawL : POSE.jawR;
      const other = place === 'neckL' ? POSE.jawR : POSE.jawL;
      if (has(POSE.chin, jaw, other)) {
        // 얼굴이 보이면 턱선 기준: 목 옆은 턱 끝과 턱선 사이, 턱 아래에서 시작
        const jw = dist(P[jaw], P[other]);
        let down = perp(norm(sub(P[POSE.jawR], P[POSE.jawL])));
        if (down.y < 0) down = scale(down, -1);
        const top = add(lerp(P[POSE.chin], P[jaw], 0.45), scale(down, jw * 0.1));
        return { a: top, b: add(top, scale(down, jw * 0.45)), widthRatio: 0.55, cylinder: false, conf: 1 };
      }
      const [ear, sh] = place === 'neckL' ? [POSE.earL, POSE.shoulderL] : [POSE.earR, POSE.shoulderR];
      const oth = place === 'neckL' ? POSE.shoulderR : POSE.shoulderL;
      if (!has(ear, sh, oth)) return null;
      // 얼굴 점이 없으면: 귀 아래에서 어깨 중심 쪽으로
      const neckBase = lerp(P[sh], lerp(P[sh], P[oth], 0.5), 0.8);
      const top = lerp(P[ear], neckBase, 0.35);
      return { a: top, b: neckBase, widthRatio: 0.55, cylinder: false, conf: V(ear, sh) };
    }
    case 'chest': {
      if (!has(POSE.shoulderL, POSE.shoulderR)) return null;
      const l = P[POSE.shoulderL];
      const r = P[POSE.shoulderR];
      const sw = dist(l, r);
      const m = lerp(l, r, 0.5);
      // 어깨선에 수직인 아래 방향
      let down = perp(norm(sub(l, r)));
      if (down.y < 0) down = scale(down, -1);
      // 쇄골 바로 아래(어깨선에서 어깨 폭의 5~30%)
      const a = add(m, scale(down, sw * 0.05));
      return { a, b: add(a, scale(down, sw * 0.25)), widthRatio: 1.6, cylinder: false, conf: V(POSE.shoulderL, POSE.shoulderR) };
    }
  }
}

const COLS = 12;
const ROWS = 6;

/** 부위 축과 도안 크기로 격자를 만든다. */
export function tattooMesh(place: TattooPlace, pose: PosePoints, prm: PlaceParams): TattooMesh | null {
  const ax = placeAxis(place, pose);
  if (!ax) return null;
  const segLen = dist(ax.a, ax.b);
  if (segLen < 8) return null;
  const d = norm(sub(ax.b, ax.a));
  const n = perp(d);
  const limbW = prm.width ?? segLen * ax.widthRatio;
  // 가로로 긴 도안(레터링 등)을 팔에 넣을 때는 팔을 따라 세로로 돌려 넣는다(실제 팔 레터링처럼)
  const rotate = ax.cylinder && prm.aspect > 1.3;
  const aspect = rotate ? 1 / prm.aspect : prm.aspect;
  // 도안 세로 길이: 부위 길이의 (0.25 ~ 0.85). 가로는 도안 비율
  let h = segLen * (0.25 + 0.6 * prm.size);
  let w = h * aspect;
  const R = limbW / 2;
  // 원기둥은 둘레의 절반(π·R), 평면(목·가슴)은 부위 폭을 넘지 않게: 넘치면 전체를 줄인다
  const maxW = ax.cylinder ? Math.PI * R * 0.9 : limbW;
  if (w > maxW) {
    h *= maxW / w;
    w = maxW;
  }
  const along = Math.min(0.92, Math.max(0.08, prm.along ?? 0.5));
  const across = Math.min(0.95, Math.max(-0.95, prm.across ?? 0));
  const center = lerp(ax.a, ax.b, along);
  // 팔: 둘레 각도로 돌린다(정면에서 벗어날수록 옆으로 감기며 흐려짐). 평면: 옆으로 민다
  const th0 = ax.cylinder ? Math.asin(across) : 0;
  const off0 = ax.cylinder ? 0 : across * (limbW / 2);
  const data = new Float32Array((COLS + 1) * (ROWS + 1) * 5);
  let k = 0;
  for (let j = 0; j <= ROWS; j++) {
    const v = j / ROWS;
    const along = (v - 0.5) * h;
    for (let i = 0; i <= COLS; i++) {
      const u = i / COLS;
      const arc = (u - 0.5) * w;
      let off: number;
      let vis = 1;
      if (ax.cylinder) {
        const th = arc / R + th0;
        off = R * Math.sin(th);
        vis = Math.max(0, Math.cos(th));
      } else {
        off = arc + off0;
      }
      // 도안 오른쪽(u=1)이 화면에서 축의 어느 쪽으로 가는지: 도안 위(v=0)가 몸 쪽(a)이므로
      // 아래 방향 d 기준으로 오른쪽 = -perp(d)
      const p = add(center, add(scale(d, along), scale(n, -off)));
      data[k++] = p.x;
      data[k++] = p.y;
      // 돌려 넣을 때: 도안의 가로(글 방향)가 팔 축을 따라 몸 쪽 → 먼 쪽으로
      data[k++] = rotate ? v : u;
      data[k++] = rotate ? 1 - u : v;
      data[k++] = vis;
    }
  }
  const index = new Uint16Array(COLS * ROWS * 6);
  let q = 0;
  for (let j = 0; j < ROWS; j++) {
    for (let i = 0; i < COLS; i++) {
      const a = j * (COLS + 1) + i;
      const b = a + 1;
      const c = a + COLS + 1;
      const e = c + 1;
      index.set([a, b, c, b, e, c], q);
      q += 6;
    }
  }
  return { data, index, confidence: ax.conf };
}

/**
 * 분할(몸 피부 확률, RGBA8의 G)에서 부위 굵기를 잰다: 축 가운데에서 수직 방향으로 피부가 끊기는 곳까지.
 * 반환: 지름(px, 영상 좌표) 또는 null
 */
export function measureLimbWidth(
  flags: Uint8ClampedArray,
  segW: number,
  segH: number,
  videoW: number,
  videoH: number,
  a: Vec2,
  b: Vec2,
): number | null {
  const sx = segW / videoW;
  const sy = segH / videoH;
  const d = norm(sub(b, a));
  const n = perp(d);
  const skinAt = (p: Vec2): boolean => {
    const x = Math.round(p.x * sx);
    const y = Math.round(p.y * sy);
    if (x < 0 || y < 0 || x >= segW || y >= segH) return false;
    return flags[(y * segW + x) * 4 + 1] > 128;
  };
  const segLen = dist(a, b);
  const maxR = segLen * 0.6;
  const step = Math.max(1, 1 / Math.min(sx, sy)) * 0.75;
  const widths: number[] = [];
  for (const t of [0.35, 0.5, 0.65]) {
    const c = lerp(a, b, t);
    if (!skinAt(c)) continue;
    let r1 = 0;
    while (r1 < maxR && skinAt(add(c, scale(n, r1 + step)))) r1 += step;
    let r2 = 0;
    while (r2 < maxR && skinAt(add(c, scale(n, -(r2 + step))))) r2 += step;
    if (r1 < maxR && r2 < maxR) widths.push(r1 + r2 + step);
  }
  if (widths.length === 0) return null;
  widths.sort((x, y) => x - y);
  return widths[Math.floor(widths.length / 2)];
}

export const ALL_PLACES: TattooPlace[] = ['forearmL', 'forearmR', 'upperArmL', 'upperArmR', 'neckL', 'neckR', 'chest'];

/**
 * 화면에서 누른 점(px) → 가장 가까운 부위와 그 안의 위치. 부위 굵기 안쪽(여유 30%)을 누른 경우만.
 * widthOf: 부위별 실제 굵기(px, 모르면 null)
 */
export function placeFromPoint(pose: PosePoints, pt: Vec2, widthOf: (place: TattooPlace) => number | null = () => null): { place: TattooPlace; along: number; across: number } | null {
  let best: { place: TattooPlace; along: number; across: number; score: number } | null = null;
  for (const place of ALL_PLACES) {
    const ax = placeAxis(place, pose);
    if (!ax) continue;
    const len = dist(ax.a, ax.b);
    if (len < 8) continue;
    const d = norm(sub(ax.b, ax.a));
    const n = perp(d);
    const rel = sub(pt, ax.a);
    const t = (rel.x * d.x + rel.y * d.y) / len;
    const sgn = rel.x * n.x + rel.y * n.y;
    const halfW = (widthOf(place) ?? len * ax.widthRatio) / 2;
    // 축 밖(앞뒤)으로 조금 벗어난 것도 받아 준다
    if (t < -0.15 || t > 1.15) continue;
    const across = -sgn / halfW;
    if (Math.abs(across) > 1.3) continue;
    // 점수: 축에서 떨어진 정도(반폭 기준) + 축 끝 밖으로 나간 정도
    const score = Math.abs(across) + Math.max(0, -t, t - 1) * 3;
    if (!best || score < best.score) best = { place, along: Math.min(0.92, Math.max(0.08, t)), across: Math.min(0.95, Math.max(-0.95, across)), score };
  }
  return best && { place: best.place, along: best.along, across: best.across };
}
