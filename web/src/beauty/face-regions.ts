// 얼굴 점(MediaPipe Face Landmarker 478점) → 메이크업 영역 다각형.
// 좌표는 영상 픽셀. L/R은 **사람 기준**(정면 영상에서 사람 왼쪽 = 이미지 오른쪽).
// 점 번호는 MediaPipe 얼굴 메쉬 표준 번호다.

import type { Vec2 } from '../engine/math.ts';

/** 입술 바깥 윤곽(시계 방향) */
export const LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
/** 입술 안쪽(입 벌린 곳) 윤곽 */
export const LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];

/** 사람 오른쪽 눈(이미지 왼쪽): 윗눈꺼풀(바깥 → 안쪽), 아랫눈꺼풀(안쪽 → 바깥) */
export const EYE_R_UPPER = [33, 246, 161, 160, 159, 158, 157, 173, 133];
export const EYE_R_LOWER = [133, 155, 154, 153, 145, 144, 163, 7, 33];
/** 사람 왼쪽 눈(이미지 오른쪽) */
export const EYE_L_UPPER = [263, 466, 388, 387, 386, 385, 384, 398, 362];
export const EYE_L_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263];

/** 눈썹 아랫선(바깥 → 안쪽)과 윗선(안쪽 → 바깥) */
export const BROW_R_LOWER = [46, 53, 52, 65, 55];
export const BROW_R_UPPER = [107, 66, 105, 63, 70];
export const BROW_L_LOWER = [276, 283, 282, 295, 285];
export const BROW_L_UPPER = [336, 296, 334, 293, 300];

/** 얼굴 폭 기준점(양쪽 광대 옆) */
export const FACE_SIDE_R = 234;
export const FACE_SIDE_L = 454;
/** 볼 중심(광대 아래) */
export const CHEEK_R = 205;
export const CHEEK_L = 425;
/** 광대 바깥쪽(블러셔가 관자놀이 쪽으로 퍼지는 방향) */
export const CHEEKBONE_R = 116;
export const CHEEKBONE_L = 345;
/** 코끝 */
export const NOSE_TIP = 1;
/** 조명 추정용 볼 피부 표본(광대 아래·광대 바깥, 양쪽) */
export const SKIN_SAMPLES = [CHEEK_R, CHEEK_L, CHEEKBONE_R, CHEEKBONE_L];
/** 입술 살 표본: LIPS_OUTER·LIPS_INNER에서 같은 번호끼리 짝을 이룬다(아랫입술 3곳, 윗입술 3곳) */
const LIP_BODY = [3, 5, 7, 13, 15, 17];

/** 값이 있는 삼각형 망의 정점. v: 진하기(0~1), s·t: 부위 안 좌표(펄 반짝이 위치 고정용, 얼굴 폭 단위) */
export interface MVert {
  x: number;
  y: number;
  v: number;
  s: number;
  t: number;
}

/** 립 모양: 풀 립(윤곽 가득), 그라데이션(안쪽 진하고 바깥으로 옅게), 블러(가운데 진하고 경계를 흐리게) */
export type LipStyle = 'full' | 'gradient' | 'blur';

export interface FaceRegions {
  /** 얼굴 폭(px) — 경계 흐림 크기 등의 기준 */
  faceW: number;
  /** 얼굴 좌우 돌림(-1 사람 오른쪽이 멀어짐 ~ 1 사람 왼쪽이 멀어짐, 0 정면)과 위아래 끄덕임(대략값) */
  yaw: number;
  pitch: number;
  /** 사람 오른쪽/왼쪽 절반이 보이는 정도(0~1). 얼굴을 돌려 먼 쪽이 코 뒤로 숨을수록 0 */
  visR: number;
  visL: number;
  lipsOuter: Vec2[];
  lipsInner: Vec2[];
  /** 립: 안쪽 윤곽 → 바깥 윤곽 띠(입 벌린 안쪽은 비어 있음). 진하기는 립 모양에 따라 */
  lip: MVert[];
  /** 입술 살 가운데(바깥·안쪽 윤곽의 중간) — 입술 평균 밝기 표본 위치 */
  lipBody: Vec2[];
  /** 볼 피부 표본 위치 — 조명(밝기·색온도) 추정용 */
  skinPts: Vec2[];
  /** 아이섀도: 속눈썹 쪽 1 → 위로 갈수록 0 */
  shadow: MVert[];
  /** 아이라이너: 눈꼬리로 갈수록 두꺼워지고 날개로 이어지는 띠 */
  liner: MVert[];
  /** 눈을 뜬 부분(흰자·눈동자): 아이섀도가 번져 들어가지 않게 빼는 영역 */
  eyeR: Vec2[];
  eyeL: Vec2[];
  /** 눈썹: 앞머리는 옅게, 꼬리는 가늘게 모이는 띠 */
  brow: MVert[];
  /** 블러셔: 가운데 1 → 가장자리 0인 타원 부채꼴 */
  blush: MVert[];
  /** 쉐딩(광대 아래·턱선·관자놀이·코 옆)과 하이라이터(콧대·광대 위·인중 위·턱 끝·이마) */
  contour: MVert[];
  highlight: MVert[];
  /** 피부 보정에서 뺄 이목구비(눈·속눈썹, 눈썹, 입술) 다각형 */
  features: Vec2[][];
  /** 얼굴 점 전체를 감싸는 사각형(px) — 피부 보정 계산 범위 */
  bbox: { x0: number; y0: number; x1: number; y1: number };
}

const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
const smooth = (e0: number, e1: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** 같은 길이의 점 줄(rows)을 이어 삼각형 망으로. val(행, 열)로 정점 진하기, 부위 좌표는 줄을 따라 잰 거리(얼굴 폭 단위) */
export function bandMesh(rows: Vec2[][], val: (r: number, c: number) => number, faceW = 1): MVert[] {
  const out: MVert[] = [];
  const n = rows[0].length;
  // s: 첫 줄을 따라 잰 길이, t: 줄 사이 거리 — 얼굴이 움직여도 반짝이 무늬가 피부에 붙어 있게
  const s: number[] = [0];
  for (let c = 1; c < n; c++) s.push(s[c - 1] + dist(rows[0][c - 1], rows[0][c]) / faceW);
  const vert = (r: number, c: number): MVert => {
    const p = rows[r][c];
    return { x: p.x, y: p.y, v: val(r, c), s: s[c], t: dist(rows[0][c], p) / faceW };
  };
  for (let r = 0; r + 1 < rows.length; r++) {
    for (let c = 0; c + 1 < n; c++) {
      const a = vert(r, c);
      const b = vert(r, c + 1);
      const d = vert(r + 1, c);
      const e = vert(r + 1, c + 1);
      out.push(a, b, d, b, e, d);
    }
  }
  return out;
}

/** 타원을 가운데(val 1)에서 가장자리(0)로 옅어지는 부채꼴 망으로 */
function ellipseMesh(c: Vec2, rx: number, ry: number, dir: Vec2, peak: number): MVert[] {
  const nx = -dir.y;
  const ny = dir.x;
  const K = 24;
  const ring = (f: number): Vec2[] =>
    Array.from({ length: K + 1 }, (_, k) => {
      const a = (k / K) * Math.PI * 2;
      const u = Math.cos(a) * rx * f;
      const v = Math.sin(a) * ry * f;
      return { x: c.x + dir.x * u + nx * v, y: c.y + dir.y * u + ny * v };
    });
  const center = Array.from({ length: K + 1 }, () => c);
  // 가운데 → 0.5 → 가장자리: 가우시안에 가까운 모양
  return bandMesh([center, ring(0.5), ring(1)], (r) => peak * [1, 0.62, 0][r]);
}

/** 눈꺼풀 선(바깥 → 안쪽)과 눈썹 아랫선 사이를 여러 줄로 나눈 아이섀도 망. height 0~1: 눈썹까지 얼마나 올릴지 */
function shadowMesh(p: Vec2[], upper: number[], browLower: number[], height: number, faceW: number, vis: number): MVert[] {
  const lid = upper.map((i) => p[i]);
  const brow = browLower.map((i) => p[i]);
  const browAt = (k: number): Vec2 => {
    const f = (k / (lid.length - 1)) * (brow.length - 1);
    const i = Math.min(brow.length - 2, Math.floor(f));
    return lerp(brow[i], brow[i + 1], f - i);
  };
  // 속눈썹 쪽이 가장 진하고 위로 갈수록 옅다(그라데이션 섀도)
  // 첫 줄은 속눈썹 선 아래(눈 안쪽 — 합성 때 눈 영역으로 빠짐): 흐려도 속눈썹 선에서 진하기가 빠지지 않게
  const hs = [-0.25, 0, 0.4, 0.75, 1];
  const vals = [1, 1, 0.95, 0.65, 0];
  const rows = hs.map((h) => lid.map((q, k) => lerp(q, browAt(k), height * h)));
  // 눈 앞머리(안쪽 끝)는 옅게: 콧대 쪽으로 번지지 않게
  const n = lid.length;
  return bandMesh(rows, (r, c) => vals[r] * vis * (0.55 + 0.45 * smooth(n - 1, n - 4, c)), faceW);
}

/** 선(점 줄)을 따라 폭 width의 띠. val(열 번호 0~1 위치) → 진하기 */
function lineBand(pts: Vec2[], width: number, val: (t: number) => number, faceW: number): MVert[] {
  const n = pts.length;
  const side = (sgn: number): Vec2[] =>
    pts.map((q, k) => {
      const a = pts[Math.max(0, k - 1)];
      const b = pts[Math.min(n - 1, k + 1)];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const l = Math.hypot(dx, dy) || 1;
      return { x: q.x - (dy / l) * width * 0.5 * sgn, y: q.y + (dx / l) * width * 0.5 * sgn };
    });
  const L = side(1);
  const R = side(-1);
  // 가운데 줄을 두어 띠 가로 방향으로도 가운데가 진하게
  return bandMesh([L, pts, R], (r, c) => val(c / (n - 1)) * (r === 1 ? 1 : 0.35), faceW);
}

/** 윤곽선(점 줄)에서 얼굴 안쪽으로 depth만큼 들어간 띠: 윤곽 쪽 진하고 안쪽으로 옅게 */
function inwardBand(pts: Vec2[], center: Vec2, depth: number, val: (t: number) => number, faceW: number): MVert[] {
  const n = pts.length;
  const inner = pts.map((q) => {
    const dx = center.x - q.x;
    const dy = center.y - q.y;
    const l = Math.hypot(dx, dy) || 1;
    return { x: q.x + (dx / l) * depth, y: q.y + (dy / l) * depth };
  });
  return bandMesh([pts, inner], (r, c) => (r === 0 ? val(c / (n - 1)) : 0), faceW);
}

/** 아래 속눈썹 선 아래 음영(눈꼬리 쪽이 진하고 앞쪽은 옅게). lower: 안쪽 → 바깥 */
function underMesh(p: Vec2[], lower: number[], upper: number[], faceW: number, vis: number): MVert[] {
  const lid = lower.map((i) => p[i]);
  const eye = [...upper, ...lower].map((i) => p[i]);
  const ctr = eye.reduce((a, q) => ({ x: a.x + q.x / eye.length, y: a.y + q.y / eye.length }), { x: 0, y: 0 });
  // 눈 높이: 윗·아랫 눈꺼풀 가운데 사이
  const h = Math.max(dist(p[upper[4]], p[lower[4]]), faceW * 0.02);
  const off = (k: number): Vec2[] =>
    lid.map((q) => {
      const dx = q.x - ctr.x;
      const dy = q.y - ctr.y;
      const l = Math.hypot(dx, dy) || 1;
      return { x: q.x + (dx / l) * h * k, y: q.y + (dy / l) * h * k };
    });
  const n = lid.length;
  // 첫 줄은 눈 안쪽(합성 때 빠짐)으로 조금 들여 흐림 뒤에도 속눈썹 선에서 진하기 유지
  return bandMesh([off(-0.25), lid, off(0.45)], (r, c) => [0.5, 0.5, 0][r] * vis * (0.3 + 0.7 * smooth(0, n - 3, c)), faceW);
}

/** 윗눈꺼풀 속눈썹 선 위로 두께가 있는 아이라이너 띠(눈꼬리로 갈수록 두껍고, 날개 끝에서 한 점으로 모임) */
function linerMesh(p: Vec2[], upper: number[], lower: number[], wing: number, faceW: number, vis: number): MVert[] {
  const lash = upper.map((i) => p[i]); // 바깥 → 안쪽
  const eye = [...upper, ...lower].map((i) => p[i]);
  const ctr = eye.reduce((a, q) => ({ x: a.x + q.x / eye.length, y: a.y + q.y / eye.length }), { x: 0, y: 0 });
  const eyeW = dist(lash[0], lash[lash.length - 1]);
  const T = faceW * 0.011;
  const n = lash.length;
  const up: Vec2[] = [];
  const low: Vec2[] = [];
  for (let k = 0; k < n; k++) {
    const a = lash[Math.max(0, k - 1)];
    const b = lash[Math.min(n - 1, k + 1)];
    let nx = -(b.y - a.y);
    let ny = b.x - a.x;
    const l = Math.hypot(nx, ny) || 1;
    nx /= l;
    ny /= l;
    // 눈 가운데에서 멀어지는 쪽(위)으로
    if (nx * (lash[k].x - ctr.x) + ny * (lash[k].y - ctr.y) < 0) {
      nx = -nx;
      ny = -ny;
    }
    // 두께: 안쪽 끝 0.25T → 바깥 끝 1.1T
    const f = 1 - k / (n - 1);
    const t = T * (0.25 + 0.85 * Math.pow(f, 1.4));
    up.push({ x: lash[k].x + nx * t, y: lash[k].y + ny * t });
    // 속눈썹 선과 틈이 생기지 않게 눈 쪽으로 살짝 겹친다
    low.push({ x: lash[k].x - nx * t * 0.2, y: lash[k].y - ny * t * 0.2 });
  }
  if (wing > 0) {
    // 날개 방향: 아랫눈꺼풀 바깥쪽 → 눈꼬리 방향을 조금 위로
    const outer = lash[0];
    const lowOuter = p[lower[lower.length - 2]];
    const dx = outer.x - lowOuter.x;
    const dy = outer.y - lowOuter.y;
    const l = Math.hypot(dx, dy) || 1;
    const upDir = { x: up[0].x - outer.x, y: up[0].y - outer.y };
    const ul = Math.hypot(upDir.x, upDir.y) || 1;
    const tip = {
      x: outer.x + (dx / l) * eyeW * wing + (upDir.x / ul) * eyeW * wing * 0.35,
      y: outer.y + (dy / l) * eyeW * wing + (upDir.y / ul) * eyeW * wing * 0.35,
    };
    up.unshift(tip);
    low.unshift(tip);
  }
  const m = low.length;
  return bandMesh([low, up], (_, c) => vis * (0.7 + 0.3 * smooth(m - 1, m - 4, c)), faceW);
}

/** 눈썹 띠: 아랫선·윗선을 짝지어, 꼬리는 한 점으로 모으고 앞머리는 옅게 */
function browMesh(p: Vec2[], lower: number[], upper: number[], faceW: number, vis: number): MVert[] {
  const lo = lower.map((i) => p[i]); // 바깥 → 안쪽
  const hi = [...upper].reverse().map((i) => p[i]); // 바깥 → 안쪽
  // 얼굴 점 눈썹 영역은 실제 눈썹보다 조금 크다: 가운데 선 쪽으로 10% 좁힌다
  const a = lo.map((q, k) => lerp(q, hi[k], 0.1));
  const b = hi.map((q, k) => lerp(q, lo[k], 0.1));
  // 꼬리: 바깥 끝 두 점의 가운데에서 조금 더 나간 한 점
  const m0 = lerp(a[0], b[0], 0.5);
  const m1 = lerp(a[1], b[1], 0.5);
  const tail = { x: m0.x + (m0.x - m1.x) * 0.3, y: m0.y + (m0.y - m1.y) * 0.3 };
  a[0] = lerp(a[0], m0, 0.35);
  b[0] = lerp(b[0], m0, 0.35);
  const vals = [0.75, 0.95, 1, 1, 0.8, 0.35]; // 꼬리 → 앞머리
  return bandMesh([[tail, ...a], [tail, ...b]], (_, c) => vals[c] * vis, faceW);
}

/** 쉐딩·하이라이터 망. 사람 오른쪽(R)·왼쪽(L)은 대칭 번호 */
function contourMeshes(p: Vec2[], faceW: number, visR: number, visL: number): { contour: MVert[]; highlight: MVert[] } {
  const P = (ids: number[]): Vec2[] => ids.map((i) => p[i]);
  const center = lerp(lerp(p[FACE_SIDE_R], p[FACE_SIDE_L], 0.5), lerp(p[10], p[152], 0.5), 0.5);
  const fade = (t: number): number => smooth(0, 0.25, t) * smooth(1, 0.6, t);
  const contour: MVert[] = [];
  const highlight: MVert[] = [];
  // 사람 오른쪽·왼쪽: 광대 아래, 턱선, 관자놀이, 코 옆(위 → 아래), 광대 위 하이라이트(바깥, 안쪽, 아래 기준)
  const sides = [
    { vis: visR, hollow: [93, 147, 187], jaw: [132, 58, 172, 136, 150], temple: [54, 21, 162], nose: [193, 122, 196, 3, 236], bone: [116, 118, 123] },
    { vis: visL, hollow: [323, 376, 411], jaw: [361, 288, 397, 365, 379], temple: [284, 251, 389], nose: [417, 351, 419, 248, 456], bone: [345, 347, 352] },
  ];
  for (const { vis, hollow, jaw, temple, nose, bone } of sides) {
    if (vis <= 0) continue;
    // 광대 아래: 귀 쪽이 진하고 입꼬리 쪽으로 사라진다
    contour.push(...lineBand(P(hollow), faceW * 0.075, (t) => vis * (1 - smooth(0.35, 1, t)), faceW));
    contour.push(...inwardBand(P(jaw), center, faceW * 0.09, (t) => vis * 0.7 * smooth(1, 0.7, t), faceW));
    contour.push(...inwardBand(P(temple), center, faceW * 0.08, () => vis * 0.55, faceW));
    contour.push(...lineBand(P(nose), faceW * 0.03, (t) => vis * 0.7 * smooth(1, 0.55, t) * smooth(0, 0.15, t), faceW));
    // 광대 위 하이라이트: 광대 바깥·안쪽 사이, 아래 기준점 쪽으로 조금 내린 타원
    const [b0, b1, b2] = P(bone);
    const c = lerp(lerp(b0, b1, 0.5), b2, 0.3);
    const dx = b1.x - b0.x;
    const dy = b1.y - b0.y;
    const l = Math.hypot(dx, dy) || 1;
    highlight.push(...ellipseMesh(c, faceW * 0.075, faceW * 0.03, { x: dx / l, y: dy / l }, vis * 0.9));
  }
  const vc = Math.min(visR, visL);
  // 콧대: 미간 아래 → 코끝 위
  highlight.push(...lineBand(P([168, 6, 197, 195, 5]), faceW * 0.03, (t) => vc * fade(t), faceW));
  const dirH = { x: (p[FACE_SIDE_L].x - p[FACE_SIDE_R].x) / faceW, y: (p[FACE_SIDE_L].y - p[FACE_SIDE_R].y) / faceW };
  // 인중 위(입술 산 바로 위), 턱 끝, 이마 가운데
  highlight.push(...ellipseMesh(lerp(p[0], p[164], 0.4), faceW * 0.035, faceW * 0.013, dirH, vc * 0.8));
  highlight.push(...ellipseMesh(lerp(p[200], p[199], 0.5), faceW * 0.04, faceW * 0.025, dirH, vc * 0.7));
  highlight.push(...ellipseMesh(lerp(p[9], p[151], 0.5), faceW * 0.06, faceW * 0.045, dirH, vc * 0.6));
  return { contour, highlight };
}

/** 478점(픽셀 좌표)에서 메이크업 영역을 만든다. overlip: -1(입술 안쪽으로) ~ 1(윤곽보다 크게) */
export function faceRegions(p: Vec2[], opts: { shadowHeight?: number; linerWing?: number; overlip?: number; lipStyle?: LipStyle } = {}): FaceRegions {
  const faceW = dist(p[FACE_SIDE_R], p[FACE_SIDE_L]);
  const sh = opts.shadowHeight ?? 0.62;
  const wing = opts.linerWing ?? 0.25;
  // 얼굴 돌림: 코끝에서 양쪽 얼굴 옆까지 거리 비율(정면 0.5)
  const nose = p[NOSE_TIP];
  const dR = dist(nose, p[FACE_SIDE_R]);
  const dL = dist(nose, p[FACE_SIDE_L]);
  const fracR = dR / (dR + dL || 1);
  const yaw = Math.max(-1, Math.min(1, (0.5 - fracR) * 2.5));
  const eyeMid = lerp(p[EYE_R_UPPER[0]], p[EYE_L_UPPER[0]], 0.5);
  const mouthMid = lerp(p[61], p[291], 0.5);
  const pitch = Math.max(-1, Math.min(1, (dist(eyeMid, nose) / (dist(eyeMid, mouthMid) || 1) - 0.55) * 3));
  // 먼 쪽 절반이 좁아질수록 옅게(코 뒤로 숨는 부분)
  const visR = smooth(0.1, 0.3, fracR);
  const visL = smooth(0.1, 0.3, 1 - fracR);
  const blush = (cheek: number, bone: number, frac: number, vis: number): MVert[] => {
    const c0 = p[cheek];
    const b = p[bone];
    // 볼 중심과 광대 바깥의 사이, 광대 쪽으로 조금 올린 위치
    const c = lerp(c0, b, 0.35);
    const dx = b.x - c0.x;
    const dy = b.y - c0.y;
    const l = Math.hypot(dx, dy) || 1;
    // 돌린 쪽 볼은 화면에서 좁아 보인다: 가로 반지름을 그 쪽 절반 폭에 맞춘다
    const k = Math.max(0.35, Math.min(1.3, frac * 2));
    return ellipseMesh(c, faceW * 0.16 * k, faceW * 0.105, { x: dx / l, y: dy / l }, vis);
  };
  // 오버립: 입술 가운데에서 위아래로 넓힌다(입꼬리는 거의 그대로)
  const over = Math.max(-1, Math.min(1, opts.overlip ?? 0));
  const outer = LIPS_OUTER.map((i) => p[i]);
  const lc = outer.reduce((a, q) => ({ x: a.x + q.x / outer.length, y: a.y + q.y / outer.length }), { x: 0, y: 0 });
  const lipsOuter = over === 0 ? outer : outer.map((q) => ({ x: lc.x + (q.x - lc.x) * (1 + 0.025 * over), y: lc.y + (q.y - lc.y) * (1 + 0.14 * over) }));
  const lipsInner = LIPS_INNER.map((i) => p[i]);
  const style = opts.lipStyle ?? 'full';
  const ring = (pts: Vec2[]): Vec2[] => [...pts, pts[0]];
  const lipRows =
    style === 'full'
      ? [ring(lipsInner), ring(lipsOuter)]
      : [ring(lipsInner), ring(lipsInner.map((q, k) => lerp(q, lipsOuter[k], 0.5))), ring(lipsOuter)];
  const lipVals = style === 'full' ? [1, 1] : style === 'gradient' ? [1, 0.7, 0.08] : [1, 0.9, 0.3];
  // 이목구비: 눈은 속눈썹까지 들어가게 가운데에서 1.35배로 넓힌다
  const grow = (pts: Vec2[], k: number): Vec2[] => {
    const c = pts.reduce((a, q) => ({ x: a.x + q.x / pts.length, y: a.y + q.y / pts.length }), { x: 0, y: 0 });
    return pts.map((q) => ({ x: c.x + (q.x - c.x) * k, y: c.y + (q.y - c.y) * k }));
  };
  const eyeR = [...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)].map((i) => p[i]);
  const eyeL = [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)].map((i) => p[i]);
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const q of p) {
    if (q.x < x0) x0 = q.x;
    if (q.y < y0) y0 = q.y;
    if (q.x > x1) x1 = q.x;
    if (q.y > y1) y1 = q.y;
  }
  return {
    faceW,
    yaw,
    pitch,
    visR,
    visL,
    lipsOuter,
    lipsInner,
    lip: bandMesh(lipRows, (r) => lipVals[r], faceW),
    lipBody: LIP_BODY.map((k) => lerp(outer[k], lipsInner[k], 0.5)),
    skinPts: SKIN_SAMPLES.map((i) => p[i]),
    shadow: [
      ...shadowMesh(p, EYE_R_UPPER, BROW_R_LOWER, sh, faceW, visR),
      ...shadowMesh(p, EYE_L_UPPER, BROW_L_LOWER, sh, faceW, visL),
      ...underMesh(p, EYE_R_LOWER, EYE_R_UPPER, faceW, visR),
      ...underMesh(p, EYE_L_LOWER, EYE_L_UPPER, faceW, visL),
    ],
    liner: [...linerMesh(p, EYE_R_UPPER, EYE_R_LOWER, wing, faceW, visR), ...linerMesh(p, EYE_L_UPPER, EYE_L_LOWER, wing, faceW, visL)],
    eyeR,
    eyeL,
    brow: [...browMesh(p, BROW_R_LOWER, BROW_R_UPPER, faceW, visR), ...browMesh(p, BROW_L_LOWER, BROW_L_UPPER, faceW, visL)],
    blush: [...blush(CHEEK_R, CHEEKBONE_R, fracR, visR), ...blush(CHEEK_L, CHEEKBONE_L, 1 - fracR, visL)],
    ...contourMeshes(p, faceW, visR, visL),
    features: [
      grow(eyeR, 1.35),
      grow(eyeL, 1.35),
      grow([...BROW_R_LOWER, ...BROW_R_UPPER].map((i) => p[i]), 1.15),
      grow([...BROW_L_LOWER, ...BROW_L_UPPER].map((i) => p[i]), 1.15),
      grow(outer, 1.08),
    ],
    bbox: { x0, y0, x1, y1 },
  };
}

/**
 * 단순 다각형을 삼각형으로 나눈다(귀 자르기). 결과는 점 번호 3개씩.
 * 얼굴 영역처럼 점이 수십 개인 다각형용(매 프레임 계산해도 가볍다).
 */
export function triangulate(poly: Vec2[]): number[] {
  const n = poly.length;
  if (n < 3) return [];
  let area = 0;
  for (let i = 0; i < n; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % n];
    area += a.x * b.y - b.x * a.y;
  }
  const ccw = area > 0;
  const idx = Array.from({ length: n }, (_, i) => i);
  const out: number[] = [];
  const cross = (a: Vec2, b: Vec2, c: Vec2): number => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  const inside = (p: Vec2, a: Vec2, b: Vec2, c: Vec2): boolean => {
    const d1 = cross(a, b, p);
    const d2 = cross(b, c, p);
    const d3 = cross(c, a, p);
    return ccw ? d1 >= 0 && d2 >= 0 && d3 >= 0 : d1 <= 0 && d2 <= 0 && d3 <= 0;
  };
  let guard = 0;
  while (idx.length > 3 && guard++ < n * n) {
    let clipped = false;
    for (let k = 0; k < idx.length; k++) {
      const i0 = idx[(k + idx.length - 1) % idx.length];
      const i1 = idx[k];
      const i2 = idx[(k + 1) % idx.length];
      const a = poly[i0];
      const b = poly[i1];
      const c = poly[i2];
      const cr = cross(a, b, c);
      if (ccw ? cr <= 0 : cr >= 0) continue; // 오목한 꼭짓점
      let ok = true;
      for (const j of idx) {
        if (j === i0 || j === i1 || j === i2) continue;
        if (inside(poly[j], a, b, c)) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      out.push(i0, i1, i2);
      idx.splice(k, 1);
      clipped = true;
      break;
    }
    if (!clipped) break; // 자기 교차 등: 남은 부분은 부채꼴로
  }
  for (let k = 1; k + 1 < idx.length; k++) out.push(idx[0], idx[k], idx[k + 1]);
  return out;
}

/** 두께 있는 선(폴리라인)을 삼각형 띠로. 끝으로 갈수록 가늘어지게(taper: 0~1). */
export function strokeStrip(line: Vec2[], width: number, taper = 0.7): Vec2[] {
  const out: Vec2[] = [];
  for (let i = 0; i < line.length; i++) {
    const a = line[Math.max(0, i - 1)];
    const b = line[Math.min(line.length - 1, i + 1)];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const l = Math.hypot(dx, dy) || 1;
    const nx = -dy / l;
    const ny = dx / l;
    const t = i / Math.max(1, line.length - 1);
    // 가운데가 두껍고 양 끝이 가늘다(바깥 끝 = 앞쪽이 더 두꺼운 날개)
    const w = (width / 2) * (1 - taper * Math.pow(Math.abs(t - 0.35) / 0.65, 2));
    out.push({ x: line[i].x + nx * w, y: line[i].y + ny * w }, { x: line[i].x - nx * w, y: line[i].y - ny * w });
  }
  return out;
}
