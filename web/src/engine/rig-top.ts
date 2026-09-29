// 상의 착용 리그: 상품 이미지 좌표 → 화면 좌표 변형.
//
// 몸판: 어깨선과 몸통 축(어깨 중점 → 엉덩이 중점)으로 만든 "신체 좌표계"에 얹는다.
//       가로 배율은 어깨 폭, 세로 배율은 몸통 길이에 맞추되 상품 고유의 기장 비율은 살린다.
//       팔을 내리면 겨드랑이 아래 옆선이 팔 중심선을 넘지 않게 오므린다(실제 옷처럼 팔 안쪽에 모임).
// 소매: 상품 사진의 소매를 네 꼭짓점(어깨점·겨드랑이·끝단 바깥·끝단 안쪽)으로 된 곡면으로 보고,
//       어깨점·겨드랑이는 몸판 진동 둘레에 그대로 붙이고(틈 없음), 윗선은 팔 바깥쪽,
//       아랫선은 팔 안쪽을 따라가게 한다. 긴팔은 팔꿈치에서 굽는다.

import { LM, type BodyFrame } from './body.ts';
import { PART, type GarmentAsset, type PartMesh } from './garment.ts';
import { add, dist, dot, lerp, mid, norm, perp, scale, smoothstep, sub, type Vec2 } from './math.ts';

export interface TopFit {
  /** 어깨 봉제선 폭 ÷ 어깨 랜드마크 폭. 봉제선은 관절보다 조금 바깥에 놓인다. */
  seamWidth: number;
  /** 옷 전체를 어깨선 위로 올리는 양(어깨 폭 비율). 관절 중심보다 어깨 표면이 높기 때문. */
  lift: number;
  /** 기장 배율(1 = 상품 비율 그대로). */
  length: number;
}

export const DEFAULT_TOP_FIT: TopFit = { seamWidth: 1.08, lift: 0.06, length: 1 };

/**
 * 기장 기준: 보통 티셔츠(몸길이 ÷ 어깨 봉제선 폭 ≈ 1.55)는 밑단이 어깨→엉덩이 관절 거리의 약 1.12배에 온다.
 * 상품마다 이 비율이 다르면(크롭·롱) 그만큼 밑단 위치를 옮겨 상품의 기장 특성을 살린다.
 */
const STD_LENGTH_RATIO = 1.55;
const STD_HEM_T = 1.12;

interface SleeveRig {
  mesh: PartMesh;
  shoulderKey: string;
  armpitKey: string;
  lmShoulder: number;
  lmElbow: number;
  lmWrist: number;
  /** 착용자 왼쪽이면 +1(몸 좌표계 u 방향), 오른쪽이면 -1. */
  side: 1 | -1;
  /** 정점별 곡면 좌표: s 뿌리(0) → 끝단(1), v 윗선(0) → 아랫선(1) */
  s: Float32Array;
  v: Float32Array;
  /** 윗선(어깨점→끝단 바깥)·아랫선(겨드랑이→끝단 안쪽) 길이(상품 픽셀) */
  topLen: number;
  bottomLen: number;
  /** 뿌리(진동 둘레)·끝단 폭(상품 픽셀) */
  rootWidth: number;
  endWidth: number;
}

export class TopRig {
  readonly asset: GarmentAsset;
  /** 그리는 순서(뒤 → 앞). update 후 갱신된다. */
  order: PartMesh[] = [];
  private readonly torso: PartMesh | undefined;
  private readonly neckInner: PartMesh | undefined;
  private readonly sleeves: SleeveRig[] = [];
  private readonly gMid: Vec2;
  private readonly gw: number;
  /** 어깨선 → 밑단 세로 길이(상품 픽셀) */
  private readonly gLen: number;
  /** 어깨선 → 겨드랑이 세로 길이(상품 픽셀) */
  private readonly armpitDy: number;

  constructor(asset: GarmentAsset) {
    this.asset = asset;
    const kp = asset.kp;
    this.gMid = mid(kp.shoulderL, kp.shoulderR);
    this.gw = dist(kp.shoulderL, kp.shoulderR) / (asset.info.widthScale ?? 1);
    const hemY = kp.hemL && kp.hemR ? (kp.hemL.y + kp.hemR.y) / 2 : this.gMid.y + this.gw * STD_LENGTH_RATIO;
    this.gLen = Math.max(1, hemY - this.gMid.y);
    const armpitY = kp.armpitL && kp.armpitR ? (kp.armpitL.y + kp.armpitR.y) / 2 : this.gMid.y + this.gLen * 0.28;
    this.armpitDy = Math.max(1, armpitY - this.gMid.y);
    this.torso = asset.meshes.find((m) => m.partId === PART.torso);
    this.neckInner = asset.meshes.find((m) => m.partId === PART.neckInner);
    const defs = [
      { partId: PART.sleeveL, sh: 'shoulderL', ap: 'armpitL', out: 'sleeveOuterL', inn: 'sleeveInnerL', side: 1 as const, lms: LM.leftShoulder, lme: LM.leftElbow, lmw: LM.leftWrist },
      { partId: PART.sleeveR, sh: 'shoulderR', ap: 'armpitR', out: 'sleeveOuterR', inn: 'sleeveInnerR', side: -1 as const, lms: LM.rightShoulder, lme: LM.rightElbow, lmw: LM.rightWrist },
    ];
    for (const d of defs) {
      const mesh = asset.meshes.find((m) => m.partId === d.partId);
      if (!mesh || !kp[d.sh] || !kp[d.ap] || !kp[d.out] || !kp[d.inn]) continue;
      const p00 = kp[d.sh];
      const p10 = kp[d.out];
      const p01 = kp[d.ap];
      const p11 = kp[d.inn];
      const s = new Float32Array(mesh.vertexCount);
      const v = new Float32Array(mesh.vertexCount);
      for (let i = 0; i < mesh.vertexCount; i++) {
        const [si, vi] = inverseBilinear({ x: mesh.src[i * 2], y: mesh.src[i * 2 + 1] }, p00, p10, p01, p11);
        // 소매 사각형 밖(격자 여백)의 외삽이 폭주하지 않게 제한한다.
        s[i] = Number.isFinite(si) ? Math.min(1.6, Math.max(-0.6, si)) : 0;
        v[i] = Number.isFinite(vi) ? Math.min(1.8, Math.max(-0.8, vi)) : 0.5;
      }
      this.sleeves.push({
        mesh,
        shoulderKey: d.sh,
        armpitKey: d.ap,
        lmShoulder: d.lms,
        lmElbow: d.lme,
        lmWrist: d.lmw,
        side: d.side,
        s,
        v,
        topLen: dist(p00, p10),
        bottomLen: dist(p01, p11),
        rootWidth: dist(p00, p01),
        endWidth: dist(p10, p11),
      });
    }
  }

  update(body: BodyFrame, fit: TopFit = DEFAULT_TOP_FIT): void {
    const k = (body.shoulderW * fit.seamWidth) / this.gw;
    // 세로 배율: 몸통 길이에 맞춰 밑단 위치를 정하되 상품의 기장 비율을 반영한다.
    const hemT = STD_HEM_T * (this.gLen / this.gw / STD_LENGTH_RATIO) * fit.length;
    const kv = (hemT * body.axisLen) / this.gLen;
    const lift = fit.lift * body.shoulderW;
    const sm = body.shoulderMid;
    const axisDir = norm(sub(body.hipMid, sm), body.down);
    const { u, down, hipU } = body;
    const gMid = this.gMid;

    // 팔을 내려 팔이 몸통 옆에 있으면, 겨드랑이 아래 몸판 옆선이 팔 중심선을 넘지 않게 오므린다.
    const armLim = [1, -1].map((side) => armLateralLimit(body, sm, axisDir, u, side as 1 | -1));
    const armpitYr = this.armpitDy * kv - lift;
    const mapInto = (gx: number, gy: number, out: Float32Array, o: number, constrain = false): void => {
      let xr = (gx - gMid.x) * k;
      // 어깨선 위(목둘레)는 가로와 같은 배율, 아래(몸통)는 몸통 길이 배율
      const dy = gy - gMid.y;
      const yr = (dy < 0 ? dy * k : dy * kv) - lift;
      if (constrain && yr > 0) {
        const lim = armLim[xr >= 0 ? 0 : 1](yr);
        const ax = Math.abs(xr);
        if (ax > lim) {
          const squeezed = lim + (ax - lim) * 0.2;
          const rw = smoothstep(armpitYr * 0.3, armpitYr, yr);
          xr = Math.sign(xr) * (ax + (squeezed - ax) * rw);
        }
      }
      let ox: number;
      let oy: number;
      let lx = u.x;
      let ly = u.y;
      if (yr <= 0) {
        ox = sm.x + down.x * yr;
        oy = sm.y + down.y * yr;
      } else {
        ox = sm.x + axisDir.x * yr;
        oy = sm.y + axisDir.y * yr;
        const w = Math.min(1, yr / body.axisLen);
        const lxx = u.x + (hipU.x - u.x) * w;
        const lyy = u.y + (hipU.y - u.y) * w;
        const l = Math.hypot(lxx, lyy) || 1;
        lx = lxx / l;
        ly = lyy / l;
      }
      out[o] = ox + lx * xr;
      out[o + 1] = oy + ly * xr;
    };
    const tmp = new Float32Array(2);
    const mapPoint = (p: Vec2, constrain = false): Vec2 => {
      mapInto(p.x, p.y, tmp, 0, constrain);
      return { x: tmp[0], y: tmp[1] };
    };

    // 소매가 진동 둘레에 붙어 몸판 옆을 덮으므로 몸판은 오므리지 않는다(오므리면 겨드랑이 아래 틈이 생김).
    // 팔 위치 제한(armLim)은 겨드랑이 위치를 팔 안쪽으로 당길 때만 쓴다.
    for (const m of [this.torso, this.neckInner]) {
      if (!m) continue;
      for (let i = 0; i < m.vertexCount; i++) mapInto(m.src[i * 2], m.src[i * 2 + 1], m.dst, i * 2);
    }

    const sleeveOrder: { mesh: PartMesh; z: number }[] = [];
    for (const sl of this.sleeves) {
      const kp = this.asset.kp;
      // 진동 둘레 양 끝: 몸판과 똑같이 변형해 소매 뿌리가 몸판에 틈 없이 붙게 한다.
      const cap = mapPoint(kp[sl.shoulderKey]);
      const pit = mapPoint(kp[sl.armpitKey], true);
      const arm = new ArmPath(body, sl.lmShoulder, sl.lmElbow, sl.lmWrist, sl.side);
      // 바깥쪽(몸 중심 → 어깨 쪽) 법선 부호
      const sign = dot(arm.normal(0), sub(cap, sm)) >= 0 ? 1 : -1;
      const w = body.shoulderW;
      const armR = (a: number): number => w * (0.15 - 0.045 * Math.min(1, Math.max(0, a / arm.length)));
      const topLen = sl.topLen * k;
      const botLen = sl.bottomLen * k;
      // 겨드랑이가 팔 중심선 위 어디쯤인지(팔을 내리면 어깨에서 조금 아래)
      const pitArc = Math.max(0, dot(sub(pit, arm.at(0)), arm.dir(0)));
      const halfW = (s: number): number => ((sl.rootWidth + (sl.endWidth - sl.rootWidth) * Math.min(1, Math.max(0, s))) * k) / 2;
      const outerAt = (a: number, off: number): Vec2 => add(arm.at(a), scale(arm.normal(a), sign * off));
      const innerAt = (a: number, off: number): Vec2 => add(arm.at(a), scale(arm.normal(a), -sign * off));
      const off0 = Math.max(armR(0), halfW(0.35));
      const corrTop = sub(cap, outerAt(0, off0));
      const corrBot = sub(pit, innerAt(pitArc, off0));
      const m = sl.mesh;
      for (let i = 0; i < m.vertexCount; i++) {
        const s = sl.s[i];
        const v = sl.v[i];
        const aTop = s * topLen;
        const aBot = pitArc + s * botLen;
        // 소매 폭: 상품 폭(뿌리→끝단)과 팔 두께 중 큰 값. 뿌리 쪽은 진동 둘레 폭이 크므로 35% 지점 폭으로 제한.
        const off = Math.max(armR((aTop + aBot) / 2), halfW(Math.max(s, 0.35)));
        const fade = 1 - Math.min(1, Math.max(0, s));
        const top = add(outerAt(aTop, off), scale(corrTop, fade));
        const bot = add(innerAt(aBot, off), scale(corrBot, fade));
        m.dst[i * 2] = top.x + (bot.x - top.x) * v;
        m.dst[i * 2 + 1] = top.y + (bot.y - top.y) * v;
      }
      // 손목이 카메라에 가까울수록 나중에(앞에) 그린다.
      sleeveOrder.push({ mesh: m, z: body.z[sl.lmWrist] + body.z[sl.lmElbow] });
    }
    sleeveOrder.sort((a, b) => b.z - a.z);
    // 소매 뿌리가 진동 둘레에 붙어 있으므로 소매를 몸판 "앞"에 그린다(팔을 내리면 소매가 몸판 옆을 덮음).
    this.order = [
      ...(this.neckInner ? [this.neckInner] : []),
      ...(this.torso ? [this.torso] : []),
      ...sleeveOrder.map((s) => s.mesh),
    ];
  }
}

/** 어깨 관절 → 팔꿈치 → 손목 폴리라인. 팔꿈치에서 방향이 부드럽게 바뀐다. */
class ArmPath {
  readonly s: Vec2;
  readonly e: Vec2;
  readonly d1: Vec2;
  readonly d2: Vec2;
  readonly upper: number;
  readonly length: number;
  private readonly joint: number;

  constructor(body: BodyFrame, lmS: number, lmE: number, lmW: number, side: 1 | -1) {
    const { shoulder, elbow, wrist } = armPose(body, lmS, lmE, lmW, side);
    this.s = shoulder;
    this.e = elbow;
    this.d1 = norm(sub(elbow, shoulder), body.down);
    this.d2 = norm(sub(wrist, elbow), this.d1);
    this.upper = Math.max(1, dist(shoulder, elbow));
    this.length = this.upper + dist(elbow, wrist);
    this.joint = 0.1 * this.length;
  }

  at(a: number): Vec2 {
    if (a <= this.upper) return add(this.s, scale(this.d1, a));
    return add(this.e, scale(this.d2, a - this.upper));
  }

  dir(a: number): Vec2 {
    const w = smoothstep(this.upper - this.joint, this.upper + this.joint, a);
    return norm(lerp(this.d1, this.d2, w), this.d1);
  }

  /** 진행 방향의 시계 방향 법선(부호는 호출하는 쪽에서 정한다) */
  normal(a: number): Vec2 {
    return perp(this.dir(a));
  }
}

/** 사각형(p00, p10, p01, p11) 안의 점 p의 쌍선형 좌표 (s, v). 사각형 밖이면 외삽 값. */
export function inverseBilinear(p: Vec2, p00: Vec2, p10: Vec2, p01: Vec2, p11: Vec2): [number, number] {
  const cr = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;
  const e = sub(p10, p00);
  const f = sub(p01, p00);
  const g = { x: p00.x - p10.x + p11.x - p01.x, y: p00.y - p10.y + p11.y - p01.y };
  const h = sub(p, p00);
  const k2 = cr(g, f);
  const k1 = cr(e, f) + cr(h, g);
  const k0 = cr(h, e);
  let v: number;
  if (Math.abs(k2) < 1e-6 * Math.max(1, Math.abs(k1))) {
    v = -k0 / k1;
  } else {
    const disc = Math.sqrt(Math.max(0, k1 * k1 - 4 * k0 * k2));
    const v1 = (-k1 - disc) / (2 * k2);
    const v2 = (-k1 + disc) / (2 * k2);
    // [0,1]에 더 가까운 해를 고른다
    const dist01 = (x: number): number => (x < 0 ? -x : x > 1 ? x - 1 : 0);
    v = dist01(v1) <= dist01(v2) ? v1 : v2;
  }
  const denX = e.x + g.x * v;
  const denY = e.y + g.y * v;
  const s = Math.abs(denX) > Math.abs(denY) ? (h.x - f.x * v) / denX : (h.y - f.y * v) / denY;
  return [s, v];
}

/** 팔 랜드마크가 잘 보이지 않으면 몸 옆으로 내린 자세로 대신한다. */
export function armPose(body: BodyFrame, lmS: number, lmE: number, lmW: number, side: 1 | -1): { shoulder: Vec2; elbow: Vec2; wrist: Vec2 } {
  const w = body.shoulderW;
  const shoulder = body.p[lmS];
  const hang = norm(add(body.down, scale(body.u, side * 0.18)));
  const elbowDefault = add(shoulder, scale(hang, 0.8 * w));
  const we = smoothstep(0.25, 0.6, body.vis[lmE]);
  const elbow = lerp(elbowDefault, body.p[lmE], we);
  const upperDir = norm(sub(elbow, shoulder), hang);
  const wristDefault = add(elbow, scale(upperDir, 0.7 * w));
  const ww = smoothstep(0.25, 0.6, body.vis[lmW]) * we;
  const wrist = lerp(wristDefault, body.p[lmW], ww);
  return { shoulder, elbow, wrist };
}

/**
 * 몸 좌표계(어깨 중점 기준, 가로 u·세로 몸통 축)에서 한쪽 팔 중심선의 가로 위치를 세로 위치의 함수로 돌려준다.
 * 팔이 몸 옆에 내려와 있을 때만 제한(팔 중심선 - 여유)을 주고, 팔을 들었거나 몸 앞을 가로지르면 제한하지 않는다.
 */
function armLateralLimit(body: BodyFrame, sm: Vec2, axisDir: Vec2, u: Vec2, side: 1 | -1): (yr: number) => number {
  const lms = side === 1 ? [LM.leftShoulder, LM.leftElbow, LM.leftWrist] : [LM.rightShoulder, LM.rightElbow, LM.rightWrist];
  const { shoulder, elbow, wrist } = armPose(body, lms[0], lms[1], lms[2], side);
  const w = body.shoulderW;
  const toBody = (p: Vec2): { x: number; y: number } => {
    const d = sub(p, sm);
    return { x: dot(d, u) * side, y: dot(d, axisDir) };
  };
  const pts = [toBody(shoulder), toBody(elbow), toBody(wrist)];
  return (yr: number): number => {
    for (let i = 0; i < 2; i++) {
      const a = pts[i];
      const b = pts[i + 1];
      const lo = Math.min(a.y, b.y);
      const hi = Math.max(a.y, b.y);
      if (yr < lo || yr > hi || hi - lo < 1e-3) continue;
      const x = a.x + ((b.x - a.x) * (yr - a.y)) / (b.y - a.y);
      // 팔이 자기 쪽 옆에 있을 때만(몸 앞을 가로지르면 제한 없음)
      if (x < 0.3 * w || x > 1.2 * w) return Infinity;
      return x - 0.05 * w;
    }
    return Infinity;
  };
}
