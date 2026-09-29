// 상의 착용 리그: 상품 이미지 좌표 → 화면 좌표 변형.
//
// 몸판: 어깨선과 몸통 축(어깨 중점 → 엉덩이 중점)으로 만든 "신체 좌표계"에 상품을 실제 비율대로 얹는다.
//       어깨가 기울거나 몸을 옆으로 굽히면 좌표계가 함께 기울어 옷이 따라간다.
// 소매: 소매 뿌리(어깨 봉제선 ~ 겨드랑이)에서 시작해 위팔 → 아래팔 방향 폴리라인을 따라 쓸어 붙인다.
//       뿌리 근처는 몸판 변형과 섞어 봉제선이 벌어지지 않게 한다.

import { LM, type BodyFrame } from './body.ts';
import { PART, type GarmentAsset, type PartMesh } from './garment.ts';
import { add, dist, dot, lerp, mid, norm, perp, scale, smoothstep, sub, type Vec2 } from './math.ts';

export interface TopFit {
  /** 어깨 봉제선 폭 ÷ 어깨 랜드마크 폭. 봉제선은 관절보다 조금 바깥에 놓인다. */
  seamWidth: number;
  /** 옷 전체를 어깨선 위로 올리는 양(어깨 폭 비율). 관절 중심보다 어깨 표면이 높기 때문. */
  lift: number;
}

export const DEFAULT_TOP_FIT: TopFit = { seamWidth: 1.24, lift: 0.16 };

interface SleeveRig {
  mesh: PartMesh;
  shoulderKey: string;
  armpitKey: string;
  lmShoulder: number;
  lmElbow: number;
  lmWrist: number;
  /** 착용자 왼쪽이면 +1(몸 좌표계 u 방향), 오른쪽이면 -1. */
  side: 1 | -1;
  /** 정점별 소매 축 방향 위치(뿌리 0 → 끝 1). */
  s: Float32Array;
  /** 정점별 소매 축에서의 수직 거리(상품 픽셀, 바깥쪽 +). */
  t: Float32Array;
  length: number;
  /** 어깨 기준점의 소매 좌표(s, t). */
  sShoulder: number;
  tShoulder: number;
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

  constructor(asset: GarmentAsset) {
    this.asset = asset;
    const kp = asset.kp;
    this.gMid = mid(kp.shoulderL, kp.shoulderR);
    this.gw = dist(kp.shoulderL, kp.shoulderR);
    this.torso = asset.meshes.find((m) => m.partId === PART.torso);
    this.neckInner = asset.meshes.find((m) => m.partId === PART.neckInner);
    const defs = [
      { partId: PART.sleeveL, sh: 'shoulderL', ap: 'armpitL', out: 'sleeveOuterL', inn: 'sleeveInnerL', side: 1 as const, lms: LM.leftShoulder, lme: LM.leftElbow, lmw: LM.leftWrist },
      { partId: PART.sleeveR, sh: 'shoulderR', ap: 'armpitR', out: 'sleeveOuterR', inn: 'sleeveInnerR', side: -1 as const, lms: LM.rightShoulder, lme: LM.rightElbow, lmw: LM.rightWrist },
    ];
    for (const d of defs) {
      const mesh = asset.meshes.find((m) => m.partId === d.partId);
      if (!mesh || !kp[d.sh] || !kp[d.ap] || !kp[d.out] || !kp[d.inn]) continue;
      const root = mid(kp[d.sh], kp[d.ap]);
      const end = mid(kp[d.out], kp[d.inn]);
      const axis = norm(sub(end, root));
      const length = dist(root, end);
      let normal = perp(axis);
      if (dot(normal, sub(kp[d.sh], root)) < 0) normal = scale(normal, -1);
      const s = new Float32Array(mesh.vertexCount);
      const t = new Float32Array(mesh.vertexCount);
      for (let i = 0; i < mesh.vertexCount; i++) {
        const rel = { x: mesh.src[i * 2] - root.x, y: mesh.src[i * 2 + 1] - root.y };
        s[i] = dot(rel, axis) / length;
        t[i] = dot(rel, normal);
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
        t,
        length,
        sShoulder: dot(sub(kp[d.sh], root), axis) / length,
        tShoulder: dot(sub(kp[d.sh], root), normal),
      });
    }
  }

  update(body: BodyFrame, fit: TopFit = DEFAULT_TOP_FIT): void {
    const k = (body.shoulderW * fit.seamWidth) / this.gw;
    const lift = fit.lift * body.shoulderW;
    const sm = body.shoulderMid;
    const axisDir = norm(sub(body.hipMid, sm), body.down);
    const { u, down, hipU } = body;
    const gMid = this.gMid;

    const mapInto = (gx: number, gy: number, out: Float32Array, o: number): void => {
      const xr = (gx - gMid.x) * k;
      const yr = (gy - gMid.y) * k - lift;
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
    const mapPoint = (p: Vec2): Vec2 => {
      mapInto(p.x, p.y, tmp, 0);
      return { x: tmp[0], y: tmp[1] };
    };

    for (const m of [this.torso, this.neckInner]) {
      if (!m) continue;
      for (let i = 0; i < m.vertexCount; i++) mapInto(m.src[i * 2], m.src[i * 2 + 1], m.dst, i * 2);
    }

    const sleeveOrder: { mesh: PartMesh; z: number }[] = [];
    for (const sl of this.sleeves) {
      // 소매 윗선은 어깨 끝점(봉제선 위쪽)에 고정하고, 소매 축은 위팔 → 아래팔 방향을 따라간다.
      // 팔을 내리면 소매가 몸판 옆을 덮으며 내려오고, 팔을 들면 소매 단면이 봉제선과 나란해진다.
      const top = mapPoint(this.asset.kp[sl.shoulderKey]);
      const { elbow, wrist, shoulder } = armPose(body, sl.lmShoulder, sl.lmElbow, sl.lmWrist, sl.side);
      const d1 = norm(sub(elbow, shoulder), down);
      const d2 = norm(sub(wrist, elbow), d1);
      const upper = Math.max(1, dist(shoulder, elbow));
      // 바깥쪽(몸 중심에서 어깨 쪽)을 향하는 법선을 소매 윗선 방향으로 삼는다.
      const n1 = perp(d1);
      const sign = dot(n1, sub(top, sm)) >= 0 ? 1 : -1;
      const sleeveLen = sl.length * k;
      const joint = 0.12 * sleeveLen;
      // 어깨 기준점이 top에 정확히 오도록 소매 축의 시작점을 잡는다.
      const c0x = top.x - n1.x * sign * sl.tShoulder * k;
      const c0y = top.y - n1.y * sign * sl.tShoulder * k;
      const c1x = c0x + d1.x * upper;
      const c1y = c0y + d1.y * upper;
      const m = sl.mesh;
      for (let i = 0; i < m.vertexCount; i++) {
        const arc = (sl.s[i] - sl.sShoulder) * sleeveLen;
        let cx: number;
        let cy: number;
        if (arc <= upper) {
          cx = c0x + d1.x * arc;
          cy = c0y + d1.y * arc;
        } else {
          cx = c1x + d2.x * (arc - upper);
          cy = c1y + d2.y * (arc - upper);
        }
        // 팔꿈치 근처에서는 방향을 부드럽게 바꿔 접힘을 줄인다.
        const wj = smoothstep(upper - joint, upper + joint, arc);
        const dx = d1.x + (d2.x - d1.x) * wj;
        const dy = d1.y + (d2.y - d1.y) * wj;
        const dl = Math.hypot(dx, dy) || 1;
        const nx = (-dy / dl) * sign;
        const ny = (dx / dl) * sign;
        const off = sl.t[i] * k;
        m.dst[i * 2] = cx + nx * off;
        m.dst[i * 2 + 1] = cy + ny * off;
      }
      // 손목이 카메라에 가까울수록 나중에(앞에) 그린다.
      sleeveOrder.push({ mesh: m, z: body.z[sl.lmWrist] + body.z[sl.lmElbow] });
    }
    sleeveOrder.sort((a, b) => b.z - a.z);
    this.order = [
      ...(this.neckInner ? [this.neckInner] : []),
      ...(this.torso ? [this.torso] : []),
      ...sleeveOrder.map((s) => s.mesh),
    ];
  }
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
