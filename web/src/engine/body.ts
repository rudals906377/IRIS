// MediaPipe Pose 랜드마크(정규화 좌표)를 픽셀 좌표의 안정된 신체 상태로 바꾼다.
// - One Euro 필터로 떨림 제거
// - 엉덩이가 화면 밖일 때(웹캠 상반신 구도) 체형 비율로 보완
// - 추적이 끊기면 짧게 유지한 뒤 서서히 사라지게 함(틀린 위치에 옷을 그리지 않기 위해)

import { DEFAULT_ONE_EURO, OneEuro, PointFilter, type OneEuroParams } from './filters.ts';
import { clamp01, dist, dot, lerp, mid, norm, perp, scale, smoothstep, sub, add, type Vec2 } from './math.ts';

export const LM = {
  nose: 0,
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftPinky: 17,
  rightPinky: 18,
  leftIndex: 19,
  rightIndex: 20,
  leftThumb: 21,
  rightThumb: 22,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28,
} as const;

export interface RawLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface BodyFrame {
  t: number;
  width: number;
  height: number;
  /** 33개 랜드마크의 필터된 픽셀 좌표. */
  p: Vec2[];
  vis: number[];
  /** 픽셀 단위 상대 깊이(작을수록 카메라에 가까움, 엉덩이 중점 기준). */
  z: number[];
  /** 0~1. 추적이 끊기면 줄어들고, 옷의 불투명도에 곱해진다. */
  confidence: number;
  shoulderMid: Vec2;
  shoulderW: number;
  /** 착용자 오른쪽 어깨 → 왼쪽 어깨 방향 단위 벡터(정면이면 화면 오른쪽). */
  u: Vec2;
  /** 어깨선에 수직인 아래 방향. */
  down: Vec2;
  hipMid: Vec2;
  hipU: Vec2;
  axisLen: number;
  /** 몸통 좌우 회전 추정(라디안, 깊이만 사용한 원시값). 0이면 정면. */
  yaw: number;
  /**
   * 평활한 몸통 회전(라디안). 0 정면, +는 착용자 왼쪽 어깨가 뒤로 간 방향, ±π는 등을 보인 상태.
   * 어깨 깊이(z) 차이와 어깨 폭 ÷ 몸통 길이의 줄어듦을 함께 쓴다.
   */
  turn: number;
}

/** 엉덩이가 보이지 않을 때 쓰는 체형 비율(어깨 랜드마크 간 거리 대비 어깨–엉덩이 거리). */
const TORSO_RATIO = 1.35;
const LOST_HOLD_MS = 200;
const FADE_OUT_MS = 250;
const FADE_IN_MS = 120;

export class BodyTracker {
  private readonly filters: PointFilter[] = [];
  private readonly zFilters: OneEuro[] = [];
  private last: BodyFrame | null = null;
  private lastSeen = -1;
  private confidence = 0;
  private lastT = -1;
  /** 정면일 때의 어깨 폭 ÷ 몸통 길이(사람마다 다르므로 관찰하며 갱신, 처음엔 관찰값으로 시작) */
  private frontRatio = NaN;
  private turnS = 0;
  private turnC = 1;

  constructor(params: OneEuroParams = DEFAULT_ONE_EURO) {
    for (let i = 0; i < 33; i++) {
      this.filters.push(new PointFilter(params));
      this.zFilters.push(new OneEuro({ ...params, minCutoff: params.minCutoff * 0.7 }));
    }
  }

  setFilterParams(params: OneEuroParams): void {
    for (const f of this.filters) f.setParams(params);
  }

  /**
   * landmarks가 null이면 사람을 찾지 못한 프레임. 반환값이 null이면 옷을 그리지 않는다.
   */
  update(landmarks: RawLandmark[] | null, tMs: number, width: number, height: number): BodyFrame | null {
    const dt = this.lastT < 0 ? 0 : tMs - this.lastT;
    this.lastT = tMs;

    const usable =
      landmarks !== null &&
      (landmarks[LM.leftShoulder].visibility ?? 1) > 0.5 &&
      (landmarks[LM.rightShoulder].visibility ?? 1) > 0.5;

    if (!usable) {
      if (this.last === null) return null;
      if (tMs - this.lastSeen > LOST_HOLD_MS) {
        this.confidence = Math.max(0, this.confidence - dt / FADE_OUT_MS);
      }
      if (this.confidence <= 0) {
        this.reset();
        return null;
      }
      return { ...this.last, t: tMs, confidence: this.confidence };
    }

    this.lastSeen = tMs;
    this.confidence = Math.min(1, this.confidence + (dt > 0 ? dt / FADE_IN_MS : 0.34));

    const p: Vec2[] = [];
    const vis: number[] = [];
    const z: number[] = [];
    for (let i = 0; i < 33; i++) {
      const lm = landmarks![i];
      p.push(this.filters[i].filter(lm.x * width, lm.y * height, tMs));
      vis.push(lm.visibility ?? 1);
      z.push(this.zFilters[i].filter(lm.z * width, tMs));
    }

    const sL = p[LM.leftShoulder];
    const sR = p[LM.rightShoulder];
    const shoulderMid = mid(sL, sR);
    const shoulderW = Math.max(1, dist(sL, sR));
    const u = norm(sub(sL, sR));
    let down = perp(u);
    const nose = p[LM.nose];
    if (vis[LM.nose] > 0.5 && dot(sub(nose, shoulderMid), down) > 0) down = scale(down, -1);

    // 엉덩이: 보이면 추정값을 쓰고, 안 보이면 체형 비율로 보완한다.
    const hipVis = Math.min(vis[LM.leftHip], vis[LM.rightHip]);
    const hipW = smoothstep(0.3, 0.75, hipVis);
    const priorHip = add(shoulderMid, scale(down, TORSO_RATIO * shoulderW));
    const measuredHip = mid(p[LM.leftHip], p[LM.rightHip]);
    let hipMid = lerp(priorHip, measuredHip, hipW);
    // 엉덩이 예측이 비정상적으로 짧거나 길면 보정한다.
    const axis = sub(hipMid, shoulderMid);
    const axisLenRaw = Math.hypot(axis.x, axis.y);
    const minLen = 0.8 * shoulderW;
    const maxLen = 2.2 * shoulderW;
    if (axisLenRaw < minLen || axisLenRaw > maxLen || dot(axis, down) < 0.3 * axisLenRaw) {
      hipMid = priorHip;
    }
    const axisLen = Math.max(1, dist(hipMid, shoulderMid));
    const hipUMeasured = norm(sub(p[LM.leftHip], p[LM.rightHip]), u);
    const hipU = norm(lerp(u, dot(hipUMeasured, u) > 0.5 ? hipUMeasured : u, hipW));

    // 정면이면 0, 옆으로 돌수록 커지고, 등을 보이면 ±π에 가까워진다.
    const yaw = Math.atan2(z[LM.leftShoulder] - z[LM.rightShoulder], sL.x - sR.x);

    // 평활 회전. 깊이(z) 추정은 정면에서도 ±20° 흔들리므로 방향(부호)에만 쓰고,
    // 크기는 어깨 폭이 정면일 때보다 얼마나 줄었는지로 잰다(엉덩이가 보일 때). 등을 보이면 어깨 좌우가 뒤집힌다.
    let turn = 0;
    const backView = sL.x < sR.x;
    if (hipW > 0.8) {
      const r = shoulderW / axisLen;
      this.frontRatio = Number.isFinite(this.frontRatio) ? Math.min(0.95, Math.max(0.55, Math.max(this.frontRatio * 0.998, r))) : r;
      let mag = Math.acos(Math.min(1, r / this.frontRatio));
      if (backView) mag = Math.PI - mag;
      const dead = 0.22;
      mag = mag > dead ? mag - dead * 0.5 : 0;
      const sign = yaw === 0 ? 1 : Math.sign(yaw);
      turn = sign * mag;
    } else {
      // 엉덩이가 안 보이면(상반신 구도) 깊이 각도만 쓸 수 있다: 작은 값은 잡음으로 보고 무시
      const dead = 0.4;
      const a = Math.abs(yaw);
      turn = a > dead ? Math.sign(yaw) * Math.min(Math.PI, (a - dead) * 1.4) : 0;
      if (backView) turn = Math.sign(yaw || 1) * Math.PI;
    }
    const a = dt > 0 ? Math.min(1, dt / 90) : 1;
    this.turnS += (Math.sin(turn) - this.turnS) * a;
    this.turnC += (Math.cos(turn) - this.turnC) * a;
    turn = Math.atan2(this.turnS, this.turnC);

    const frame: BodyFrame = {
      t: tMs,
      width,
      height,
      p,
      vis,
      z,
      confidence: clamp01(this.confidence),
      shoulderMid,
      shoulderW,
      u,
      down,
      hipMid,
      hipU,
      axisLen,
      yaw,
      turn,
    };
    this.last = frame;
    return frame;
  }

  reset(): void {
    for (const f of this.filters) f.reset();
    for (const f of this.zFilters) f.reset();
    this.last = null;
    this.confidence = 0;
    this.turnS = 0;
    this.turnC = 1;
    this.frontRatio = NaN;
  }
}
