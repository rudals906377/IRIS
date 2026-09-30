// 착용자 몸통 윤곽 측정: 매 프레임 분할(사람 확률)에서 몸통 양옆 가장자리까지의 거리를 높이별로 잰다.
// 옷을 이 윤곽에 맞춰 입혀 몸에 달라붙게 한다. 팔이 몸 옆에 붙어 있으면 팔 앞에서 멈춘다(팔은 몸통이 아님).
//
// 결과는 몸 좌표계(어깨 중점 기준, 가로 u = 착용자 오른쪽→왼쪽, 세로 = 몸통 축)의 반폭이다.
//   left[k]:  +u 쪽(착용자 왼쪽) 가장자리까지 거리, right[k]: -u 쪽 거리
//   k번째 표본의 높이 = TORSO_T[k] × 몸통 길이(어깨선 → 엉덩이)

import { LM, type BodyFrame } from './body.ts';
import { add, dot, norm, scale, sub, type Vec2 } from './math.ts';

export const TORSO_SAMPLES = 16;
/** 표본 높이(몸통 길이 비율): 어깨선 조금 아래부터 엉덩이 아래까지 */
export const TORSO_T = Array.from({ length: TORSO_SAMPLES }, (_, k) => 0.04 + (k / (TORSO_SAMPLES - 1)) * 1.26);

export interface TorsoWidths {
  left: Float32Array;
  right: Float32Array;
  /** 측정에 성공한 표본 비율 0~1 */
  quality: number;
  /** 팔 반두께: [착용자 왼팔, 오른팔] × ARM_SAMPLES(어깨→손목), 픽셀. 못 재면 NaN */
  arms?: [Float32Array, Float32Array];
}

export const ARM_SAMPLES = 8;

export interface SegPlane {
  /** RGBA8, A = 사람 확률 */
  flags: Uint8ClampedArray | Uint8Array;
  width: number;
  height: number;
}

function distToSeg(p: Vec2, a: Vec2, b: Vec2): number {
  const ab = sub(b, a);
  const l2 = dot(ab, ab) || 1;
  const t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / l2));
  return Math.hypot(p.x - a.x - ab.x * t, p.y - a.y - ab.y * t);
}

/** 한 프레임 측정(시간 평활은 TorsoTracker가 한다). 측정 못 한 표본은 NaN. */
export function measureTorso(seg: SegPlane, body: BodyFrame): TorsoWidths {
  const { width: vw, height: vh } = body;
  const sx = seg.width / vw;
  const sy = seg.height / vh;
  const sw = body.frontW;
  const axis = sub(body.hipMid, body.shoulderMid);
  const axisDir = norm(axis, body.down);
  const left = new Float32Array(TORSO_SAMPLES).fill(NaN);
  const right = new Float32Array(TORSO_SAMPLES).fill(NaN);
  // 팔: 어깨 관절 부근(몸통과 붙은 곳)은 빼고 위팔 중간부터 손목까지
  const arms: [Vec2, Vec2][] = [];
  for (const [s, e, w] of [
    [LM.leftShoulder, LM.leftElbow, LM.leftWrist],
    [LM.rightShoulder, LM.rightElbow, LM.rightWrist],
  ]) {
    if (body.vis[e] < 0.3) continue;
    const S = body.p[s];
    const E = body.p[e];
    arms.push([add(S, scale(sub(E, S), 0.4)), E]);
    if (body.vis[w] > 0.3) arms.push([E, body.p[w]]);
  }
  const armR = sw * 0.11;
  const prob = (p: Vec2): number => {
    const x = Math.floor(p.x * sx);
    const y = Math.floor(p.y * sy);
    if (x < 0 || y < 0 || x >= seg.width || y >= seg.height) return 0;
    return seg.flags[(y * seg.width + x) * 4 + 3];
  };
  /** 옷(분할의 '옷' 클래스) 확률 0~255 */
  const cloth = (p: Vec2): number => {
    const x = Math.floor(p.x * sx);
    const y = Math.floor(p.y * sy);
    if (x < 0 || y < 0 || x >= seg.width || y >= seg.height) return 0;
    return seg.flags[(y * seg.width + x) * 4 + 2];
  };
  const step = Math.max(1, 1 / Math.max(sx, sy)); // 분할 픽셀 하나 크기(영상 픽셀)
  let ok = 0;
  for (let k = 0; k < TORSO_SAMPLES; k++) {
    const t = TORSO_T[k];
    const wgt = Math.min(1, t);
    const lat = norm(add(scale(body.u, 1 - wgt), scale(body.hipU, wgt)), body.u);
    const c = add(body.shoulderMid, scale(axisDir, t * body.axisLen));
    if (prob(c) < 128) continue;
    for (const side of [1, -1] as const) {
      let d = 0;
      const max = sw * 1.1;
      let hit = NaN;
      // 지금 입은 옷의 가장자리에 맞춘다: 몸통이 '옷'으로 분할돼 있으면 옷이 끝나는 곳(맨살 팔·배경)에서 멈춘다.
      let clothRun = 0;
      let n = 0;
      while (d < max) {
        d += step;
        const p = add(c, scale(lat, side * d));
        if (prob(p) < 128) {
          hit = d;
          break;
        }
        n++;
        const cl = cloth(p) >= 110;
        if (cl) clothRun++;
        if (!cl && d > sw * 0.28 && clothRun / n > 0.7) {
          hit = d;
          break;
        }
        if (d > sw * 0.25 && arms.some(([a, b]) => distToSeg(p, a, b) < armR)) {
          hit = d;
          break;
        }
      }
      // 너무 좁거나(분할 구멍) 넓으면(팔을 벌린 옷자락 등) 믿지 않는다
      if (Number.isFinite(hit) && hit > sw * 0.28 && hit < sw * 0.95) {
        (side === 1 ? left : right)[k] = hit;
      }
    }
    if (Number.isFinite(left[k]) && Number.isFinite(right[k])) ok++;
  }
  return { left, right, quality: ok / TORSO_SAMPLES, arms: [measureArm(body, prob, step, 1), measureArm(body, prob, step, -1)] };
}

/**
 * 팔 반두께: 어깨→팔꿈치→손목 선을 따라가며, 팔 바깥쪽(몸 반대편)으로 사람 영역이 끝나는 곳까지 거리.
 * 안쪽은 몸통과 붙어 있을 수 있어 쓰지 않는다. 두꺼운 후드 소매도 이 두께로 덮는다.
 */
function measureArm(body: BodyFrame, prob: (p: Vec2) => number, step: number, side: 1 | -1): Float32Array {
  const out = new Float32Array(ARM_SAMPLES).fill(NaN);
  const [s, e, w] = side === 1 ? [LM.leftShoulder, LM.leftElbow, LM.leftWrist] : [LM.rightShoulder, LM.rightElbow, LM.rightWrist];
  if (body.vis[e] < 0.4) return out;
  const S = body.p[s];
  const E = body.p[e];
  const W = body.vis[w] > 0.4 ? body.p[w] : add(E, scale(norm(sub(E, S), body.down), 0.7 * body.frontW));
  const sw = body.frontW;
  const l1 = Math.hypot(E.x - S.x, E.y - S.y);
  const l2 = Math.hypot(W.x - E.x, W.y - E.y);
  const L = l1 + l2 || 1;
  for (let k = 0; k < ARM_SAMPLES; k++) {
    const a = ((k + 0.5) / ARM_SAMPLES) * L;
    const [P, Q, t] = a <= l1 ? [S, E, a / (l1 || 1)] : [E, W, (a - l1) / (l2 || 1)];
    const c = add(P, scale(sub(Q, P), t));
    if (prob(c) < 128) continue;
    const dir = norm(sub(Q, P), body.down);
    let n = { x: -dir.y, y: dir.x };
    // 바깥쪽 = 몸 중심에서 멀어지는 쪽
    if (dot(n, sub(c, body.shoulderMid)) < 0) n = scale(n, -1);
    // 양쪽으로 재서 얇은 쪽을 쓴다(팔을 들면 한쪽이 머리·몸통과 붙어 두께가 부풀려진다)
    const reach = (dirN: Vec2): number => {
      let d = 0;
      while (d < sw * 0.4) {
        d += step;
        if (prob(add(c, scale(dirN, d))) < 128) break;
      }
      return d;
    };
    const dOut = reach(n);
    const dIn = reach(scale(n, -1));
    const d = Math.min(dOut, dIn * 1.1);
    if (d > sw * 0.04 && d < sw * 0.25) out[k] = d;
  }
  return out;
}

/** 체형 기본값(측정이 없을 때): 어깨 폭 대비 반폭 */
export function defaultHalfWidth(t: number, shoulderW: number): number {
  // 가슴(0.55) → 허리(0.47) → 엉덩이(0.52)
  const v = t < 0.5 ? 0.55 - 0.16 * t : t < 0.85 ? 0.47 : 0.47 + 0.12 * Math.min(1, (t - 0.85) / 0.4);
  return v * shoulderW;
}

/**
 * 측정값을 시간·높이로 평활한다. 빠진 표본은 이웃 표본 → 이전 프레임 → 체형 기본값 순으로 메운다.
 * 결과는 어깨 폭으로 나눈 비율로 보관해(거리 변화에 강함) 호출 시 현재 어깨 폭을 곱한다.
 */
export class TorsoTracker {
  private left = new Float32Array(TORSO_SAMPLES).fill(NaN);
  private right = new Float32Array(TORSO_SAMPLES).fill(NaN);
  private arms = [new Float32Array(ARM_SAMPLES).fill(NaN), new Float32Array(ARM_SAMPLES).fill(NaN)];
  quality = 0;

  update(m: TorsoWidths | null, shoulderW: number, alpha = 0.35): void {
    if (!m) return;
    this.quality = this.quality * 0.8 + m.quality * 0.2;
    for (const [src, dst] of [
      [m.left, this.left],
      [m.right, this.right],
    ] as const) {
      const filled = fillGaps(src);
      for (let k = 0; k < TORSO_SAMPLES; k++) {
        const v = filled[k] / shoulderW;
        if (!Number.isFinite(v)) continue;
        dst[k] = Number.isFinite(dst[k]) ? dst[k] + (v - dst[k]) * alpha : v;
      }
    }
    if (m.arms) {
      for (let i = 0; i < 2; i++) {
        const filled = fillGaps(m.arms[i]);
        for (let k = 0; k < ARM_SAMPLES; k++) {
          const v = filled[k] / shoulderW;
          if (!Number.isFinite(v)) continue;
          const d = this.arms[i];
          d[k] = Number.isFinite(d[k]) ? d[k] + (v - d[k]) * alpha : v;
        }
      }
    }
  }

  /** 팔 반두께(픽셀). side 1 = 착용자 왼팔, a01 = 어깨(0)→손목(1). 모르면 NaN */
  armAt(side: 1 | -1, a01: number, shoulderW: number): number {
    const arr = this.arms[side === 1 ? 0 : 1];
    const x = Math.max(0, Math.min(ARM_SAMPLES - 1, a01 * ARM_SAMPLES - 0.5));
    const i = Math.floor(x);
    const j = Math.min(ARM_SAMPLES - 1, i + 1);
    const a = arr[i];
    const b = arr[j];
    const v = Number.isFinite(a) && Number.isFinite(b) ? a + (b - a) * (x - i) : Number.isFinite(a) ? a : b;
    return v * shoulderW;
  }

  reset(): void {
    this.left.fill(NaN);
    this.right.fill(NaN);
    for (const a of this.arms) a.fill(NaN);
    this.quality = 0;
  }

  /** 높이 t(몸통 길이 비율)에서 +u(착용자 왼쪽)·-u 쪽 반폭(픽셀) */
  at(t: number, shoulderW: number): { left: number; right: number } {
    const f = (arr: Float32Array): number => {
      const x = Math.max(0, Math.min(TORSO_SAMPLES - 1, ((t - TORSO_T[0]) / (TORSO_T[TORSO_SAMPLES - 1] - TORSO_T[0])) * (TORSO_SAMPLES - 1)));
      const i = Math.floor(x);
      const j = Math.min(TORSO_SAMPLES - 1, i + 1);
      const a = arr[i];
      const b = arr[j];
      const v = Number.isFinite(a) && Number.isFinite(b) ? a + (b - a) * (x - i) : Number.isFinite(a) ? a : b;
      const def = defaultHalfWidth(t, 1);
      // 측정 품질이 낮으면 체형 기본값 쪽으로
      const w = Math.min(1, this.quality * 1.6);
      return (Number.isFinite(v) ? def + (v - def) * w : def) * shoulderW;
    };
    return { left: f(this.left), right: f(this.right) };
  }
}

/** 빠진 표본(NaN)을 양옆 표본의 선형 보간으로 채운다. 3표본 중앙값으로 튀는 값도 누른다. */
export function fillGaps(src: Float32Array): Float32Array {
  const n = src.length;
  const out = new Float32Array(n).fill(NaN);
  for (let k = 0; k < n; k++) {
    const v = [src[k - 1], src[k], src[k + 1]].filter((x) => x !== undefined && Number.isFinite(x)) as number[];
    if (Number.isFinite(src[k]) && v.length === 3) out[k] = v.sort((a, b) => a - b)[1];
    else out[k] = src[k];
  }
  let prev = -1;
  for (let k = 0; k < n; k++) {
    if (!Number.isFinite(out[k])) continue;
    if (prev >= 0 && k - prev > 1) {
      for (let j = prev + 1; j < k; j++) out[j] = out[prev] + ((out[k] - out[prev]) * (j - prev)) / (k - prev);
    }
    prev = k;
  }
  return out;
}
