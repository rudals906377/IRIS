// 얼굴 점(478개, 정규화 좌표) → 떨림을 줄인 픽셀 좌표 + 신뢰도.
// - 얼굴을 놓치면 짧게 유지한 뒤 서서히 사라진다(틀린 자리에 화장이 남지 않게).

import { RigidShapeFilter, type RigidShapeParams } from './filters.ts';
import type { Vec2 } from './math.ts';

export interface FaceFrame {
  t: number;
  /** 478개 점의 픽셀 좌표 */
  p: Vec2[];
  /** 0~1. 얼굴을 놓치면 줄어들고 효과의 불투명도에 곱해진다 */
  confidence: number;
}

// 떨림 줄이기는 두 단계로 한다(RigidShapeFilter, Procrustes 분해).
//  ① 머리 전체 움직임(이동·회전·크기): 움직임이 잡음 수준이면 강하게 누르고, 크면 바로 따라간다.
//  ② 얼굴 안 모양(표정·점별 잔떨림): 머리 움직임을 뺀 얼굴 좌표에서 점마다 같은 방식으로 거른다.
// 점마다 따로 거르던 방식보다 빠르게 고개를 돌릴 때 덜 늦고(화장이 얼굴에서 미끄러지지 않음),
// 가만히 있을 때 덜 떨린다. 기준값은 얼굴 폭에 비례(얼굴이 클수록 점 잡음도 px로 크다).
// 합성 영상 측정(tools/harness 참고): 빠른 흔들기 오차 16.7 → 4.1px, 말할 때 입술 오차 3.0 → 1.4px, 정지 떨림 0.18 → 0.08~0.13px
const FACE_FILTER: RigidShapeParams = {
  // 눈꼬리, 콧대, 이마, 얼굴 옆
  rigid: [33, 133, 362, 263, 1, 4, 5, 6, 168, 197, 195, 10, 151, 9, 234, 454, 127, 356, 93, 323],
  poseD0: 0.006,
  poseD1: 0.03,
  poseAMin: 0.05,
  shapeD0: 0.007,
  shapeD1: 0.024,
  shapeAMin: 0.06,
};
const LOST_HOLD_MS = 150;
const FADE_OUT_MS = 200;
const FADE_IN_MS = 100;

export class FaceTracker {
  private readonly filter = new RigidShapeFilter(FACE_FILTER);
  private last: FaceFrame | null = null;
  private lastSeen = -1;
  private lastT = -1;
  private confidence = 0;

  update(landmarks: { x: number; y: number }[] | null, tMs: number, width: number, height: number): FaceFrame | null {
    const dt = this.lastT < 0 ? 0 : tMs - this.lastT;
    this.lastT = tMs;
    if (!landmarks || landmarks.length < 468) {
      if (!this.last) return null;
      if (tMs - this.lastSeen > LOST_HOLD_MS) this.confidence = Math.max(0, this.confidence - dt / FADE_OUT_MS);
      if (this.confidence <= 0) {
        this.reset();
        return null;
      }
      return { ...this.last, t: tMs, confidence: this.confidence };
    }
    this.lastSeen = tMs;
    this.confidence = Math.min(1, this.confidence + (dt > 0 ? dt / FADE_IN_MS : 0.34));
    const raw = landmarks.map((l) => ({ x: l.x * width, y: l.y * height }));
    const faceW = Math.hypot(raw[454].x - raw[234].x, raw[454].y - raw[234].y) || 1;
    const p: Vec2[] = this.filter.filter(raw, faceW, tMs);
    this.last = { t: tMs, p, confidence: this.confidence };
    return this.last;
  }

  reset(): void {
    this.filter.reset();
    this.last = null;
    this.confidence = 0;
  }
}
