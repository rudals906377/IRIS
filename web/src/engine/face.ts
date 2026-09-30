// 얼굴 점(478개, 정규화 좌표) → 떨림을 줄인 픽셀 좌표 + 신뢰도.
// - One Euro 필터: 가만히 있을 때는 떨림을 강하게 누르고, 빨리 움직이면 지연 없이 따라간다.
// - 얼굴을 놓치면 짧게 유지한 뒤 서서히 사라진다(틀린 자리에 화장이 남지 않게).

import { PointFilter, type OneEuroParams } from './filters.ts';
import type { Vec2 } from './math.ts';

export interface FaceFrame {
  t: number;
  /** 478개 점의 픽셀 좌표 */
  p: Vec2[];
  /** 0~1. 얼굴을 놓치면 줄어들고 효과의 불투명도에 곱해진다 */
  confidence: number;
}

/** 얼굴은 몸보다 작고 세밀해 떨림 억제를 조금 더 강하게 */
const FACE_ONE_EURO: OneEuroParams = { minCutoff: 1.2, beta: 0.01, dCutoff: 1.0 };
const LOST_HOLD_MS = 150;
const FADE_OUT_MS = 200;
const FADE_IN_MS = 100;

export class FaceTracker {
  private filters: PointFilter[] = [];
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
    if (this.filters.length !== landmarks.length) this.filters = landmarks.map(() => new PointFilter(FACE_ONE_EURO));
    this.lastSeen = tMs;
    this.confidence = Math.min(1, this.confidence + (dt > 0 ? dt / FADE_IN_MS : 0.34));
    const p = landmarks.map((l, i) => this.filters[i].filter(l.x * width, l.y * height, tMs));
    this.last = { t: tMs, p, confidence: this.confidence };
    return this.last;
  }

  reset(): void {
    for (const f of this.filters) f.reset();
    this.last = null;
    this.confidence = 0;
  }
}
