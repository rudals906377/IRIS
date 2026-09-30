// 얼굴 점(478개, 정규화 좌표) → 떨림을 줄인 픽셀 좌표 + 신뢰도.
// - 얼굴을 놓치면 짧게 유지한 뒤 서서히 사라진다(틀린 자리에 화장이 남지 않게).

import { AdaptiveFilter2 } from './filters.ts';
import type { Vec2 } from './math.ts';

export interface FaceFrame {
  t: number;
  /** 478개 점의 픽셀 좌표 */
  p: Vec2[];
  /** 0~1. 얼굴을 놓치면 줄어들고 효과의 불투명도에 곱해진다 */
  confidence: number;
}

// 떨림 줄이기는 두 단계로 한다(Procrustes 분해).
//  ① 머리 전체 움직임(이동·회전·크기): 움직임이 잡음 수준이면 강하게 누르고, 크면 바로 따라간다.
//  ② 얼굴 안 모양(표정·점별 잔떨림): 머리 움직임을 뺀 얼굴 좌표에서 점마다 같은 방식으로 거른다.
// 점마다 따로 거르던 방식보다 빠르게 고개를 돌릴 때 덜 늦고(화장이 얼굴에서 미끄러지지 않음),
// 가만히 있을 때 덜 떨린다. 기준값은 얼굴 폭에 비례(얼굴이 클수록 점 잡음도 px로 크다).
// 합성 영상 측정(tools/harness 참고): 빠른 흔들기 오차 16.7 → 4.1px, 말할 때 입술 오차 3.0 → 1.4px, 정지 떨림 0.18 → 0.08~0.13px
const POSE_D0 = 0.006;
const POSE_D1 = 0.03;
const POSE_AMIN = 0.05;
const SHAPE_D0 = 0.007;
const SHAPE_D1 = 0.024;
const SHAPE_AMIN = 0.06;
/** 머리 자세를 잴 때 쓰는 잘 안 움직이는 점(눈꼬리, 콧대, 이마, 얼굴 옆) */
const RIGID = [33, 133, 362, 263, 1, 4, 5, 6, 168, 197, 195, 10, 151, 9, 234, 454, 127, 356, 93, 323];
const LOST_HOLD_MS = 150;
const FADE_OUT_MS = 200;
const FADE_IN_MS = 100;

export class FaceTracker {
  private pose = [new AdaptiveFilter2(0, 1, 1), new AdaptiveFilter2(0, 1, 1)];
  private shape: AdaptiveFilter2[] = [];
  /** 기준 모양(첫 프레임, 가운데 기준 px)과 그 크기 */
  private ref: Vec2[] | null = null;
  private refScale = 1;
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
    let cx = 0;
    let cy = 0;
    for (const j of RIGID) {
      cx += raw[j].x / RIGID.length;
      cy += raw[j].y / RIGID.length;
    }
    if (!this.ref || this.shape.length !== raw.length) {
      this.ref = raw.map((q) => ({ x: q.x - cx, y: q.y - cy }));
      this.refScale = Math.sqrt(RIGID.reduce((a, j) => a + this.ref![j].x ** 2 + this.ref![j].y ** 2, 0) / RIGID.length) || 1;
      this.shape = raw.map(() => new AdaptiveFilter2(0, 1, 1));
      for (const f of this.pose) f.reset();
    }
    // ① 기준 모양 → 현재의 닮음 변환(회전 th, 크기 s)
    let sxx = 0;
    let sxy = 0;
    let ss = 0;
    for (const j of RIGID) {
      const r = this.ref[j];
      const x = raw[j].x - cx;
      const y = raw[j].y - cy;
      sxx += r.x * x + r.y * y;
      sxy += r.x * y - r.y * x;
      ss += r.x * r.x + r.y * r.y;
    }
    const th = Math.atan2(sxy, sxx);
    const s = Math.hypot(sxx, sxy) / ss || 1;
    // 이동은 px, 회전·크기는 기준 크기를 곱해 px로 환산해 같은 기준으로 거른다
    const k = this.refScale;
    for (const f of this.pose) {
      f.d0 = POSE_D0 * faceW;
      f.d1 = POSE_D1 * faceW;
      f.aMin = POSE_AMIN;
    }
    const pc = this.pose[0].filter(cx, cy, tMs);
    const pr = this.pose[1].filter(s * k, th * k, tMs);
    const fs = pr.x / k;
    const fth = pr.y / k;
    // ② 얼굴 안 좌표(머리 움직임을 되돌린 기준 크기 px)에서 점마다 거른다
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const fc = Math.cos(fth);
    const fsn = Math.sin(fth);
    const refW = faceW / s;
    const p = raw.map((q, j) => {
      const f = this.shape[j];
      f.d0 = SHAPE_D0 * refW;
      f.d1 = SHAPE_D1 * refW;
      f.aMin = SHAPE_AMIN;
      const u = q.x - cx;
      const v = q.y - cy;
      const l = f.filter((u * c + v * sn) / s, (-u * sn + v * c) / s, tMs);
      return { x: pc.x + fs * (l.x * fc - l.y * fsn), y: pc.y + fs * (l.x * fsn + l.y * fc) };
    });
    this.last = { t: tMs, p, confidence: this.confidence };
    return this.last;
  }

  reset(): void {
    this.ref = null;
    this.shape = [];
    for (const f of this.pose) f.reset();
    this.last = null;
    this.confidence = 0;
  }
}
