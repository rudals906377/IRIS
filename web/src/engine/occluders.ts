// 가림(occlusion) 도형: 옷보다 앞에 있을 수 있는 팔·손을 캡슐(둥근 막대)로 근사한다.
// MediaPipe는 손 마스크를 주지 않으므로 관절점으로 직접 만든다.
//
// 채널 의미(렌더러의 가림 버퍼 RGB):
//  R: 팔 중심부(가는 캡슐) → 몸판만 가린다(옷을 입은 팔이 몸판 앞에 있을 때)
//  G: 팔 주변(넓은 캡슐) → 이 안의 "몸 피부" 픽셀이 몸판을 가린다(맨살 팔의 정확한 윤곽)
//  B: 손 → 모든 부위를 가린다

import { LM, type BodyFrame } from './body.ts';
import { armPose } from './rig-top.ts';
import { add, lerp, scale, smoothstep, type Vec2 } from './math.ts';

export interface Capsule {
  a: Vec2;
  b: Vec2;
  r: number;
  /** [R, G, B] 기록 강도 0~1 */
  ch: [number, number, number];
}

export function buildArmOccluders(body: BodyFrame): Capsule[] {
  const w = body.shoulderW;
  const out: Capsule[] = [];
  const arms = [
    { s: LM.leftShoulder, e: LM.leftElbow, wr: LM.leftWrist, pi: LM.leftPinky, ix: LM.leftIndex, th: LM.leftThumb, side: 1 as const },
    { s: LM.rightShoulder, e: LM.rightElbow, wr: LM.rightWrist, pi: LM.rightPinky, ix: LM.rightIndex, th: LM.rightThumb, side: -1 as const },
  ];
  for (const a of arms) {
    const { shoulder, elbow, wrist } = armPose(body, a.s, a.e, a.wr, a.side);
    const ve = smoothstep(0.3, 0.6, body.vis[a.e]);
    const vw = smoothstep(0.3, 0.6, body.vis[a.wr]) * ve;
    if (ve > 0) {
      // 위팔은 어깨에서 조금 떨어진 곳부터(어깨 자체는 옷이 덮어야 함)
      out.push({ a: lerp(shoulder, elbow, 0.35), b: elbow, r: 0.09 * w, ch: [ve, 0, 0] });
      out.push({ a: lerp(shoulder, elbow, 0.25), b: elbow, r: 0.18 * w, ch: [0, ve, 0] });
    }
    if (vw > 0) {
      out.push({ a: elbow, b: wrist, r: 0.08 * w, ch: [vw, 0, 0] });
      out.push({ a: elbow, b: wrist, r: 0.15 * w, ch: [0, vw, 0] });
      // 손: 손목에서 검지·새끼·엄지 끝 방향으로 캡슐 3개 + 손바닥
      const hand = [a.pi, a.ix, a.th]
        .filter((i) => body.vis[i] > 0.3)
        .map((i) => body.p[i]);
      const hr = 0.075 * w;
      for (const tip of hand) {
        // 랜드마크는 손가락 뿌리 근처라서 조금 연장한다
        const ext = add(wrist, scale({ x: tip.x - wrist.x, y: tip.y - wrist.y }, 1.6));
        out.push({ a: wrist, b: ext, r: hr, ch: [vw, vw, vw] });
      }
      if (hand.length >= 2) {
        out.push({ a: hand[0], b: hand[1], r: hr * 1.1, ch: [vw, vw, vw] });
      }
      out.push({ a: wrist, b: wrist, r: hr * 1.3, ch: [vw, vw, vw] });
    }
  }
  return out;
}
