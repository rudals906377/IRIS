// 손톱 위치 계산(순수 계산, 단위 테스트 대상): 손 점 21개(MediaPipe Hand Landmarker) → 손톱마다 사각형(중심·축·크기).
// 손톱은 손가락 마지막 마디(DIP → 끝) 위에 있다. 손바닥이 카메라를 향하면 손톱이 안 보이므로
// 손등 방향(세 점으로 만든 면의 법선)과 왼손/오른손 정보로 보임 정도를 정한다.

import { dist, lerp, norm, sub, type Vec2 } from '../engine/math.ts';

/** 손가락별 (끝, 마지막 마디, 뿌리 관절) 점 번호: 엄지, 검지, 중지, 약지, 소지 */
export const FINGERS: [number, number, number][] = [
  [4, 3, 2],
  [8, 7, 5],
  [12, 11, 9],
  [16, 15, 13],
  [20, 19, 17],
];

export interface HandPoints {
  /** 21점 픽셀 좌표 */
  p: Vec2[];
  /** 21점 깊이(손목 기준, 작을수록 카메라에 가까움). 정규화 좌표계 */
  z: number[];
  /** 모델이 판단한 손: 'Left' | 'Right' (영상 기준) */
  handed: string;
}

export interface NailQuad {
  /** 손톱 가운데 */
  c: Vec2;
  /** 손가락 방향(뿌리 → 끝) 단위 벡터 */
  dir: Vec2;
  /** 손톱 길이·폭(px) */
  len: number;
  width: number;
  /** 0~1: 손등이 카메라를 향한 정도(손톱이 보이는 정도) */
  vis: number;
  /** 손가락 피부 기준점(마지막 마디 관절 조금 뒤): 손톱과 피부를 밝기로 가르는 기준 */
  ref: Vec2;
  /** 손가락 번호(0 엄지 ~ 4 소지) */
  finger: number;
}

/**
 * 손등이 카메라를 향한 정도(−1 ~ 1): 손목·검지 뿌리·소지 뿌리가 이루는 면의 법선 z와 손 종류로 판단.
 * 모델이 'Right'라고 한 손은 손등이 카메라를 향할 때 이 값이 양수다(손등이 보이는 시험 영상 두 개로 확인한 부호).
 */
export function backFacing(h: HandPoints): number {
  const w = h.p[0];
  const a = h.p[5];
  const b = h.p[17];
  const ax = a.x - w.x;
  const ay = a.y - w.y;
  const bx = b.x - w.x;
  const by = b.y - w.y;
  const nz = ax * by - ay * bx;
  const scale = Math.max(1e-6, Math.hypot(ax, ay) * Math.hypot(bx, by));
  const s = nz / scale;
  return h.handed === 'Right' ? s : -s;
}

export function nailQuads(h: HandPoints, facingSign = 1): NailQuad[] {
  const facing = backFacing(h) * facingSign;
  // 손등이 약간이라도 보이면(옆으로 누운 손) 서서히 나타나게
  const handVis = Math.min(1, Math.max(0, (facing + 0.1) / 0.4));
  const out: NailQuad[] = [];
  FINGERS.forEach(([tip, dip, mcp], i) => {
    const t = h.p[tip];
    const d = h.p[dip];
    const seg = dist(t, d);
    if (seg < 2) return;
    const dir = norm(sub(t, d));
    const thumb = i === 0;
    // 손가락이 손바닥 쪽으로 접혀 끝이 손목 방향을 향하면 손톱은 반대편을 보고 있어 안 보인다
    let fingerVis = 1;
    if (!thumb) {
      const handDir = norm(sub(h.p[mcp], h.p[0]));
      const cos = dir.x * handDir.x + dir.y * handDir.y;
      fingerVis = Math.min(1, Math.max(0, (cos + 0.1) / 0.4));
    }
    if (fingerVis * handVis < 0.02) return;
    // 손톱 길이: 마지막 마디의 약 절반. 끝점(TIP)은 손가락 끝에 거의 닿아 있으므로
    // 손톱은 마디의 절반 지점 ~ 끝점까지(가운데 = 마디 끝에서 75% 지점). 실제 웹캠 캡처에 손 점을 겹쳐 맞춘 값
    const len = seg * (thumb ? 0.55 : 0.5);
    let c = lerp(d, t, thumb ? 0.72 : 0.75);
    // 손톱 폭: 실제 손톱은 손가락보다 좁다. 엄지 > 검지·중지 > 약지 > 소지
    let width = seg * [0.56, 0.5, 0.5, 0.48, 0.44][i];
    // 손가락 끝이 카메라 쪽으로 향하면(깊이 차이가 크면) 손톱이 짧아 보인다
    const dz = h.z[tip] - h.z[dip];
    const foreshort = Math.max(0.35, 1 - Math.abs(dz) * 6);
    if (thumb) {
      // 손등이 보일 때 엄지는 옆으로 누워 있어 손톱이 비스듬히(좁게) 보이고,
      // 마디 축 위가 아니라 손바닥 반대쪽(다른 손가락에서 먼 쪽)으로 치우쳐 있다
      width *= 0.75;
      const nrm = { x: -dir.y, y: dir.x };
      const away = sub(t, h.p[9]);
      const side = nrm.x * away.x + nrm.y * away.y >= 0 ? 1 : -1;
      c = { x: c.x + nrm.x * side * width * 0.3, y: c.y + nrm.y * side * width * 0.3 };
    }
    const ref = { x: d.x - dir.x * seg * 0.2, y: d.y - dir.y * seg * 0.2 };
    out.push({ c, dir, len: len * foreshort, width, vis: handVis * fingerVis, ref, finger: i });
  });
  return out;
}
