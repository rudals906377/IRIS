// 2D 벡터 유틸. 매 프레임 수백 번 쓰이므로 단순한 객체와 함수로 둔다.

export interface Vec2 {
  x: number;
  y: number;
}

export const v = (x: number, y: number): Vec2 => ({ x, y });
export const add = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x + b.x, y: a.y + b.y });
export const sub = (a: Vec2, b: Vec2): Vec2 => ({ x: a.x - b.x, y: a.y - b.y });
export const scale = (a: Vec2, s: number): Vec2 => ({ x: a.x * s, y: a.y * s });
export const dot = (a: Vec2, b: Vec2): number => a.x * b.x + a.y * b.y;
export const cross = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;
export const len = (a: Vec2): number => Math.hypot(a.x, a.y);
export const dist = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);
export const mid = (a: Vec2, b: Vec2): Vec2 => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
export const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
/** 이미지 좌표계(y 아래)에서 시계 방향 90° 회전. (1,0) → (0,1) */
export const perp = (a: Vec2): Vec2 => ({ x: -a.y, y: a.x });

export function norm(a: Vec2, fallback: Vec2 = { x: 1, y: 0 }): Vec2 {
  const l = Math.hypot(a.x, a.y);
  return l > 1e-9 ? { x: a.x / l, y: a.y / l } : fallback;
}

export const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
export const clamp01 = (x: number): number => clamp(x, 0, 1);
export const mix = (a: number, b: number, t: number): number => a + (b - a) * t;

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
}
