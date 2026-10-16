/// <reference types="node" />
// 사진 모드 네일(손톱 틀·옮겨 그리기) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nailFrame, transferNail } from '../src/beauty/still-nail.ts';

/** 중심(cx,cy), 반지름(a: 길이 방향, b: 폭), 각도 th 의 타원 마스크 */
function ellipse(w: number, h: number, cx: number, cy: number, a: number, b: number, th: number): Uint8Array {
  const m = new Uint8Array(w * h);
  const c = Math.cos(th), s = Math.sin(th);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const dx = x - cx, dy = y - cy;
    const u = (dx * c + dy * s) / a, v = (-dx * s + dy * c) / b;
    if (u * u + v * v <= 1) m[y * w + x] = 1;
  }
  return m;
}

test('손톱 틀: 주축과 길이, 방향 힌트대로 뿌리 → 끝', () => {
  const m = ellipse(100, 100, 50, 50, 30, 12, 0); // 가로로 긴 손톱
  const f = nailFrame(m, 100, 100, { x: -1, y: 0 })!; // 끝은 왼쪽
  assert.ok(f.ax < -0.99, `ax ${f.ax}`);
  assert.ok(Math.abs(f.t1 - f.t0 - 60) < 3);
  const g = nailFrame(m, 100, 100, { x: 1, y: 0 })!;
  assert.ok(g.ax > 0.99);
});

test('옮겨 그리기: 고른 손톱의 뿌리 색은 내 뿌리로, 끝 색은 내 끝으로(모양·방향이 달라도)', () => {
  // 고른 사진: 세로 손톱(끝이 위), 아래(뿌리)는 빨강, 위(끝)는 파랑
  const SW = 80, SH = 120;
  const sm = ellipse(SW, SH, 40, 60, 40, 15, Math.PI / 2);
  const src = new Uint8ClampedArray(SW * SH * 4);
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
    const o = (y * SW + x) * 4;
    const tip = y < 60;
    src[o] = tip ? 0 : 255; src[o + 2] = tip ? 255 : 0; src[o + 3] = 255;
  }
  const sF = nailFrame(sm, SW, SH, { x: 0, y: -1 })!;
  // 내 사진: 가로로 누운 짧은 손톱(끝이 오른쪽), 회색
  const W = 120, H = 80;
  const dm = ellipse(W, H, 60, 40, 25, 14, 0);
  const dst = new Uint8ClampedArray(W * H * 4).fill(128);
  const dF = nailFrame(dm, W, H, { x: 1, y: 0 })!;
  transferNail(dst, W, H, dm, dF, src, SW, SH, sF, 0);
  const px = (x: number, y: number) => { const o = (y * W + x) * 4; return [dst[o], dst[o + 1], dst[o + 2]]; };
  const root = px(42, 40), tip = px(78, 40), out = px(5, 5);
  assert.ok(root[0] > 200 && root[2] < 60, `뿌리 ${root}`);
  assert.ok(tip[2] > 200 && tip[0] < 60, `끝 ${tip}`);
  assert.deepEqual(out, [128, 128, 128]);
});

test('방향 힌트가 없으면 피부색 쪽을 뿌리로 판단', () => {
  // 피부색 손가락이 아래에서 올라와 손톱(빨강)으로 이어지고, 위는 흰 배경
  const W = 80, H = 120;
  const m = ellipse(W, H, 40, 50, 25, 12, Math.PI / 2);
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    const finger = Math.abs(x - 40) < 26 && y > 40; // 손가락은 손톱보다 넓다
    const c = m[y * W + x] ? [200, 20, 30] : finger ? [225, 180, 160] : [250, 250, 250];
    rgba[o] = c[0]; rgba[o + 1] = c[1]; rgba[o + 2] = c[2]; rgba[o + 3] = 255;
  }
  // 손가락이 손톱 아래쪽으로 이어지므로 끝은 위(ay < 0)
  const f = nailFrame(m, W, H, null, rgba)!;
  assert.ok(f.ay < -0.9, `ay ${f.ay}`);
});

test('볼록 껍질 채우기: 반사광 구멍과 오목한 들쭉날쭉을 메운다', async () => {
  const { hullFill } = await import('../src/beauty/still-nail.ts');
  const W = 60, H = 60;
  const m = ellipse(W, H, 30, 30, 20, 12, 0.3);
  m[30 * W + 30] = 0; m[30 * W + 31] = 0; // 구멍
  for (let y = 18; y < 24; y++) for (let x = 38; x < 46; x++) m[y * W + x] = 0; // 가장자리 파임
  const before = m.reduce((s, v) => s + v, 0);
  hullFill(m, W, H);
  assert.equal(m[30 * W + 30], 1);
  assert.ok(m.reduce((s, v) => s + v, 0) > before);
  assert.equal(m[2 * W + 2], 0);
});
