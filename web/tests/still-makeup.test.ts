/// <reference types="node" />
// 사진 모드 메이크업 옮기기 단위 테스트(가짜 얼굴 점)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { transferMakeup } from '../src/beauty/still-makeup.ts';
import { FACE_OVAL } from '../src/beauty/face-measure.ts';
import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_INNER, LIPS_OUTER } from '../src/beauty/face-regions.ts';

/** 가운데 (cx, cy), 반지름 R 인 둥근 가짜 얼굴의 점 468개 */
function face(cx: number, cy: number, R: number) {
  const p: { x: number; y: number }[] = [];
  // 나머지 점: 얼굴 안 격자(살짝 흔들어 같은 점이 없게)
  for (let i = 0; i < 468; i++) {
    const gx = (i % 22) / 21 - 0.5, gy = Math.floor(i / 22) / 21 - 0.5;
    p.push({ x: cx + gx * R * 1.2 + (i % 7) * 0.01, y: cy + gy * R * 1.2 + (i % 5) * 0.01 });
  }
  const ring = (ids: number[], ox: number, oy: number, rx: number, ry: number) =>
    ids.forEach((k, j) => { const t = (j / ids.length) * Math.PI * 2; p[k] = { x: cx + ox * R + rx * R * Math.cos(t), y: cy + oy * R + ry * R * Math.sin(t) }; });
  ring(FACE_OVAL, 0, 0, 1, 1.2);
  ring(LIPS_OUTER, 0, 0.6, 0.3, 0.1);
  ring(LIPS_INNER, 0, 0.6, 0.18, 0.02);
  ring([...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], -0.4, -0.2, 0.15, 0.06);
  ring([...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)], 0.4, -0.2, 0.15, 0.06);
  ring([...BROW_R_LOWER, ...BROW_R_UPPER], -0.4, -0.45, 0.2, 0.04);
  ring([...BROW_L_LOWER, ...BROW_L_UPPER], 0.4, -0.45, 0.2, 0.04);
  p[234] = { x: cx - R, y: cy };
  p[454] = { x: cx + R, y: cy };
  return p;
}

test('메이크업: 고른 사진 볼의 블러셔(붉은 기)가 내 볼로, 눈 안과 얼굴 밖은 그대로', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const skin = [200, 170, 150];
  const my = new Uint8ClampedArray(W * H * 4);
  const ref = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) { my[i * 4 + c] = skin[c]; ref[i * 4 + c] = skin[c]; }
    my[i * 4 + 3] = ref[i * 4 + 3] = 255;
  }
  // 고른 사진: 오른볼(이미지 왼쪽 아래)에 블러셔
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if ((x - 70) ** 2 + (y - 135) ** 2 < 15 ** 2) { const o = (y * W + x) * 4; ref[o] = 225; ref[o + 1] = 140; ref[o + 2] = 140; }
  }
  // 내 눈 안: 검은 눈동자
  const eyeO = (y: number, x: number) => (y * W + x) * 4;
  for (let y = 97; y < 100; y++) for (let x = 73; x < 79; x++) { const o = eyeO(y, x); my[o] = my[o + 1] = my[o + 2] = 30; }
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  const cheek = (70 + 135 * W) * 4;
  assert.ok(out[cheek] - out[cheek + 1] > my[cheek] - my[cheek + 1] + 15, `볼 붉은 기 ${out[cheek] - out[cheek + 1]}`);
  assert.equal(out[eyeO(98, 76)], 30, '눈 안 그대로');
  assert.equal(out[eyeO(5, 5)], my[eyeO(5, 5)], '얼굴 밖 그대로');
  // 반대쪽 볼(블러셔 없음)은 거의 그대로
  const other = (130 + 135 * W) * 4;
  assert.ok(Math.abs(out[other + 1] - my[other + 1]) < 8, `반대 볼 ${out[other + 1]}`);
});

test('주근깨 찾기: 둥근 어두운 점은 잡고, 가는 선(머리카락·손가락 가장자리)은 안 잡는다', async () => {
  const { dotMask } = await import('../src/beauty/still-makeup.ts');
  const w = 80, h = 40, L = new Float32Array(w * h).fill(70);
  // 왼쪽: 지름 4px 점, 오른쪽: 세로로 긴 선(폭 2px)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if ((x - 20) ** 2 + (y - 20) ** 2 <= 4) L[y * w + x] = 50;
    if (x >= 59 && x <= 60) L[y * w + x] = 50;
  }
  const m = dotMask(L, w, h);
  assert.ok(m[20 * w + 20] > 0.5, `점 ${m[20 * w + 20]}`);
  assert.ok(m[20 * w + 60] < 0.2, `선 ${m[20 * w + 60]}`);
});
