/// <reference types="node" />
// 사진 모드 팔 타투 옮기기 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { armStrip, measureHalfWidth, stripPoint, transferArmInk, type Arm, type InkMap } from '../src/beauty/still-tattoo.ts';

test('팔 반폭 재기: 폭 20 인 세로 띠 → 10 근처', () => {
  const W = 100, H = 100, skin = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 40; x < 60; x++) skin[y * W + x] = 1;
  const r = measureHalfWidth(skin, W, H, { x: 50, y: 10 }, { x: 50, y: 90 });
  assert.ok(r !== null && Math.abs(r - 10) <= 1, `반폭 ${r}`);
});

test('띠: 굵은 쪽이 어깨(앞), 가운데·반폭을 따라간다', () => {
  // 아래로 갈수록 가늘어지는 팔(위 반폭 30 → 아래 15), 세로
  const W = 200, H = 400, m = new Float32Array(W * H);
  for (let y = 20; y < 380; y++) {
    const h = 30 - ((y - 20) / 360) * 15;
    for (let x = Math.round(100 - h); x < Math.round(100 + h); x++) m[y * W + x] = 1;
  }
  const st = armStrip(m, W, H, { x: 0, y: 0, w: W, h: H })!;
  assert.ok(st, '띠');
  assert.ok(st.p[0].y < st.p[st.p.length - 1].y, '어깨→손목 방향');
  assert.ok(st.ratio > 1.4, `굵기 비 ${st.ratio}`);
  const top = stripPoint(st, 2, 0)!, mid = stripPoint(st, st.len / 2, 1)!;
  assert.ok(Math.abs(top.y - 22) < 12, `어깨 끝 ${top.y}`);
  assert.ok(Math.abs(Math.abs(mid.x - 100) - 22.5) < 4, `가운데 반폭 ${mid.x}`);
  assert.equal(stripPoint(st, st.len + 5, 0), null);
});

test('띠: ㄱ 자로 굽은 팔도 중심선을 따라간다', () => {
  // 위팔: 세로(x 80~120, y 20~200), 아래팔: 가로(x 80~380, y 160~200)
  const W = 400, H = 240, m = new Float32Array(W * H);
  for (let y = 20; y < 200; y++) for (let x = 80; x < 125; x++) m[y * W + x] = 1;
  for (let y = 165; y < 200; y++) for (let x = 80; x < 380; x++) m[y * W + x] = 1;
  const st = armStrip(m, W, H, { x: 0, y: 0, w: W, h: H })!;
  assert.ok(st.len > 380, `길이 ${st.len}`);
  // 가운데에서 먼 쪽 끝 근처는 아래팔 가운데 줄(y≈182) 위
  const p = stripPoint(st, st.len * 0.85, 0)!;
  const p2 = stripPoint(st, st.len * 0.15, 0)!;
  const onLo = (q: { x: number; y: number }) => Math.abs(q.y - 182) < 8 && q.x > 150;
  const onUp = (q: { x: number; y: number }) => Math.abs(q.x - 102) < 8 && q.y < 150;
  assert.ok((onLo(p) && onUp(p2)) || (onLo(p2) && onUp(p)), `중심선 ${JSON.stringify([p, p2])}`);
});

test('팔 대 팔: 고른 사진 팔 중간의 잉크 줄이 내 팔(기울어짐) 중간으로 간다', () => {
  // 고른 사진: 세로 팔(어깨 (50,10) → 팔꿈치 (50,60) → 손목 (50,110)), 팔꿈치 높이에 가로 잉크 줄
  const ink: InkMap = { rgba: new Uint8ClampedArray(100 * 120 * 4), w: 100, h: 120, rect: { x: 0, y: 0, w: 100, h: 120 } };
  for (let y = 58; y < 63; y++) for (let x = 40; x < 60; x++) { const o = (y * 100 + x) * 4; ink.rgba[o] = ink.rgba[o + 1] = ink.rgba[o + 2] = 40; ink.rgba[o + 3] = 255; }
  const src: Arm = { s: { x: 50, y: 10 }, e: { x: 50, y: 60 }, w: { x: 50, y: 110 }, rUp: 10, rLo: 10 };
  // 내 팔: 가로로 누운 팔, 두 배 크기
  const W = 260, H = 80;
  const dst: Arm = { s: { x: 20, y: 40 }, e: { x: 120, y: 40 }, w: { x: 220, y: 40 }, rUp: 20, rLo: 20 };
  const img = new Uint8ClampedArray(W * H * 4).fill(200);
  const skin = new Float32Array(W * H).fill(1);
  const out = transferArmInk(img, W, H, dst, skin, ink, src);
  assert.ok(out[(40 * W + 120) * 4] < 120, `팔꿈치 ${out[(40 * W + 120) * 4]}`);
  assert.ok(out[(40 * W + 60) * 4] > 190, `위팔 ${out[(40 * W + 60) * 4]}`);
  // 띠 함수로 넘겨도 같은 자리
  const out2 = transferArmInk(img, W, H, dst, skin, ink, (l, s) => ({ x: 50 + s * 10, y: 10 + l / 2 }));
  assert.ok(out2[(40 * W + 120) * 4] < 120, `띠 팔꿈치 ${out2[(40 * W + 120) * 4]}`);
  // 피부가 아닌 곳(옷)은 그대로
  const noSkin = new Float32Array(W * H);
  assert.equal(transferArmInk(img, W, H, dst, noSkin, ink, src)[(40 * W + 120) * 4], 200);
});
