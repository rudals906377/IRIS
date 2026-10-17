/// <reference types="node" />
// 사진 모드 팔 타투 옮기기 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { armStrip, inkMip, measureHalfWidth, stripPoint, transferArmInk, type Arm, type InkMap } from '../src/beauty/still-tattoo.ts';

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

test('몸 분할이 놓친 사진: 관절점 둘레 피부색으로 팔을 잡고 잉크 선 구멍은 메운다', async () => {
  const { poseSkin } = await import('../src/beauty/tattoo-extract.ts');
  const W = 200, H = 200, d = new Uint8ClampedArray(W * H * 4);
  // 배경: 회색, 팔(어깨 (100,40) → 손목 (100,160), 폭 30): 피부색, 가운데 가로 잉크 줄
  for (let i = 0; i < W * H; i++) { d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = 120; d[i * 4 + 3] = 255; }
  for (let y = 40; y < 160; y++) for (let x = 85; x < 115; x++) {
    const o = (y * W + x) * 4;
    const ink = y >= 98 && y < 102;
    d[o] = ink ? 40 : 215; d[o + 1] = ink ? 40 : 170; d[o + 2] = ink ? 45 : 140;
  }
  const pose = Array.from({ length: 33 }, () => ({ x: -100, y: -100 }));
  const vis = new Array(33).fill(0);
  pose[11] = { x: 100, y: 40 }; pose[13] = { x: 100, y: 100 }; pose[15] = { x: 100, y: 160 };
  pose[12] = { x: 40, y: 40 };
  vis[11] = vis[13] = vis[15] = vis[12] = 1;
  const m = poseSkin(d, W, H, pose, vis, []);
  assert.equal(m[70 * W + 100], 1, '팔');
  assert.equal(m[100 * W + 100], 1, '잉크 줄(메움)');
  assert.equal(m[100 * W + 160], 0, '배경');
});

test('몸통 타투: 세 점 아핀 대응과 옮기기(피부에만)', async () => {
  const { affine3, transferTorsoInk } = await import('../src/beauty/still-tattoo.ts');
  // 고른 사진 → 내 사진: 2배 확대 + (10, 20) 이동
  const a = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 0, y: 10 }];
  const b = a.map((p) => ({ x: p.x * 2 + 10, y: p.y * 2 + 20 }));
  const M = affine3(a, b)!;
  assert.ok(Math.abs(M[0] * 5 + M[1] * 7 + M[2] - 20) < 1e-9 && Math.abs(M[3] * 5 + M[4] * 7 + M[5] - 34) < 1e-9);
  assert.equal(affine3([{ x: 0, y: 0 }, { x: 1, y: 1 }, { x: 2, y: 2 }], b), null);
  // 고른 사진 (20, 20) 둘레 잉크 점 → 내 사진 (50, 60)
  const ink: InkMap = { rgba: new Uint8ClampedArray(40 * 40 * 4), w: 40, h: 40, rect: { x: 0, y: 0, w: 40, h: 40 } };
  for (let y = 18; y < 23; y++) for (let x = 18; x < 23; x++) { const o = (y * 40 + x) * 4; ink.rgba[o] = ink.rgba[o + 1] = ink.rgba[o + 2] = 30; ink.rgba[o + 3] = 255; }
  const W = 120, H = 120, img = new Uint8ClampedArray(W * H * 4).fill(200), skin = new Float32Array(W * H).fill(1);
  const out = transferTorsoInk(img, W, H, skin, ink, affine3(b, a)!, M);
  assert.ok(out[(60 * W + 50) * 4] < 100, `잉크 자리 ${out[(60 * W + 50) * 4]}`);
  assert.equal(out[(20 * W + 20) * 4], 200);
});

test('잉크 맵 미리 줄이기: 1px 가는 선이 끊기지 않고 옅게 이어진다(영역은 그대로)', () => {
  const w = 40, h = 40, rgba = new Uint8ClampedArray(w * h * 4).fill(255);
  for (let i = 0; i < w * h; i++) rgba[i * 4 + 3] = 0;
  for (let y = 0; y < h; y++) { const i = (y * w + 17) * 4; rgba.set([20, 20, 20, 255], i); }
  const m = inkMip({ rgba, w, h, rect: { x: 0, y: 0, w: 40, h: 40 } }, 4);
  assert.equal(m.w, 10);
  assert.deepEqual(m.rect, { x: 0, y: 0, w: 40, h: 40 });
  for (let y = 0; y < m.h; y++) {
    const i = (y * m.w + 4) * 4;
    assert.ok(Math.abs(m.rgba[i + 3] - 64) <= 1, `진하기 ${m.rgba[i + 3]}`);
    assert.ok(m.rgba[i] < 40, '잉크 색은 그대로');
  }
});
