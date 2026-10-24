/// <reference types="node" />
// 사진 모드 네일(손톱 틀·옮겨 그리기) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { nailFrame, splitPolished, transferNail, syntheticNail } from '../src/beauty/still-nail.ts';

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

test('가상 손톱 그림: 손톱 모양 마스크·틀(끝이 위)과 광택', async () => {
  const { syntheticNail } = await import('../src/beauty/still-nail.ts');
  const n = syntheticNail([200, 30, 40]);
  let a = 0;
  for (const v of n.mask) a += v;
  assert.ok(a > n.w * n.h * 0.5, `면적 ${a}`);
  assert.ok(n.frame.ay < -0.9, `끝 방향 ${n.frame.ay}`);
  let maxR = 0;
  for (let i = 0; i < n.mask.length; i++) if (n.mask[i]) maxR = Math.max(maxR, n.rgba[i * 4 + 1]);
  assert.ok(maxR > 100, '광택 띠');
});

test('손톱 대비: 둘레 피부와 같은 색이면 작고, 빨간 매니큐어면 크다', async () => {
  const { maskContrast } = await import('../src/beauty/still-nail.ts');
  const w = 40, h = 40, rgba = new Uint8ClampedArray(w * h * 4), mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) { rgba[i * 4] = 220; rgba[i * 4 + 1] = 180; rgba[i * 4 + 2] = 160; rgba[i * 4 + 3] = 255; }
  for (let y = 15; y < 25; y++) for (let x = 15; x < 25; x++) mask[y * w + x] = 1;
  assert.ok(maskContrast(rgba, w, h, mask) < 1);
  for (let y = 15; y < 25; y++) for (let x = 15; x < 25; x++) { const o = (y * w + x) * 4; rgba[o] = 200; rgba[o + 1] = 20; rgba[o + 2] = 30; }
  assert.ok(maskContrast(rgba, w, h, mask) > 30);
});

test('손톱 마스크 넓히기: 빨간 매니큐어 테두리(마스크 밖 1~2px)까지 넓히고, 피부로는 안 번진다', async () => {
  const { growNailMask } = await import('../src/beauty/still-nail.ts');
  const w = 60, h = 60, rgba = new Uint8ClampedArray(w * h * 4), mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const r = Math.hypot(x - 30, y - 30);
    // 실제 손톱(반지름 12) 빨강, 가장자리 12~13 은 빨강·피부 섞임, 밖은 피부
    rgba.set(r < 12 ? [220, 30, 40, 255] : r < 13 ? [225, 110, 100, 255] : [230, 185, 160, 255], i * 4);
    if (r < 10) mask[i] = 1;
  }
  growNailMask(mask, w, h, rgba, 4);
  assert.equal(mask[30 * w + 30 + 11], 1, '빨간 테두리');
  assert.equal(mask[30 * w + 30 + 12], 1, '섞인 가장자리');
  assert.equal(mask[30 * w + 30 + 15], 0, '피부');
});

test('매니큐어 손톱 나누기: 두 손톱 + 사이 손가락 살을 한 덩어리로 잡으면 손톱 둘로, 손톱만 잡았으면 그대로', () => {
  const w = 80, h = 60, rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) rgba.set([235, 190, 170, 255], i * 4);
  const mask = new Uint8Array(w * h);
  // 빨간 손톱 둘(x 20~29, 45~54) + 그 사이 살(x 30~44)까지 한 마스크
  for (let y = 20; y < 40; y++) for (let x = 20; x < 55; x++) {
    mask[y * w + x] = 1;
    if (x < 30 || x >= 45) rgba.set([210, 30, 40, 255], (y * w + x) * 4);
  }
  const parts = splitPolished(mask, w, h, rgba);
  assert.equal(parts.length, 2);
  for (const p of parts) assert.equal(p[30 * w + 37], 0, '사이 살은 빠짐');
  // 손톱만 잡은 마스크는 그대로(같은 배열)
  const one = new Uint8Array(w * h);
  for (let y = 20; y < 40; y++) for (let x = 20; x < 30; x++) one[y * w + x] = 1;
  const r = splitPolished(one, w, h, rgba);
  assert.equal(r.length, 1);
  assert.equal(r[0], one);
});

test('옮겨 그리기 명암: 내 손톱의 원래 무늬(어두운 크롬 드립)는 새 네일에 비치지 않고, 광택은 남음', () => {
  // 고른 손톱: 고른 연보라
  const SW = 80, SH = 120;
  const sm = ellipse(SW, SH, 40, 60, 40, 15, Math.PI / 2);
  const src = new Uint8ClampedArray(SW * SH * 4);
  for (let i = 0; i < SW * SH; i++) src.set([170, 150, 230, 255], i * 4);
  const sF = nailFrame(sm, SW, SH, { x: 0, y: -1 })!;
  // 내 손톱: 흰 바탕(밝기 200)에 가는 어두운 갈색 드립(x 55~60) + 작은 흰 광택(x 70~72)
  const W = 120, H = 80;
  const dm = ellipse(W, H, 60, 40, 30, 14, 0);
  const dst = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const c = x >= 55 && x <= 60 ? [70, 50, 45] : x >= 70 && x <= 72 && y >= 36 && y <= 44 ? [250, 250, 250] : [205, 198, 200];
    dst.set([...c, 255], (y * W + x) * 4);
  }
  const dF = nailFrame(dm, W, H, { x: 1, y: 0 })!;
  transferNail(dst, W, H, dm, dF, src, SW, SH, sF, 0.35);
  const lum = (x: number, y: number) => { const o = (y * W + x) * 4; return 0.2126 * dst[o] + 0.7152 * dst[o + 1] + 0.0722 * dst[o + 2]; };
  const base = lum(45, 40), drip = lum(57, 40), gloss = lum(71, 40);
  // 전에는 드립 자리가 바탕보다 30% 어두워 원래 무늬가 그대로 비쳤다
  assert.ok(drip > base * 0.93, `드립 ${drip.toFixed(0)} / 바탕 ${base.toFixed(0)}`);
  assert.ok(gloss > base * 1.03, `광택 ${gloss.toFixed(0)} / 바탕 ${base.toFixed(0)}`);
});

test('가상 손톱: 손톱 밖도 가장자리 색이라 끝 가장자리 표본이 검정과 섞이지 않는다', () => {
  const n = syntheticNail([242, 190, 204]);
  let dark = 0;
  for (let i = 0; i < n.w * n.h; i++) if (n.rgba[i * 4] + n.rgba[i * 4 + 1] + n.rgba[i * 4 + 2] < 150) dark++;
  assert.equal(dark, 0, `검은 칸 ${dark}`);
  // 모양은 마스크가 그대로
  assert.equal(n.mask[0], 0);
  assert.equal(n.mask[Math.floor(n.h / 2) * n.w + Math.floor(n.w / 2)], 1);
});
