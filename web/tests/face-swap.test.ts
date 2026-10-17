/// <reference types="node" />
// 사진 속에 내 얼굴(삼각분할·휘기·섞기) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { delaunay, fillPoly, swapFace, warpTriangles } from '../src/beauty/face-swap.ts';

test('들로네: 격자 점 9개 → 삼각형 8개, 모든 점 사용', () => {
  const pts = [];
  for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) pts.push({ x: x * 10 + (y % 2) * 0.1, y: y * 10 });
  const t = delaunay(pts);
  assert.equal(t.length / 3, 8);
  assert.equal(new Set(t).size, 9);
});

test('다각형 채우기: 사각형 안만 1', () => {
  const m = fillPoly([{ x: 2, y: 2 }, { x: 8, y: 2 }, { x: 8, y: 8 }, { x: 2, y: 8 }], 10, 10);
  assert.equal(m[5 * 10 + 5], 1);
  assert.equal(m[0], 0);
});

test('얼굴 바꾸기: 윤곽 안은 내 얼굴 무늬(조명은 고른 사진), 밖은 그대로', () => {
  // 점 468개(가짜): 원 위 36개(윤곽) + 안쪽 격자. 두 사진은 크기·위치만 다름
  const mk = (cx: number, cy: number, r: number) => {
    const p = [];
    for (let i = 0; i < 468; i++) {
      if (i < 36) p.push({ x: cx + r * Math.cos((i / 36) * Math.PI * 2), y: cy + r * Math.sin((i / 36) * Math.PI * 2) });
      else { const k = i - 36; p.push({ x: cx + ((k % 21) / 20 - 0.5) * r, y: cy + ((Math.floor(k / 21) % 21) / 20 - 0.5) * r }); }
    }
    return p;
  };
  const oval = Array.from({ length: 36 }, (_, i) => i);
  // 점 234·454(얼굴 폭 기준)를 윤곽 위 점으로 바꿔 둔다
  const W = 120, H = 120, SW = 80, SH = 80;
  const dp = mk(60, 60, 40), sp = mk(40, 40, 30);
  dp[234] = dp[18]; dp[454] = dp[0]; sp[234] = sp[18]; sp[454] = sp[0];
  const dst = new Uint8ClampedArray(W * H * 4).fill(100);
  const src = new Uint8ClampedArray(SW * SH * 4);
  // 내 얼굴: 가는 줄무늬(눈썹·속눈썹 같은 작은 무늬) + 전체적으로 밝음(조명) → 무늬는 남고 밝기는 고른 사진(100)을 따른다
  for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) { const o = (y * SW + x) * 4; const v = (Math.floor(x / 2) % 2) ? 230 : 150; src[o] = src[o + 1] = src[o + 2] = v; src[o + 3] = 255; }
  const out = swapFace(dst, W, H, dp, src, SW, SH, sp, oval);
  const g = (x: number, y: number) => out[(y * W + x) * 4];
  let mn = 255, mx = 0, sum = 0;
  for (let x = 50; x < 70; x++) { const v = g(x, 60); mn = Math.min(mn, v); mx = Math.max(mx, v); sum += v; }
  assert.ok(mx - mn > 25, `무늬 대비 ${mx - mn}`);
  assert.ok(Math.abs(sum / 20 - 100) < 25, `평균 밝기 ${sum / 20}`);
  assert.equal(g(2, 2), 100);
});

test('얼굴 바꾸기: 내 사진에서 피부가 아닌 곳(앞머리)은 옮기지 않는다', () => {
  const mk = (cx: number, cy: number, r: number) => {
    const p = [];
    for (let i = 0; i < 468; i++) {
      if (i < 36) p.push({ x: cx + r * Math.cos((i / 36) * Math.PI * 2), y: cy + r * Math.sin((i / 36) * Math.PI * 2) });
      else { const k = i - 36; p.push({ x: cx + ((k % 21) / 20 - 0.5) * r, y: cy + ((Math.floor(k / 21) % 21) / 20 - 0.5) * r }); }
    }
    return p;
  };
  const oval = Array.from({ length: 36 }, (_, i) => i);
  const W = 120, H = 120;
  const dp = mk(60, 60, 40), sp = mk(60, 60, 40);
  dp[234] = dp[18]; dp[454] = dp[0]; sp[234] = sp[18]; sp[454] = sp[0];
  const dst = new Uint8ClampedArray(W * H * 4).fill(180);
  const src = new Uint8ClampedArray(W * H * 4).fill(170);
  const srcSkin = new Float32Array(W * H).fill(1);
  // 내 사진 위쪽(이마)은 검은 앞머리
  for (let y = 0; y < 50; y++) for (let x = 0; x < W; x++) { const o = (y * W + x) * 4; src[o] = src[o + 1] = src[o + 2] = 20; srcSkin[y * W + x] = 0; }
  const out = swapFace(dst, W, H, dp, src, W, H, sp, oval, null, null, srcSkin);
  assert.ok(out[(35 * W + 60) * 4] > 150, `이마 ${out[(35 * W + 60) * 4]}`);
  // 피부끼리 조명 비를 맞추므로 볼은 고른 사진 밝기 근처(앞머리 때문에 하얗게 뜨지 않음)
  assert.ok(Math.abs(out[(75 * W + 60) * 4] - 180) < 12, `볼 ${out[(75 * W + 60) * 4]}`);
});

test('휘기: 내 쪽 삼각형이 뒤집히면(고개 돌린 사진에서 가려진 자리) fold 로 표시', () => {
  const dp = [{ x: 10, y: 10 }, { x: 50, y: 10 }, { x: 10, y: 50 }, { x: 50, y: 50 }];
  const src = new Uint8ClampedArray(60 * 60 * 4).fill(100);
  const same = warpTriangles(dp, 60, 60, src, 60, 60, dp);
  const mirror = warpTriangles(dp, 60, 60, src, 60, 60, dp.map((p) => ({ x: 60 - p.x, y: p.y })));
  const i = 30 * 60 + 30;
  assert.equal(same.cover[i], 1);
  assert.equal(same.fold[i], 0);
  assert.equal(mirror.fold[i], 1);
});

test('얼굴 바꾸기: 어두운 피부 → 밝은 피부 사진이어도 명암 대비가 크게 부풀거나 밝은 곳이 하얗게 날아가지 않는다', () => {
  const mk = (cx: number, cy: number, r: number) => {
    const p = [];
    for (let i = 0; i < 468; i++) {
      if (i < 36) p.push({ x: cx + r * Math.cos((i / 36) * Math.PI * 2), y: cy + r * Math.sin((i / 36) * Math.PI * 2) });
      else { const k = i - 36; p.push({ x: cx + ((k % 21) / 20 - 0.5) * r, y: cy + ((Math.floor(k / 21) % 21) / 20 - 0.5) * r }); }
    }
    return p;
  };
  const oval = Array.from({ length: 36 }, (_, i) => i);
  const W = 120, H = 120;
  const dp = mk(60, 60, 40), sp = mk(60, 60, 40);
  dp[234] = dp[18]; dp[454] = dp[0]; sp[234] = sp[18]; sp[454] = sp[0];
  // 고른 사진: 밝은 피부(215) / 내 얼굴: 어두운 피부(70, 가는 줄무늬 ±12) + 이마에 밝은 반사(140)
  const dst = new Uint8ClampedArray(W * H * 4).fill(215);
  const src = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4;
    const v = (x - 60) ** 2 + (y - 40) ** 2 < 16 ? 140 : (Math.floor(x / 2) % 2 ? 82 : 58);
    src[o] = v; src[o + 1] = Math.round(v * 0.8); src[o + 2] = Math.round(v * 0.65); src[o + 3] = 255;
  }
  const out = swapFace(dst, W, H, dp, src, W, H, sp, oval);
  const lum = (x: number, y: number) => { const o = (y * W + x) * 4; return 0.299 * out[o] + 0.587 * out[o + 1] + 0.114 * out[o + 2]; };
  let mn = 255, mx = 0;
  for (let x = 50; x < 70; x++) { const v = lum(x, 70); mn = Math.min(mn, v); mx = Math.max(mx, v); }
  // 원래 줄무늬 대비(루마 ≈ 20)가 3배(비를 곱하던 때 ≈ 60)로 부풀지 않는다
  assert.ok(mx - mn < 36, `무늬 대비 ${(mx - mn).toFixed(0)}`);
  assert.ok(mx - mn > 10, `무늬는 남는다 ${(mx - mn).toFixed(0)}`);
  // 반사 자리가 하얗게 날아가지 않는다
  const o = (40 * W + 60) * 4;
  assert.ok(Math.min(out[o], out[o + 1], out[o + 2]) < 250, `반사 ${out[o]},${out[o + 1]},${out[o + 2]}`);
});
