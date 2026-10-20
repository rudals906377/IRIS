/// <reference types="node" />
// 사진 모드 염색(분위 대응 색 옮기기·가이디드 필터) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Q, colorGate, fillInterior, flyaways, guidedFilter, hairDist, pureHair, skinGate, quantilePos, recolorHair, strandTexture, earGate, edgeAlpha, softenLiftDetail, rgb2lab } from '../src/beauty/still-hair.ts';

// w×h 그림: 왼쪽 반은 머리카락(밝기 줄무늬), 오른쪽 반은 배경
function picture(w: number, h: number, base: [number, number, number], stripe: number): { rgba: Uint8ClampedArray; alpha: Float32Array } {
  const rgba = new Uint8ClampedArray(w * h * 4);
  const alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const hair = x < w / 2;
      const s = hair ? ((x % 4) < 2 ? stripe : -stripe) : 0;
      const c = hair ? base : [240, 240, 240];
      rgba[i * 4] = c[0] + s;
      rgba[i * 4 + 1] = c[1] + s;
      rgba[i * 4 + 2] = c[2] + s;
      rgba[i * 4 + 3] = 255;
      alpha[i] = hair ? 1 : 0;
    }
  }
  return { rgba, alpha };
}

const meanL = (d: { L: Float32Array }): number => d.L.reduce((s, v) => s + v, 0) / d.L.length;

test('분위표: 밝기 순으로 정렬되고 분위 위치를 되찾는다', () => {
  const { rgba, alpha } = picture(40, 40, [200, 170, 110], 30);
  const d = hairDist(rgba, alpha)!;
  for (let k = 1; k <= Q; k++) assert.ok(d.L[k] >= d.L[k - 1]);
  assert.equal(quantilePos(d.L, -1), 0);
  assert.equal(quantilePos(d.L, 1000), Q);
});

test('금발 → 흑발: 머리카락만 고른 사진 밝기로, 배경은 그대로, 결(줄무늬)은 남는다', () => {
  const W = 40, H = 40;
  const mine = picture(W, H, [200, 170, 110], 30);
  const ref = picture(W, H, [30, 25, 22], 6);
  const dm = hairDist(mine.rgba, mine.alpha)!;
  const dr = hairDist(ref.rgba, ref.alpha)!;
  const out = recolorHair(mine.rgba, W, H, mine.alpha, dm, dr, 1);
  const outD = hairDist(out, mine.alpha)!;
  assert.ok(Math.abs(meanL(outD) - meanL(dr)) < 4, `평균 밝기 ${meanL(outD).toFixed(1)} vs 참고 ${meanL(dr).toFixed(1)}`);
  // 배경 픽셀은 그대로
  const bg = (10 * W + 35) * 4;
  assert.deepEqual([out[bg], out[bg + 1], out[bg + 2]], [240, 240, 240]);
  // 줄무늬의 밝고 어두움 순서 유지
  const p0 = (10 * W + 0) * 4, p2 = (10 * W + 2) * 4;
  assert.ok(out[p0] > out[p2], '밝은 올이 어두운 올보다 밝아야 함');
});

test('진하기 0 이면 원본 그대로', () => {
  const W = 20, H = 20;
  const mine = picture(W, H, [200, 170, 110], 30);
  const ref = picture(W, H, [30, 25, 22], 6);
  const out = recolorHair(mine.rgba, W, H, mine.alpha, hairDist(mine.rgba, mine.alpha, 0.6, 50)!, hairDist(ref.rgba, ref.alpha, 0.6, 50)!, 0);
  assert.deepEqual(Array.from(out), Array.from(mine.rgba));
});

test('가이디드 필터: 거친 확률이 안내 영상의 경계에 붙는다', () => {
  const W = 60, H = 20;
  const guide = new Float32Array(W * H), p = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      guide[y * W + x] = x < 30 ? 0.1 : 0.9; // 진짜 경계는 30
      p[y * W + x] = x < 26 ? 1 : x < 34 ? 0.5 : 0; // 흐릿한 확률
    }
  }
  const q = guidedFilter(guide, p, W, H, 4, 1e-3);
  // 원래 확률은 28·32 모두 0.5(구분 없음) → 필터 뒤에는 경계 안쪽이 확실히 높고 바깥이 낮다
  assert.ok(q[10 * W + 28] > 0.6, `경계 안쪽 ${q[10 * W + 28]}`);
  assert.ok(q[10 * W + 32] < 0.4, `경계 바깥 ${q[10 * W + 32]}`);
  assert.ok(q[10 * W + 28] - q[10 * W + 32] > 0.3);
});

test('구멍 메우기: 작은 구멍은 메우고 바깥 경계는 그대로', async () => {
  const { closeHoles } = await import('../src/beauty/still-hair.ts');
  const W = 40, H = 40;
  const a = new Float32Array(W * H);
  for (let y = 5; y < 35; y++) for (let x = 5; x < 35; x++) a[y * W + x] = 1;
  a[20 * W + 20] = 0; a[20 * W + 21] = 0; // 작은 구멍
  const c = closeHoles(a, W, H, 3);
  assert.equal(c[20 * W + 20], 1);
  assert.equal(c[2 * W + 2], 0);
  assert.equal(c[20 * W + 5], 1);
});

test('분위 위치: 같은 값이 여러 분위면 그 가운데', () => {
  const qs = new Float32Array(Q + 1);
  for (let k = 0; k <= Q; k++) qs[k] = k < 10 ? 0 : k; // 0~9 분위가 모두 0(눌린 검정)
  assert.equal(quantilePos(qs, 0), 4.5);
  assert.ok(Math.abs(quantilePos(qs, 20) - 20) < 1e-6);
});

test('색 거르기: 가장자리의 머리색과 다른 픽셀(초록 나뭇잎)은 빼고, 확실한 머리카락은 그대로', async () => {
  const { colorGate } = await import('../src/beauty/still-hair.ts');
  const n = 400;
  const rgba = new Uint8ClampedArray(n * 4);
  const alpha = new Float32Array(n).fill(1);
  const coarse = new Float32Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const leaf = i >= 300;
    rgba[o] = leaf ? 20 : 35 + (i % 7); rgba[o + 1] = leaf ? 60 : 30 + (i % 5); rgba[o + 2] = leaf ? 25 : 28 + (i % 3); rgba[o + 3] = 255;
    if (i >= 250) coarse[i] = 0.5; // 가장자리
  }
  colorGate(rgba, alpha, coarse);
  assert.equal(alpha[10], 1);
  assert.ok(alpha[260] > 0.8, `가장자리 머리 ${alpha[260]}`);
  assert.ok(alpha[350] < 0.2, `나뭇잎 ${alpha[350]}`);
});

test('색 거르기: 피부와 섞인 이마 잔머리(붉은 쪽)는 남긴다', async () => {
  const { colorGate } = await import('../src/beauty/still-hair.ts');
  const n = 300;
  const rgba = new Uint8ClampedArray(n * 4);
  const alpha = new Float32Array(n).fill(1);
  const coarse = new Float32Array(n).fill(1);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    const mix = i >= 250; // 피부와 반쯤 섞인 잔머리
    rgba[o] = mix ? 140 : 35 + (i % 7); rgba[o + 1] = mix ? 100 : 30 + (i % 5); rgba[o + 2] = mix ? 85 : 28 + (i % 3); rgba[o + 3] = 255;
    if (i >= 250) coarse[i] = 0.5;
  }
  colorGate(rgba, alpha, coarse);
  assert.ok(alpha[280] > 0.9, `잔머리 ${alpha[280]}`);
});

test('투톤: 위는 주황·아래는 노랑인 머리 → 띠 분포가 다르고 옴브레로 본다', async () => {
  const { hairBands } = await import('../src/beauty/still-hair.ts');
  const w = 40, h = 90, rgba = new Uint8ClampedArray(w * h * 4), alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const top = y < h / 2;
    rgba[i * 4] = top ? 230 : 240; rgba[i * 4 + 1] = top ? 90 : 220; rgba[i * 4 + 2] = top ? 20 : 30; rgba[i * 4 + 3] = 255;
    alpha[i] = 1;
  }
  const b = hairBands(rgba, w, alpha, 0.6, 100)!;
  assert.ok(b.ombre, '옴브레');
  assert.ok(b.bands[0].b[32] < b.bands[2].b[32] || b.bands[0].a[32] > b.bands[2].a[32] + 10, '뿌리 띠가 더 붉다');
});

test('어두운 꼬리 다듬기: 검은 옷이 섞인 아래 몇 %는 10% 자리 근처로 올린다', async () => {
  const { trimDarkTail, Q } = await import('../src/beauty/still-hair.ts');
  const L = new Float32Array(Q + 1), a = new Float32Array(Q + 1), b = new Float32Array(Q + 1);
  for (let k = 0; k <= Q; k++) L[k] = k < 4 ? 5 : 50 + (k / Q) * 30;
  const t = trimDarkTail({ L, a, b, n: 1000 });
  assert.ok(t.L[0] > 40, `맨 아래 ${t.L[0]}`);
  assert.equal(t.L[Q], L[Q]);
});

test('사진 염색: 새까만 점이 섞인 어두운 머리를 밝게 바꿔도 검은 점이 남지 않는다', async () => {
  const { recolorHair, hairDist } = await import('../src/beauty/still-hair.ts');
  const w = 60, h = 60, rgba = new Uint8ClampedArray(w * h * 4), alpha = new Float32Array(w * h).fill(1);
  // 어두운 머리(루마 ~25) 사이사이에 새까만(0) 점과 루마 10 점
  for (let i = 0; i < w * h; i++) {
    const v = i % 7 === 0 ? 0 : i % 11 === 0 ? 10 : 25 + (i % 5);
    rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = v; rgba[i * 4 + 3] = 255;
  }
  const mine = hairDist(rgba, alpha, 0.6, 100)!;
  const ref = { ...hairDist(rgba.map((_, i) => (i % 4 === 3 ? 255 : 200 + (Math.floor(i / 4) % 20))), alpha, 0.6, 100)! };
  const out = recolorHair(rgba, w, h, alpha, mine, ref, 1);
  let mn = 255;
  for (let y = 5; y < h - 5; y++) for (let x = 5; x < w - 5; x++) mn = Math.min(mn, out[(y * w + x) * 4]);
  assert.ok(mn > 140, `가장 어두운 값 ${mn}`);
});

test('잔머리: 머리 밖 가는 어두운 가닥은 알파를 얻고, 가닥 사이 벽은 알파를 잃는다', () => {
  // 80×60: 왼쪽 30칸 검은 머리(알파 1), 그 오른쪽 밝은 벽. x=34 에 1px 가닥(분할 0), x=26~27 은 가닥 사이로 비친 벽(알파 1):
  // 가닥 사이 틈(6행마다 x=28~29 가 배경)으로 바깥 벽과 이어져 있다(머리에 둘러싸인 밝은 윤기 가닥은 벽이 아니라 남긴다)
  const w = 80, h = 60;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let v = 220;
    if (x < 30) { v = 30; alpha[i] = 1; }
    if (x === 26 || x === 27) v = 218;
    if ((x === 28 || x === 29) && y % 6 === 0) { v = 220; alpha[i] = 0; }
    if (x === 34) v = 60;
    if (x === 70) v = 60; // 머리에서 먼 벽의 줄(전선 등): 건드리지 않는다
    rgba.set([v, v, v, 255], i * 4);
  }
  flyaways(rgba, w, h, alpha, 12);
  const y = 30;
  assert.ok(alpha[y * w + 34] > 0.6, `가닥 ${alpha[y * w + 34]}`);
  assert.ok(alpha[y * w + 26] < 0.2, `비친 벽 ${alpha[y * w + 26]}`);
  assert.ok(alpha[y * w + 10] > 0.99, `머리 안 ${alpha[y * w + 10]}`);
  assert.equal(alpha[y * w + 70], 0);
  assert.equal(alpha[y * w + 50], 0);
});

test('검은 머리 정수리 윤기: 옴브레·하이라이트로 읽지 않는다', async () => {
  const { hairBands } = await import('../src/beauty/still-hair.ts');
  // 검은 머리, 위 1/3 의 40% 는 밝은 회보라 윤기
  const w = 40, h = 90, rgba = new Uint8ClampedArray(w * h * 4), alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const shine = y < h / 3 && x % 5 < 2;
    rgba.set(shine ? [150, 145, 165, 255] : [12, 12, 16, 255], i * 4);
    alpha[i] = 1;
  }
  const b = hairBands(rgba, w, alpha, 0.6, 100)!;
  assert.ok(!b.ombre, '옴브레 아님');
  assert.ok(b.lowSpread < 20, `가운데·끝 띠 폭 ${b.lowSpread}`);
});

test('머리색 거르기: 시스루 앞머리 사이 이마(피부색)는 고른 사진 머리 분포에서 뺀다', () => {
  const n = 1000, rgba = new Uint8ClampedArray(n * 4), hair = new Float32Array(n), face = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    rgba.set(i < 400 ? [15, 15, 18, 255] : [230, 190, 170, 255], i * 4);
    hair[i] = i < 500 ? 1 : 0;
    face[i] = i >= 500 ? 1 : 0;
  }
  const out = pureHair(rgba, hair, [face]);
  assert.equal(out.slice(0, 400).reduce((a, b) => a + b, 0), 400);
  assert.equal(out.slice(400, 500).reduce((a, b) => a + b, 0), 0);
  // 금발처럼 머리와 피부색이 비슷하면 빼지 않는다
  for (let i = 0; i < 400; i++) rgba.set([225, 185, 160, 255], i * 4);
  assert.equal(pureHair(rgba, hair, [face]).slice(0, 500).reduce((a, b) => a + b, 0), 500);
});

test('피부 거르기: 검은 머리 알파 안의 피부색 픽셀(앞머리 사이 이마)은 알파를 줄이고, 머리·금발은 그대로', () => {
  const n = 900, rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n), face = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // 0~399 검은 머리, 400~499 머리 알파 안 이마(피부색), 500~ 얼굴 피부
    rgba.set(i < 400 ? [25, 20, 18, 255] : [225, 180, 160, 255], i * 4);
    alpha[i] = i < 500 ? 1 : 0;
    face[i] = i >= 500 ? 1 : 0;
  }
  skinGate(rgba, alpha, face);
  assert.equal(alpha[10], 1);
  assert.ok(alpha[450] < 0.1, `이마 ${alpha[450]}`);
  // 금발(피부와 비슷)이면 손대지 않는다
  for (let i = 0; i < 400; i++) rgba.set([215, 185, 140, 255], i * 4);
  const a2 = new Float32Array(n);
  for (let i = 0; i < 500; i++) a2[i] = 1;
  skinGate(rgba, a2, face);
  assert.equal(a2[450], 1);
});

test('피부 거르기: 흰 배경 바로 곁 머리 가장자리의 밝은 금빛 가닥은 피부로 보고 깎지 않는다(그림 폭을 줄 때), 머리 속 두피색은 깎는다', () => {
  const w = 30, h = 30, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n), face = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // 왼쪽 0~7 흰 배경, 8~9 밝은 금빛 가장자리, 10~21 갈색 머리(가운데 15~16 두피색), 22~ 얼굴 피부
    const col = x < 8 ? [230, 230, 228] : x < 10 ? [215, 180, 150] : x < 22 ? (x === 15 || x === 16 ? [215, 180, 160] : [110, 70, 40]) : [225, 180, 160];
    rgba.set([...col, 255], i * 4);
    alpha[i] = x >= 8 && x < 22 ? (x < 10 ? 0.84 : 1) : 0;
    face[i] = x >= 22 ? 1 : 0;
  }
  const a1 = new Float32Array(alpha);
  skinGate(rgba, a1, face, w);
  assert.ok(a1[15 * w + 9] > 0.8, `가장자리 ${a1[15 * w + 9]}`);
  assert.ok(a1[15 * w + 15] < 0.3, `두피 ${a1[15 * w + 15]}`);
  // 폭을 안 주면 예전처럼 가장자리도 깎는다
  const a2 = new Float32Array(alpha);
  skinGate(rgba, a2, face);
  assert.ok(a2[15 * w + 9] < 0.75, `폭 없음 가장자리 ${a2[15 * w + 9]}`);
});

test('머리 색 거르기: 가닥 사이 어두운 그늘(색이 옅음)은 남기고, 초록·푸른 배경은 뺀다', () => {
  const w = 40, h = 40, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n).fill(1), coarse = new Float32Array(n).fill(1);
  // 확실한 갈색 머리
  for (let i = 0; i < n; i++) rgba.set([120, 80, 50, 255], i * 4);
  // 가장자리 칸(분할이 덜 확실): 어두운 그늘, 초록 배경
  const shade = 5 * w + 5, leaf = 5 * w + 30;
  rgba.set([28, 22, 19, 255], shade * 4);
  rgba.set([70, 110, 60, 255], leaf * 4);
  coarse[shade] = coarse[leaf] = 0.5;
  colorGate(rgba, alpha, coarse);
  assert.ok(alpha[shade] > 0.9, `그늘 ${alpha[shade]}`);
  assert.ok(alpha[leaf] < 0.1, `초록 ${alpha[leaf]}`);
});

test('검은 머리 밝게 염색: 배경에 닿은 가장자리(알파 높아도 배경이 섞여 밝음)는 가장 밝은 머리색이 되지 않는다', () => {
  const w = 60, h = 40, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // 검은 머리(루마 26~34, 약한 결) | 마지막 한 줄은 배경이 섞여 루마 60(알파는 0.85) | 밝은 배경
    const c = x < 30 ? 26 + ((x * 7 + y * 3) % 9) : x === 30 ? 60 : 245;
    rgba.set([c, c, c, 255], i * 4);
    alpha[i] = x < 30 ? 1 : x === 30 ? 0.85 : 0;
  }
  const ref = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { const v = 150 + (i % 60); ref.set([v + 30, v, v - 5, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 50)!, rd = hairDist(ref, new Float32Array(n).fill(1), 0.6, 50)!;
  const out = recolorHair(rgba, w, h, alpha, mine, rd, 1);
  const lum = (i: number) => 0.2126 * out[i * 4] + 0.7152 * out[i * 4 + 1] + 0.0722 * out[i * 4 + 2];
  let inner = 0, rim = 0;
  for (let y = 5; y < 35; y++) { inner += lum(y * w + 15); rim += lum(y * w + 30); }
  inner /= 30; rim /= 30;
  assert.ok(rim < inner + 15, `가장자리 ${rim.toFixed(0)}, 머리 속 ${inner.toFixed(0)}`);
});

test('검은 머리 밝게 염색: 반쯤 섞인 가장자리(알파 0.5)는 검은 선이 되지 않고, 새 색·배경보다 밝아지지도 않는다', () => {
  const w = 60, h = 40, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // 검은 머리(루마 26~34) | 두 줄은 머리 반·배경 반(루마 ≈ 135, 알파 0.5) | 밝은 배경 240
    const c = x < 30 ? 26 + ((x * 7 + y * 3) % 9) : x < 32 ? 135 : 240;
    rgba.set([c, c, c, 255], i * 4);
    alpha[i] = x < 30 ? 1 : x < 32 ? 0.5 : 0;
  }
  const ref = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { const v = 190 + (i % 30); ref.set([v + 20, v, v - 5, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 50)!, rd = hairDist(ref, new Float32Array(n).fill(1), 0.6, 50)!;
  const out = recolorHair(rgba, w, h, alpha, mine, rd, 1);
  const lum = (i: number) => 0.2126 * out[i * 4] + 0.7152 * out[i * 4 + 1] + 0.0722 * out[i * 4 + 2];
  let inner = 0, rim = 0, maxRim = 0;
  for (let y = 5; y < 35; y++) { inner += lum(y * w + 20); const r = lum(y * w + 31); rim += r; maxRim = Math.max(maxRim, r); }
  inner /= 30; rim /= 30;
  // 이상적인 값은 머리 반(새 색) + 배경 반 ≈ (머리 속 + 240) / 2. 픽셀 전체를 당기면 그보다 30 넘게 어둡다
  const ideal = (inner + 240) / 2;
  assert.ok(rim > ideal - 15, `가장자리 ${rim.toFixed(0)}, 기대 ${ideal.toFixed(0)}`);
  assert.ok(maxRim < Math.max(inner, 240) + 3, `가장자리 최대 ${maxRim.toFixed(0)}`);
});

test('올 무늬: 곱슬(결 방향이 흩어짐)은 정수리에서 사방으로 뻗는 긴 줄무늬가 아니라 짧은 결로', () => {
  const w = 160, h = 120, n = w * h;
  const alpha = new Float32Array(n).fill(1);
  // 흩어진 작은 고리(곱슬): 방향이 제각각인 2~3px 점 무늬
  let sd = 7;
  const rnd = () => ((sd = (sd * 1664525 + 1013904223) >>> 0) / 4294967296);
  const luma = new Float32Array(n);
  for (let k = 0; k < 900; k++) {
    const cx = rnd() * w, cy = rnd() * h, r = 1.5 + rnd() * 1.5;
    for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(h - 1, cy + r); y++) for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(w - 1, cx + r); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) luma[y * w + x] = 60;
    }
  }
  const t = strandTexture(luma, w, h, alpha, 3);
  // 정수리(가운데 위)에서 바깥으로 6px 떨어진 픽셀끼리의 상관: 사방 줄무늬면 높다
  const crownX = w / 2, crownY = 0.15 * w;
  let sxy = 0, sxx = 0, syy = 0;
  for (let y = 30; y < h - 10; y += 2) for (let x = 20; x < w - 20; x += 2) {
    const px = x - crownX, py = y - crownY, pl = Math.hypot(px, py);
    const x2 = Math.round(x + (px / pl) * 6), y2 = Math.round(y + (py / pl) * 6);
    if (x2 < 0 || y2 < 0 || x2 >= w || y2 >= h) continue;
    const a = t[y * w + x], b = t[y2 * w + x2];
    sxy += a * b; sxx += a * a; syy += b * b;
  }
  const corr = sxy / Math.sqrt(sxx * syy);
  assert.ok(corr < 0.3, `정수리 방향 상관 ${corr.toFixed(2)}`);
});

test('머리 속 채우기: 둘레가 다 머리카락인 곳의 덜 확신한 알파(0.85·구멍 0.5)는 1 로, 가장자리는 그대로', () => {
  const w = 80, h = 60, n = w * h;
  const a = new Float32Array(n);
  // 머리 덩어리(x 10~69, y 5~54)는 알파 0.85, 가운데 12×8 구멍은 0.5, 바깥은 0
  for (let y = 5; y < 55; y++) for (let x = 10; x < 70; x++) a[y * w + x] = x >= 34 && x < 46 && y >= 26 && y < 34 ? 0.5 : 0.85;
  const edge = a[30 * w + 10];
  fillInterior(a, w, h, 6);
  assert.ok(a[30 * w + 40] > 0.97, `구멍 ${a[30 * w + 40].toFixed(2)}`);
  assert.ok(a[15 * w + 25] > 0.97, `머리 속 ${a[15 * w + 25].toFixed(2)}`);
  assert.equal(a[30 * w + 10], edge, '가장자리 그대로');
  assert.equal(a[2 * w + 40], 0, '바깥 그대로');
});

test('잔머리: 머리 옆 가는 선이라도 색이 배경·머리 사이가 아니면(보라색 옷깃) 알파를 얻지 않는다', () => {
  // 갈색 머리(알파 1) 오른쪽 밝은 배경. x=34 는 갈색 가닥, x=37 은 같은 밝기의 보라색 선
  const w = 80, h = 60;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let c = [225, 222, 218];
    if (x < 30) { c = [95, 62, 35]; alpha[i] = 1; }
    if (x === 34) c = [120, 88, 62];
    if (x === 37) c = [92, 70, 125];
    rgba.set([...c, 255], i * 4);
  }
  flyaways(rgba, w, h, alpha, 12);
  const y = 30;
  assert.ok(alpha[y * w + 34] > 0.5, `갈색 가닥 ${alpha[y * w + 34].toFixed(2)}`);
  assert.ok(alpha[y * w + 37] < 0.15, `보라색 선 ${alpha[y * w + 37].toFixed(2)}`);
});

test('하이라이트 리본: 큰 사진 속 작은 머리에서도 리본(밝은 가닥)은 머리 크기에 맞게 가늘다', () => {
  // 900×900 사진 가운데 80×80 갈색 머리(결은 약하게). 고른 사진은 어두운 바탕 + 밝은 가닥(밝기 폭이 넓은 하이라이트)
  const W = 900, H = 900, n = W * H;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n);
  for (let i = 0; i < n; i++) rgba.set([235, 235, 235, 255], i * 4);
  for (let y = 410; y < 490; y++) for (let x = 410; x < 490; x++) {
    const i = y * W + x, v = 100 + ((x * 7 + y * 13) % 3);
    rgba.set([v, Math.round(v * 0.75), Math.round(v * 0.5), 255], i * 4);
    alpha[i] = 1;
  }
  const rn = 4000, ref = new Uint8ClampedArray(rn * 4);
  for (let i = 0; i < rn; i++) { const v = i % 5 === 0 ? 215 : 70 + (i % 30); ref.set([v, v - 5, v - 12, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 50)!, rd = hairDist(ref, new Float32Array(rn).fill(1), 0.6, 50)!;
  const out = recolorHair(rgba, W, H, alpha, mine, rd, 1);
  // 머리 속 밝은 픽셀(상위 15%) 가운데 둘레 7×7 의 60% 넘게 밝은 '덩어리' 몫: 리본이 머리에 비해 굵으면 넓은 덩어리가 된다
  const lum = (i: number) => 0.299 * out[i * 4] + 0.587 * out[i * 4 + 1] + 0.114 * out[i * 4 + 2];
  const vals: number[] = [];
  for (let y = 415; y < 485; y++) for (let x = 415; x < 485; x++) vals.push(lum(y * W + x));
  const thr = [...vals].sort((a, b) => a - b)[Math.floor(vals.length * 0.85)];
  let bright = 0, blob = 0;
  for (let y = 418; y < 482; y++) for (let x = 418; x < 482; x++) {
    if (lum(y * W + x) < thr) continue;
    bright++;
    let c = 0;
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) if (lum((y + dy) * W + x + dx) >= thr) c++;
    if (c > 49 * 0.6) blob++;
  }
  assert.ok(blob / bright < 0.1, `밝은 덩어리 몫 ${(blob / bright).toFixed(2)}`);
});

test('흑발 사진으로 바꾸기: 검은 머리(배경이 섞여 분포가 더 어두움)는 회색으로 밝아지지 않는다', () => {
  const w = 60, h = 40, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // 왼쪽 절반은 검은 배경이 머리로 읽힌 곳(루마 3~6), 오른쪽은 실제 검은 머리(루마 28~40)
    const v = x < 30 ? 3 + (x % 4) : 28 + ((x * 5 + y * 3) % 13);
    rgba.set([v, v, v, 255], i * 4);
    alpha[i] = 1;
  }
  // 고른 사진: 밝은 스튜디오의 흑발(가운데 L ≈ 15, 윤기 L ≈ 35)
  const rn = 3000, ref = new Uint8ClampedArray(rn * 4);
  for (let i = 0; i < rn; i++) { const v = i % 10 === 0 ? 85 : 20 + (i % 25); ref.set([v, v, v + 3, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 50)!, rd = hairDist(ref, new Float32Array(rn).fill(1), 0.6, 50)!;
  assert.ok(rd.L[Q / 2] < 25, `고른 사진 가운데 L ${rd.L[Q / 2].toFixed(1)}`);
  const out = recolorHair(rgba, w, h, alpha, mine, rd, 1);
  let worst = 0;
  for (let y = 5; y < 35; y++) for (let x = 35; x < 55; x++) { const o = (y * w + x) * 4; worst = Math.max(worst, out[o] - rgba[o]); }
  assert.ok(worst < 14, `원래보다 밝아진 정도 ${worst}`);
});

test('피부 거르기: 흑발 사진의 그늘진 이마(어둡지만 피부 색도)는 알파를 줄이고, 갈색 머리는 그대로', () => {
  const n = 900, rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n), face = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // 0~399 흑발, 400~449 머리 알파가 번진 그늘진 이마, 500~ 밝은 얼굴 피부
    rgba.set(i < 400 ? [8, 8, 9, 255] : i < 450 ? [110, 74, 53, 255] : [225, 180, 160, 255], i * 4);
    alpha[i] = i < 400 ? 1 : i < 450 ? 0.8 : 0;
    face[i] = i >= 500 ? 1 : 0;
  }
  skinGate(rgba, alpha, face);
  assert.equal(alpha[10], 1);
  // 전에는 밝기만 머리 쪽이라(전체 0.5 남짓) 0.8 그대로 → 흑발로 염색하면 색이 빠진 잿빛 얼룩
  assert.ok(alpha[420] < 0.3, `그늘진 이마 ${alpha[420]}`);
  // 갈색 머리(피부와 색도가 비슷)면 같은 색 픽셀도 손대지 않는다
  for (let i = 0; i < 400; i++) rgba.set([70, 45, 30, 255], i * 4);
  const a2 = new Float32Array(n);
  for (let i = 0; i < 400; i++) a2[i] = 1;
  for (let i = 400; i < 450; i++) a2[i] = 0.8;
  skinGate(rgba, a2, face);
  assert.ok(a2[420] > 0.75, `갈색 머리 ${a2[420]}`);
});

test('귀 거르기: 얼굴 옆 귀 자리의 밝고 붉은 픽셀(귀)은 덜 물들이고, 어두운 머리카락·귀 자리 밖은 그대로', () => {
  const w = 300, h = 300;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const alpha = new Float32Array(w * h);
  // 머리카락(어두운 갈색) 전체, 가운데 얼굴(밝은 피부)은 알파 0
  for (let i = 0; i < w * h; i++) { rgba.set([70, 45, 30, 255], i * 4); alpha[i] = 1; }
  for (let y = 80; y < 220; y++) for (let x = 100; x < 200; x++) { const i = y * w + x; rgba.set([230, 190, 170, 255], i * 4); alpha[i] = 0; }
  // 얼굴 점: 눈(33·263), 코(1), 옆 끝(234 왼쪽·454 오른쪽)
  const pts = Array.from({ length: 478 }, () => ({ x: 150, y: 150 }));
  pts[33] = { x: 125, y: 130 }; pts[263] = { x: 175, y: 130 }; pts[1] = { x: 150, y: 165 };
  pts[234] = { x: 100, y: 140 }; pts[454] = { x: 200, y: 140 };
  // 귀: 오른쪽 옆 끝 바깥(x 205~212, y 135~160) 밝고 붉은 살색
  for (let y = 135; y < 160; y++) for (let x = 205; x < 212; x++) rgba.set([215, 140, 120, 255], (y * w + x) * 4);
  // 귀 자리 밖의 같은 색(머리 위쪽 윤기 아님, 그냥 밝은 조각)
  for (let y = 20; y < 30; y++) for (let x = 20; x < 30; x++) rgba.set([215, 140, 120, 255], (y * w + x) * 4);
  earGate(rgba, alpha, w, h, pts);
  assert.ok(alpha[147 * w + 208] < 0.3, `귀 ${alpha[147 * w + 208]}`);
  assert.ok(alpha[147 * w + 225] > 0.95, `귀 옆 머리카락 ${alpha[147 * w + 225]}`);
  assert.ok(alpha[25 * w + 25] > 0.95, `귀 자리 밖 ${alpha[25 * w + 25]}`);
});

test('가장자리 알파: 어두운 머리 둘레의 머리색 픽셀(알파 0.7)과 이마 경계 반반 픽셀을 올리고, 금발처럼 피부와 비슷한 머리는 건드리지 않는다', () => {
  const w = 60, h = 60;
  const make = (hair: number, skin: number) => {
    const rgba = new Uint8ClampedArray(w * h * 4), alpha = new Float32Array(w * h), faceP = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      // 위 20줄: 머리 속(알파 1), 20~21줄: 머리색이지만 알파 0.7, 22줄: 머리·피부 반반(알파 0.2), 그 아래 피부(얼굴)
      const v = y < 22 ? hair : y === 22 ? (hair + skin) / 2 : skin;
      rgba.set([v, v * 0.8, v * 0.7, 255], i * 4);
      alpha[i] = y < 20 ? 1 : y < 22 ? 0.7 : y === 22 ? 0.2 : 0;
      faceP[i] = y > 26 ? 1 : 0;
    }
    return { rgba, alpha, faceP };
  };
  const dark = make(25, 170);
  edgeAlpha(dark.rgba, dark.alpha, w, h, dark.faceP);
  assert.ok(dark.alpha[21 * w + 30] > 0.9, `머리색 띠 ${dark.alpha[21 * w + 30]}`);
  assert.ok(dark.alpha[22 * w + 30] > 0.35, `경계 반반 ${dark.alpha[22 * w + 30]}`);
  assert.equal(dark.alpha[30 * w + 30], 0, '피부는 그대로');
  const blond = make(150, 175);
  edgeAlpha(blond.rgba, blond.alpha, w, h, blond.faceP);
  assert.ok(Math.abs(blond.alpha[21 * w + 30] - 0.7) < 1e-6, '금발은 그대로');
});

test('가장자리 알파: 머리 옆면의 넓은 머리색 띠(확실한 머리에서 8px, 알파 0.55)도 올리지만, 분할이 머리로 안 본 어두운 곳(알파 0)은 그대로', () => {
  const w = 60, h = 60;
  const rgba = new Uint8ClampedArray(w * h * 4), alpha = new Float32Array(w * h), faceP = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // y 40 위: x 0~29 확실한 머리, 30~39 머리색인데 알파 0.55(옆머리 띠), 그 오른쪽 피부. y 40~51 피부, y 52 아래 어두운 옷(알파 0)
    const hairPx = y < 40 && x < 40, cloth = y >= 52;
    const v = hairPx || cloth ? 25 : 170;
    rgba.set([v, v * 0.8, v * 0.7, 255], i * 4);
    alpha[i] = !hairPx ? 0 : x < 30 ? 1 : 0.55;
    faceP[i] = x > 45 ? 1 : 0;
  }
  edgeAlpha(rgba, alpha, w, h, faceP);
  assert.ok(alpha[20 * w + 38] > 0.9, `옆머리 띠 ${alpha[20 * w + 38]}`);
  assert.equal(alpha[55 * w + 20], 0, '알파 0 인 어두운 곳은 그대로');
});

test('밝게 염색한 머리 다듬기: 날카로운 흰 줄(은박) 대비는 줄이고 평균 밝기는 그대로, 세기 0 이면 그대로', () => {
  const w = 60, h = 60, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n).fill(1);
  // 2px 마다 아주 밝은 줄(윤기 반사가 흰색으로 바뀐 것)
  for (let i = 0; i < n; i++) { const x = i % w; const v = x % 4 < 1 ? 245 : 150; rgba.set([v, v - 10, v - 15, 255], i * 4); }
  const sd = (a: Uint8ClampedArray) => { let s = 0, s2 = 0; for (let i = 0; i < n; i++) { s += a[i * 4]; s2 += a[i * 4] ** 2; } const m = s / n; return { m, sd: Math.sqrt(s2 / n - m * m) }; };
  const before = sd(rgba);
  const a1 = new Uint8ClampedArray(rgba);
  softenLiftDetail(a1, w, h, alpha, 1, 1);
  const after = sd(a1);
  assert.ok(after.sd < before.sd * 0.7, `대비 ${before.sd.toFixed(1)} → ${after.sd.toFixed(1)}`);
  assert.ok(Math.abs(after.m - before.m) < 12, `평균 ${before.m.toFixed(1)} → ${after.m.toFixed(1)}`);
  const a0 = new Uint8ClampedArray(rgba);
  softenLiftDetail(a0, w, h, alpha, 1, 0);
  assert.equal(sd(a0).sd.toFixed(3), before.sd.toFixed(3));
});

test('검은 머리 밝게 염색: 내 머리 명암은 그대로 두고 색만 — 가는 윤기 반사가 흰 줄로 커지지 않고, 그늘은 남는다', () => {
  // 한국인 검은 머리: 바탕 루마 22~26, 4줄마다 가는 윤기(루마 60), 오른쪽 절반은 넓은 그늘(바탕의 60%)
  const w = 80, h = 60, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n).fill(1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const base = 22 + ((x * 3 + y) % 5);
    let c = y % 4 === 0 ? 60 : base;
    if (x >= 40) c = Math.round(c * 0.6);
    rgba.set([c, c, c, 255], (y * w + x) * 4);
  }
  const ref = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { const v = 150 + (i % 40); ref.set([v + 25, v, v - 15, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 100)!, rd = hairDist(ref, new Float32Array(n).fill(1), 0.6, 100)!;
  const out = recolorHair(rgba, w, h, alpha, mine, rd, 1);
  const lab = new Float32Array(3);
  const L = (x: number, y: number) => { const o = (y * w + x) * 4; rgb2lab(out[o], out[o + 1], out[o + 2], lab); return lab[0]; };
  let glint = 0, plain = 0, shade = 0, k = 0;
  for (let y = 12; y < 48; y += 4) for (let x = 10; x < 30; x++) { glint += L(x, y); plain += L(x, y + 2); shade += L(x + 40, y + 2); k++; }
  glint /= k; plain /= k; shade /= k;
  // 윤기는 밝힌 머리 위 반사 정도(L 15 안), 그늘 쪽은 여전히 어둡다(L 5 넘게)
  assert.ok(glint - plain < 15, `윤기 ${glint.toFixed(0)}, 바탕 ${plain.toFixed(0)}`);
  assert.ok(plain - shade > 5, `바탕 ${plain.toFixed(0)}, 그늘 ${shade.toFixed(0)}`);
});
