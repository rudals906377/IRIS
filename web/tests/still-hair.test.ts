/// <reference types="node" />
// 사진 모드 염색(분위 대응 색 옮기기·가이디드 필터) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Q, colorGate, flyaways, guidedFilter, hairDist, pureHair, skinGate, quantilePos, recolorHair } from '../src/beauty/still-hair.ts';

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
  // 80×60: 왼쪽 30칸 검은 머리(알파 1), 그 오른쪽 밝은 벽. x=34 에 1px 가닥(분할 0), x=26~27 은 머리 안으로 비친 벽(알파 1)
  const w = 80, h = 60;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const alpha = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    let v = 220;
    if (x < 30) { v = 30; alpha[i] = 1; }
    if (x === 26 || x === 27) v = 218;
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

test('검은 머리 밝게 염색: 배경이 섞인 가장자리 바로 안쪽의 새까만 픽셀이 하얀 테두리가 되지 않는다', () => {
  const w = 60, h = 40, n = w * h;
  const rgba = new Uint8ClampedArray(n * 4), alpha = new Float32Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    // 검은 머리(약한 결, 루마 28~36) | 바깥 두 줄은 새까맣게(루마 15) | 가장자리 한 줄(배경과 섞임) | 밝은 배경
    const c = x < 28 ? 30 + ((x * 7 + y * 3) % 9) : x < 30 ? 15 : x === 30 ? 140 : 245;
    rgba.set([c, c, c, 255], i * 4);
    alpha[i] = x < 30 ? 1 : x === 30 ? 0.5 : 0;
  }
  const ref = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) { const v = 150 + (i % 60); ref.set([v + 30, v, v - 5, 255], i * 4); }
  const mine = hairDist(rgba, alpha, 0.6, 50)!, rd = hairDist(ref, new Float32Array(n).fill(1), 0.6, 50)!;
  const out = recolorHair(rgba, w, h, alpha, mine, rd, 1);
  const lum = (i: number) => 0.2126 * out[i * 4] + 0.7152 * out[i * 4 + 1] + 0.0722 * out[i * 4 + 2];
  let inner = 0;
  for (let y = 5; y < 35; y++) inner += lum(y * w + 15);
  inner /= 30;
  let rim = 0;
  for (let y = 5; y < 35; y++) rim += lum(y * w + 29);
  rim /= 30;
  assert.ok(rim < inner + 15, `테두리 ${rim.toFixed(0)}, 머리 속 ${inner.toFixed(0)}`);
});
