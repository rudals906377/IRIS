/// <reference types="node" />
// 사진 모드 염색(분위 대응 색 옮기기·가이디드 필터) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Q, guidedFilter, hairDist, quantilePos, recolorHair } from '../src/beauty/still-hair.ts';

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
