/// <reference types="node" />
// 측정·필터 로직 단위 테스트(브라우저 없이 Node에서 실행: npm test)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { AdaptiveFilter2, OneEuro } from '../src/engine/filters.ts';
import { FaceTracker } from '../src/engine/face.ts';
import { DEFAULT_LOOPBACK_CONFIG, LoopbackTest } from '../src/engine/loopback.ts';
import { Rolling, quantileSorted, summarize } from '../src/engine/stats.ts';

test('분위수 계산', () => {
  assert.equal(quantileSorted([1, 2, 3, 4, 5], 0.5), 3);
  assert.equal(quantileSorted([10, 20], 0.5), 15);
  assert.ok(Number.isNaN(quantileSorted([], 0.5)));
  const r = new Rolling(3);
  for (const v of [100, 1, 2, 3]) r.push(v); // 용량 3 → 100은 밀려남
  assert.equal(r.max(), 3);
  assert.equal(r.quantile(0.5), 2);
  const s = summarize([5, 1, 3, NaN]);
  assert.equal(s.n, 3);
  assert.equal(s.p50, 3);
});

test('One Euro 필터: 정지 신호는 그대로, 계단 입력은 결국 따라감', () => {
  const f = new OneEuro();
  for (let t = 0; t < 1000; t += 33) assert.equal(f.filter(5, t), 5);
  let y = 0;
  for (let t = 1000; t < 3000; t += 33) y = f.filter(105, t);
  assert.ok(Math.abs(y - 105) < 0.5, `최종값 ${y}`);
});

/**
 * 가상의 화면·카메라 루프: 화면이 바뀐 뒤 delayMs 후 카메라가 새 밝기를 본다.
 * 카메라 프레임은 33ms 간격, 화면 갱신은 16.7ms 간격.
 */
function simulate(delayMs: number): LoopbackTest {
  let seed = 1;
  const rand = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const lb = new LoopbackTest(DEFAULT_LOOPBACK_CONFIG, rand);
  const history: { t: number; color: 0 | 1 }[] = [];
  lb.start(0);
  let nextDraw = 0;
  let nextCam = 5;
  for (let t = 0; t < 60000 && lb.running; t += 1) {
    if (t >= nextDraw) {
      history.push({ t, color: lb.onDraw(t) });
      nextDraw += 1000 / 60;
    }
    if (t >= nextCam) {
      // delayMs 전에 화면에 있던 색을 카메라가 본다
      const seen = [...history].reverse().find((h) => h.t <= t - delayMs);
      const luma = seen?.color === 1 ? 200 : 20;
      lb.onCameraFrame(t, luma);
      nextCam += 1000 / 30;
    }
  }
  return lb;
}

test('거울 루프백: 알려진 지연을 프레임 간격 오차 안에서 측정', () => {
  const lb = simulate(80);
  assert.equal(lb.phase, 'done');
  assert.equal(lb.samples.length, DEFAULT_LOOPBACK_CONFIG.trials);
  const s = summarize(lb.samples);
  // 카메라 프레임 간격(33ms)과 화면 갱신 간격(17ms)만큼 늦게 감지될 수 있다
  assert.ok(s.p50 >= 80 && s.p50 <= 80 + 50, `중앙값 ${s.p50}`);
});

test('거울 루프백: 카메라가 화면을 못 보면 실패로 끝남', () => {
  const lb = new LoopbackTest();
  lb.start(0);
  for (let t = 0; t < 3000 && lb.running; t += 33) {
    lb.onDraw(t);
    lb.onCameraFrame(t, 100); // 밝기 변화 없음
  }
  assert.equal(lb.phase, 'failed');
});


test('변위 적응 필터: 잡음 수준 흔들림은 누르고, 큰 이동은 바로 따라감', () => {
  const f = new AdaptiveFilter2(1, 5, 0.05);
  f.filter(0, 0, 0);
  let maxJ = 0;
  for (let i = 1; i <= 30; i++) {
    const q = f.filter((i % 2 ? 0.8 : -0.8), 0, i * 33.333);
    maxJ = Math.max(maxJ, Math.abs(q.x));
  }
  assert.ok(maxJ < 0.1, `잡음 ${maxJ}`);
  const q = f.filter(40, 0, 31 * 33.333);
  assert.ok(q.x > 39, `큰 이동 ${q.x}`);
});

test('얼굴 점 필터: 머리 전체가 빠르게 움직이면 지연 없이, 가만히 있으면 떨림을 줄인다', () => {
  // 가짜 얼굴 점: 폭 200px, 점 478개
  const base = Array.from({ length: 478 }, (_, i) => ({ x: 320 + 90 * Math.cos(i * 0.37), y: 240 + 110 * Math.sin(i * 0.61) }));
  base[234] = { x: 220, y: 240 };
  base[454] = { x: 420, y: 240 };
  const ft = new FaceTracker();
  const norm = (pts: { x: number; y: number }[]) => pts.map((q) => ({ x: q.x / 640, y: q.y / 480 }));
  let seed = 3;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5) * 1.2;
  let jit = 0;
  let prev: { x: number; y: number }[] | null = null;
  for (let i = 0; i < 40; i++) {
    const out = ft.update(norm(base.map((q) => ({ x: q.x + rnd(), y: q.y + rnd() }))), i * 33.333, 640, 480)!;
    if (prev && i > 10) jit += out.p.reduce((a, q, j) => a + Math.hypot(q.x - prev![j].x, q.y - prev![j].y), 0) / 478;
    prev = out.p;
  }
  assert.ok(jit / 29 < 0.2, `정지 떨림 ${jit / 29}`);
  // 한 프레임에 30px 이동 + 5° 회전: 다음 프레임에 거의 그대로 따라가야 한다
  const t = (5 * Math.PI) / 180;
  const moved = base.map((q) => ({ x: 350 + (q.x - 320) * Math.cos(t) - (q.y - 240) * Math.sin(t), y: 240 + (q.x - 320) * Math.sin(t) + (q.y - 240) * Math.cos(t) }));
  const out = ft.update(norm(moved), 40 * 33.333, 640, 480)!;
  const err = out.p.reduce((a, q, j) => a + Math.hypot(q.x - moved[j].x, q.y - moved[j].y), 0) / 478;
  assert.ok(err < 1.5, `이동 오차 ${err}`);
});
