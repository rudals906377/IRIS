/// <reference types="node" />
// 뷰티 효과의 순수 계산(얼굴 영역·삼각형 분할) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { faceRegions, strokeStrip, triangulate, LIPS_OUTER } from '../src/beauty/face-regions.ts';
import { LOOKS, PALETTES } from '../src/beauty/palettes.ts';
import { measureLimbWidth, tattooMesh, type PosePoints } from '../src/beauty/tattoo-place.ts';

type P = { x: number; y: number };
const area = (pts: P[], tris: number[]): number => {
  let a = 0;
  for (let i = 0; i < tris.length; i += 3) {
    const [p, q, r] = [pts[tris[i]], pts[tris[i + 1]], pts[tris[i + 2]]];
    a += Math.abs((q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x)) / 2;
  }
  return a;
};
const polyArea = (pts: P[]): number => {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return Math.abs(a) / 2;
};

test('삼각형 분할: 오목 다각형의 넓이를 그대로 덮는다(양 방향)', () => {
  // 가운데가 파인 "입술" 모양
  const lip: P[] = [
    { x: 0, y: 10 }, { x: 10, y: 0 }, { x: 20, y: 4 }, { x: 30, y: 0 }, { x: 40, y: 10 },
    { x: 30, y: 18 }, { x: 20, y: 20 }, { x: 10, y: 18 },
  ];
  for (const poly of [lip, [...lip].reverse()]) {
    const tris = triangulate(poly);
    assert.equal(tris.length, (poly.length - 2) * 3);
    assert.ok(Math.abs(area(poly, tris) - polyArea(poly)) < 1e-6, `${area(poly, tris)} vs ${polyArea(poly)}`);
  }
});

test('선 띠: 점마다 양옆 두 점', () => {
  const s = strokeStrip([{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 20, y: 2 }], 4);
  assert.equal(s.length, 6);
  assert.ok(Math.abs(s[2].y - s[3].y) > 1);
});

test('얼굴 영역: 478점에서 부위 다각형과 얼굴 폭', () => {
  // 가짜 얼굴 점: 원 위에 흩뿌린 점(번호만 맞으면 된다)
  const p: P[] = Array.from({ length: 478 }, (_, i) => ({ x: 200 + 80 * Math.cos(i), y: 200 + 100 * Math.sin(i * 1.3) }));
  p[234] = { x: 100, y: 200 };
  p[454] = { x: 300, y: 200 };
  const r = faceRegions(p);
  assert.equal(r.faceW, 200);
  assert.equal(r.lipsOuter.length, LIPS_OUTER.length);
  assert.ok(r.shadowR.length > 10 && r.linerL.length > 5);
  assert.ok(r.blushR.rx > r.blushR.ry);
});

test('색상표·룩: 값이 0~1 범위', () => {
  for (const list of Object.values(PALETTES)) for (const p of list) for (const v of p.color) assert.ok(v >= 0 && v <= 1);
  for (const l of Object.values(LOOKS)) for (const v of Object.values(l.parts)) assert.ok(v!.amount > 0 && v!.amount <= 1);
});

// ---- 타투 위치 ----

function armPose(): PosePoints {
  // 사람 왼팔(이미지 오른쪽): 어깨 (300,100) → 팔꿈치 (300,200) → 손목 (300,300), 아래로 곧게
  const p = { 11: { x: 300, y: 100 }, 12: { x: 100, y: 100 }, 13: { x: 300, y: 200 }, 15: { x: 300, y: 300 }, 7: { x: 240, y: 20 }, 8: { x: 160, y: 20 } };
  const vis: Record<number, number> = {};
  for (const k of Object.keys(p)) vis[Number(k)] = 1;
  return { p, vis };
}

test('타투 격자: 아래팔 가운데에 놓이고, 원기둥 가장자리는 흐려진다', () => {
  const m = tattooMesh('forearmL', armPose(), { size: 0.5, aspect: 1, width: 40 })!;
  assert.ok(m);
  const n = m.data.length / 5;
  let cx = 0;
  let cy = 0;
  let minVis = 1;
  let maxVis = 0;
  for (let i = 0; i < n; i++) {
    cx += m.data[i * 5];
    cy += m.data[i * 5 + 1];
    minVis = Math.min(minVis, m.data[i * 5 + 4]);
    maxVis = Math.max(maxVis, m.data[i * 5 + 4]);
  }
  assert.ok(Math.abs(cx / n - 300) < 1 && Math.abs(cy / n - 250) < 1);
  // 팔 폭(지름 40) 밖으로 나가지 않는다
  for (let i = 0; i < n; i++) assert.ok(Math.abs(m.data[i * 5] - 300) <= 20.01);
  assert.ok(maxVis > 0.99 && minVis < 0.3);
  // 도안 위(v=0)는 팔꿈치 쪽, 도안 오른쪽(u=1)은 화면 오른쪽(팔이 아래로 향할 때)
  assert.ok(m.data[1] < m.data[m.data.length - 4]);
  assert.ok(m.data[12 * 5] > m.data[0]);
  assert.equal(m.index.length, 12 * 6 * 6);
});

test('타투 격자: 관절이 없으면 만들지 않는다', () => {
  assert.equal(tattooMesh('forearmR', armPose(), { size: 0.5, aspect: 1, width: null }), null);
});

test('분할로 팔 굵기 재기', () => {
  const W = 64;
  const H = 64;
  const flags = new Uint8ClampedArray(W * H * 4);
  // x 28~35(8칸)가 피부인 세로 막대
  for (let y = 0; y < H; y++) for (let x = 28; x < 36; x++) flags[(y * W + x) * 4 + 1] = 255;
  // 영상 640×640 → 분할 10배 축소: 굵기 약 80px
  const w = measureLimbWidth(flags, W, H, 640, 640, { x: 315, y: 100 }, { x: 315, y: 500 });
  assert.ok(w !== null && Math.abs(w - 80) <= 15, `굵기 ${w}`);
});

test('타투 격자: 가로로 긴 도안은 팔을 따라 돌리고, 목에서는 폭을 넘지 않는다', () => {
  const m = tattooMesh('forearmL', armPose(), { size: 0.5, aspect: 3, width: 40 })!;
  const n = m.data.length / 5;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    minY = Math.min(minY, m.data[i * 5 + 1]);
    maxY = Math.max(maxY, m.data[i * 5 + 1]);
  }
  // 세로(팔 방향)로 길게 놓인다
  assert.ok(maxY - minY > 40);
  const neck = tattooMesh('neckL', armPose(), { size: 1, aspect: 3.2, width: 30 })!;
  // 가운데 줄의 양 끝(도안 가로 폭)이 목 폭 이하
  const row = 3 * 13;
  const wNeck = Math.hypot(neck.data[(row + 12) * 5] - neck.data[row * 5], neck.data[(row + 12) * 5 + 1] - neck.data[row * 5 + 1]);
  assert.ok(wNeck <= 30.01, `목 도안 폭 ${wNeck}`);
});
