/// <reference types="node" />
// 뷰티 효과의 순수 계산(얼굴 영역·삼각형 분할) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { faceRegions, strokeStrip, triangulate, LIPS_OUTER } from '../src/beauty/face-regions.ts';
import { LOOKS, PALETTES } from '../src/beauty/palettes.ts';

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
