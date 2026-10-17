/// <reference types="node" />
// 타투 자리(다리) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { placeAxis } from '../src/beauty/tattoo-place.ts';

test('다리: 허벅지는 엉덩이 조금 아래 → 무릎, 종아리는 무릎 → 발목', () => {
  const pose = { p: { 23: { x: 100, y: 300 }, 25: { x: 100, y: 500 }, 27: { x: 100, y: 700 } }, vis: { 23: 1, 25: 1, 27: 1 } };
  const t = placeAxis('thighL', pose)!;
  assert.ok(t.a.y > 300 && t.a.y < 340 && t.b.y === 500 && t.cylinder);
  const c = placeAxis('calfL', pose)!;
  assert.equal(c.a.y, 500);
  assert.equal(c.b.y, 700);
  assert.equal(placeAxis('thighR', pose), null);
});
