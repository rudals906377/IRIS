/// <reference types="node" />
// 생성 머리를 고개 방향에 맞춰 휘는 계산(3D 닮음 변환·머리 타원체) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { applyPose, buildWarpMesh, cutMesh, headProxy, projectMesh, proxyDepth, rotationDeg, similarity3, type Vec3 } from '../src/beauty/head-warp.ts';

const rotY = (deg: number): number[] => {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
};
const rotZ = (deg: number): number[] => {
  const a = (deg * Math.PI) / 180, c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
};
const mul = (A: number[], B: number[]): number[] => {
  const o = new Array(9).fill(0);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) o[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
  return o;
};

// 얼굴 비슷한 3D 점 구름(결정적)
function cloud(n = 40): Vec3[] {
  const pts: Vec3[] = [];
  for (let i = 0; i < n; i++) pts.push({ x: 300 + 80 * Math.cos(i * 1.7), y: 250 + 100 * Math.sin(i * 2.3), z: 30 * Math.sin(i * 0.9) });
  return pts;
}

test('3D 닮음 변환: 알려진 회전·크기·이동을 되찾는다', () => {
  const from = cloud();
  const R = mul(rotY(25), rotZ(-10));
  const P0 = { s: 1.3, R, t: { x: 40, y: -15, z: 7 } };
  const to = from.map((q) => applyPose(P0, q));
  const P = similarity3(from, to)!;
  assert.ok(Math.abs(P.s - 1.3) < 1e-6);
  for (let i = 0; i < 9; i++) assert.ok(Math.abs(P.R[i] - R[i]) < 1e-6, `R${i}`);
  assert.ok(Math.abs(rotationDeg(P.R) - rotationDeg(R)) < 1e-4);
  for (const q of from) {
    const a = applyPose(P, q), b = applyPose(P0, q);
    assert.ok(Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z) < 1e-6);
  }
});

test('3D 닮음 변환: 같은 점이면 회전 0°', () => {
  const p = cloud();
  const P = similarity3(p, p)!;
  assert.ok(rotationDeg(P.R) < 1e-3);
  assert.ok(Math.abs(P.s - 1) < 1e-9);
});

// 478 점 중 머리 타원체에 쓰는 점만 채운 가짜 얼굴(정면, 폭 200)
function fakeFace(): Vec3[] {
  const p: Vec3[] = Array.from({ length: 478 }, () => ({ x: 400, y: 300, z: 0 }));
  p[234] = { x: 300, y: 300, z: 20 };
  p[454] = { x: 500, y: 300, z: 20 };
  p[151] = { x: 400, y: 200, z: -60 };
  p[152] = { x: 400, y: 460, z: -40 };
  return p;
}

test('머리 타원체: 앞면이 이마를 지나고, 바깥은 중심 깊이의 평면', () => {
  const p = fakeFace();
  const h = headProxy(p);
  assert.ok(Math.abs(proxyDepth(h, p[151].x, p[151].y) - p[151].z) < 1e-6, '이마 깊이');
  assert.equal(proxyDepth(h, 1000, 1000), h.cz);
  // 가운데가 가장자리보다 카메라에 가깝다(더 작은 z)
  assert.ok(proxyDepth(h, h.cx, h.cy) < proxyDepth(h, h.cx + h.a * 0.9, h.cy));
});

test('격자 휘기: 같은 자세면 제자리, 고개를 돌리면 가까운 쪽보다 먼 쪽이 눌린다', () => {
  const p = fakeFace();
  const h = headProxy(p);
  const m = buildWarpMesh(h, 800, 600, { x0: 250, y0: 120, x1: 550, y1: 520 }, 8, 8);
  assert.equal(m.index.length, 8 * 8 * 6);
  const id = projectMesh(m, { s: 1, R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: { x: 0, y: 0, z: 0 } });
  for (let i = 0; i < m.pos.length; i++) {
    assert.ok(Math.abs(id[i * 2] - m.pos[i].x) < 1e-3 && Math.abs(id[i * 2 + 1] - m.pos[i].y) < 1e-3);
  }
  // 머리 중심 기준으로 y 축 30° 회전
  const R = rotY(30);
  const c = { x: h.cx, y: h.cy, z: h.cz };
  const rc = applyPose({ s: 1, R, t: { x: 0, y: 0, z: 0 } }, c);
  const P = { s: 1, R, t: { x: c.x - rc.x, y: c.y - rc.y, z: c.z - rc.z } };
  const xy = projectMesh(m, P);
  // 가운데 줄(5번째 행)에서 왼쪽 반과 오른쪽 반의 화면 폭
  const row = 4, cols = 8;
  const X = (i: number): number => xy[(row * (cols + 1) + i) * 2];
  const left = Math.abs(X(4) - X(0)), right = Math.abs(X(8) - X(4));
  assert.ok(Math.abs(left - right) > 5, `좌우 폭이 달라야 함 ${left.toFixed(1)} vs ${right.toFixed(1)}`);
});

test('얼굴 지우기 격자: 바깥 고리 0, 안쪽 1, 삼각형 수', () => {
  const pts = Array.from({ length: 10 }, (_, i) => ({ x: 100 + 50 * Math.cos(i * 0.628), y: 100 + 60 * Math.sin(i * 0.628) }));
  const c = cutMesh(pts);
  assert.equal(c.a.length, 21);
  assert.equal(c.a[0], 0);
  assert.equal(c.a[10], 1);
  assert.equal(c.index.length, 10 * 9);
});
