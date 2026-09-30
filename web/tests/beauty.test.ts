/// <reference types="node" />
// 뷰티 효과의 순수 계산(얼굴 영역·삼각형 분할) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bandMesh, faceRegions, strokeStrip, triangulate, LIPS_OUTER } from '../src/beauty/face-regions.ts';
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
  p[1] = { x: 200, y: 210 }; // 코끝: 정면
  const r = faceRegions(p);
  assert.equal(r.faceW, 200);
  assert.equal(r.lipsOuter.length, LIPS_OUTER.length);
  for (const m of [r.shadow, r.liner, r.brow, r.blush]) {
    assert.ok(m.length > 0 && m.length % 3 === 0);
    assert.ok(m.every((q) => q.v >= 0 && q.v <= 1));
  }
  assert.ok(Math.abs(r.yaw) < 1e-9 && r.visR === 1 && r.visL === 1);
  assert.equal(r.skinPts.length, 4);
  assert.equal(r.lipBody.length, 6);
});

test('얼굴 영역: 옆으로 돌리면 먼 쪽 화장이 옅어지고, 오버립은 입술을 넓힌다', () => {
  const p: P[] = Array.from({ length: 478 }, (_, i) => ({ x: 200 + 80 * Math.cos(i), y: 200 + 100 * Math.sin(i * 1.3) }));
  p[234] = { x: 100, y: 200 };
  p[454] = { x: 300, y: 200 };
  p[1] = { x: 125, y: 210 }; // 코끝이 사람 오른쪽(이미지 왼쪽)으로: 오른쪽 절반이 좁아짐
  const r = faceRegions(p);
  assert.ok(r.yaw > 0.5);
  assert.ok(r.visR < 0.5 && r.visL === 1);
  const peak = (m: { v: number }[], half: 0 | 1): number => Math.max(...m.slice(half * (m.length / 2), (half + 1) * (m.length / 2)).map((q) => q.v));
  assert.ok(peak(r.blush, 0) < peak(r.blush, 1));
  const lip = LIPS_OUTER.map((_, k) => ({ x: 200 + 30 * Math.cos((k / LIPS_OUTER.length) * Math.PI * 2), y: 300 + 10 * Math.sin((k / LIPS_OUTER.length) * Math.PI * 2) }));
  LIPS_OUTER.forEach((i, k) => (p[i] = lip[k]));
  const a0 = polyArea(faceRegions(p).lipsOuter);
  const a1 = polyArea(faceRegions(p, { overlip: 1 }).lipsOuter);
  const am = polyArea(faceRegions(p, { overlip: -1 }).lipsOuter);
  assert.ok(a1 > a0 * 1.1 && am < a0 * 0.9, `${am} ${a0} ${a1}`);
});

test('띠 망: 줄 2개 × 점 n개 → 삼각형 2(n-1)개, 부위 좌표는 첫 줄을 따라 누적', () => {
  const m = bandMesh([[{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 20, y: 0 }], [{ x: 0, y: 5 }, { x: 10, y: 5 }, { x: 20, y: 5 }]], (r) => 1 - r, 10);
  assert.equal(m.length, 12);
  assert.equal(Math.max(...m.map((q) => q.s)), 2);
  assert.equal(Math.max(...m.map((q) => q.t)), 0.5);
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

// ---- 사진 → 설정 계산 ----
import { dominantColors, gam, hairTarget, illumination, isSkin, lin, lipTarget, tintTarget, REF } from '../src/beauty/style-math.ts';

test('립 목표색: 기준 피부·중간 조명이면 사진 입술색이 그대로 목표색', () => {
  const lip = lin([0.8, 0.35, 0.35]);
  const t = lipTarget(lip, REF);
  const back = lin(t);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - lip[i]) < 0.01);
});

test('립 목표색: 어두운 조명의 사진은 목표색이 더 밝게 나온다(조명을 되돌림)', () => {
  const lip = lin([0.5, 0.2, 0.2]);
  const darkSkin = REF.map((v) => v * 0.4) as [number, number, number];
  const il = illumination(darkSkin);
  assert.ok(il[0] < 1 && il[1] < 1);
  const t = lipTarget(lip, darkSkin);
  assert.ok(lin(t)[0] > lip[0]);
});

test('비치는 색: 자연 비율과 같으면 화장 없음, 볼이 붉으면 붉은 블러셔', () => {
  const natural: [number, number, number] = [0.98, 0.87, 0.83];
  assert.equal(tintTarget(natural, natural, 0.9, 0.28), null);
  const t = tintTarget([0.97, 0.72, 0.66], natural, 0.9, 0.28)!;
  assert.ok(t);
  const l = lin(t.color);
  // 빨강은 거의 유지, 초록·파랑은 줄어든 색
  assert.ok(l[0] / REF[0] > 0.9 && l[1] / REF[1] < 0.7 && l[2] / REF[2] < 0.7);
});

test('헤어: 위아래 색이 다르면 옴브레', () => {
  const dark = lin([0.2, 0.15, 0.12]);
  const light = lin([0.8, 0.65, 0.45]);
  assert.equal(hairTarget(dark, dark, dark).tip, null);
  const o = hairTarget(dark, dark, light);
  assert.ok(o.tip && gam(light)[0] - o.tip[0] < 0.01);
});

test('주요 색과 피부 판정', () => {
  const px: [number, number, number][] = [];
  for (let i = 0; i < 60; i++) px.push(lin([0.1, 0.2, 0.6]));
  for (let i = 0; i < 20; i++) px.push(lin([0.9, 0.9, 0.9]));
  const d = dominantColors(px, 2);
  assert.ok(d[0].share > 0.7 && gam(d[0].color)[2] > 0.5);
  assert.ok(isSkin(lin([0.85, 0.65, 0.55])));
  assert.ok(!isSkin(lin([0.2, 0.3, 0.8])));
});

// ---- 되먹임 계산 ----
import { matchDarkness, matchLip, matchTint } from '../src/beauty/style-match.ts';

test('되먹임: 내 결과가 사진보다 옅으면 진하기를 올리고, 같으면 유지', () => {
  const natural: [number, number, number] = [0.98, 0.94, 0.92];
  const cur = { color: gam([0.6, 0.19, 0.11]) as [number, number, number], amount: 0.5 };
  const want: [number, number, number] = [0.96, 0.78, 0.74];
  const weak: [number, number, number] = [0.97, 0.86, 0.83];
  const up = matchTint(cur, want, weak, natural, natural);
  // 결과가 옅으면 색 편차가 커진다(초록이 더 줄어듦)
  assert.ok(lin(up.color)[1] < lin(cur.color)[1]);
  const same = matchTint(cur, want, want, natural, natural);
  assert.ok(Math.abs(lin(same.color)[1] - lin(cur.color)[1]) < 0.02);
  // 사진에 블러셔가 없으면 거의 끈다
  const none = matchTint(cur, natural, weak, natural, natural);
  assert.ok(none.amount <= 0.2);
});

test('되먹임: 립은 결과/목표 비율로 색을 고친다', () => {
  const cur = { color: gam([0.5, 0.2, 0.2]) as [number, number, number], amount: 0.85 };
  const m = matchLip(cur, [0.5, 0.2, 0.2], [0.4, 0.25, 0.2]);
  const l = lin(m.color);
  assert.ok(l[0] > 0.5 && l[1] < 0.2 && Math.abs(l[2] - 0.2) < 0.01);
});

test('되먹임: 어두운 부위는 밝기 비율로 진하기를 맞추고, 내 쪽이 이미 더 어두우면 끈다', () => {
  const m = matchDarkness({ color: [0, 0, 0], amount: 0.4 }, 0.25, 0.4, 0.5, 0.5);
  assert.ok(m.amount > 0.4);
  // 사진 눈썹(0.36/0.44)보다 내 자연 눈썹(0.21)이 더 어둡다 → 최소
  const off = matchDarkness({ color: [0, 0, 0], amount: 0.4 }, 0.36, 0.21, 0.44, 0.21);
  assert.equal(off.amount, 0.1);
});
