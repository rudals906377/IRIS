/// <reference types="node" />
// 뷰티 효과의 순수 계산(얼굴 영역·삼각형 분할) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bandMesh, faceRegions, strokeStrip, triangulate, LIPS_OUTER } from '../src/beauty/face-regions.ts';
import { LOOKS, PALETTES } from '../src/beauty/palettes.ts';
import { measureLimbWidth, placeFromPoint, tattooMesh, type PosePoints } from '../src/beauty/tattoo-place.ts';
import { colorSkin, denseSleeve, dropEdgeBits, extractDesign, isDesignSheet, neutralInk, proposeRect, sheetRect, sleeveInk } from '../src/beauty/tattoo-extract.ts';

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
import { dominantColors, gam, hairTarget, illumination, isSkin, lin, lipCloseup, lipTarget, tintTarget, REF } from '../src/beauty/style-math.ts';

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
  // 사진에 블러셔가 없으면 끈다
  const none = matchTint(cur, natural, weak, natural, natural);
  assert.equal(none.amount, 0);
});

test('되먹임: 결과에서 거의 안 움직인 채널(섀도 파랑)도 사진 쪽으로 고친다', () => {
  const natural: [number, number, number] = [1, 1, 1];
  // 사진: 빨강 ↑, 초록·파랑 ↓(갈색 섀도). 지금 결과: 빨강·초록은 맞지만 파랑이 그대로
  const want: [number, number, number] = [1.3, 0.92, 0.8];
  const got: [number, number, number] = [1.28, 0.92, 1.0];
  const cur = { color: gam([0.8, 0.31, 0.25]) as [number, number, number], amount: 0.85 };
  const m = matchTint(cur, want, got, natural, natural);
  assert.ok(lin(m.color)[2] < lin(cur.color)[2] * 0.95, `파랑 ${lin(cur.color)[2]} → ${lin(m.color)[2]}`);
  // 이미 맞는 빨강은 크게 바뀌지 않는다
  assert.ok(Math.abs(lin(m.color)[0] - lin(cur.color)[0]) < 0.1);
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

// ---- 스타일 AI 속성 → 힌트 ----
import { hintsFromStyleAI, promptFromStyleAI } from '../src/beauty/style-attributes.ts';

test('스타일 AI 힌트: 확신 높은 속성만 쓰고, 낮은 것은 weak에 남긴다', () => {
  const h = hintsFromStyleAI({
    category: 'makeup',
    attributes: [
      { group: 'lipTexture', label: '블러립', score: 0.8 },
      { group: 'eye', label: '핑크·코랄 섀도', score: 0.7 },
      { group: 'cheek', label: '홍조 블러셔', score: 0.3 },
      { group: 'base', label: '물광', score: 0.9 },
    ],
    secondary: { category: 'hair', attributes: [{ group: 'colorTech', label: '옴브레', score: 0.6 }] },
  });
  assert.equal(h.makeup?.lipStyle, 'blur');
  assert.equal(h.makeup?.shadow, 'pink');
  assert.equal(h.makeup?.blush, undefined);
  assert.deepEqual(h.weak, ['cheek:홍조 블러셔']);
  assert.equal(h.makeup?.base, 'dewy');
  assert.equal(h.hair?.tech, 'ombre');
});

test('스타일 AI 힌트: 네일 디자인·길이, 타투 부위·크기', () => {
  const n = hintsFromStyleAI({ category: 'nail', attributes: [{ group: 'design', label: '자석', score: 0.9 }, { group: 'length', label: '롱네일', score: 0.6 }] });
  assert.equal(n.nail?.style, 'cateye');
  assert.equal(n.nail?.length, 0.8);
  const t = hintsFromStyleAI({ category: 'tattoo', attributes: [{ group: 'placement', label: '쇄골', score: 0.7 }, { group: 'size', label: '미니', score: 0.8 }, { group: 'color', label: '블랙', score: 0.9 }] });
  assert.equal(t.tattoo?.place, 'chest');
  assert.equal(t.tattoo?.size, 0.15);
  assert.equal(t.tattoo?.color, 'black');
});

test('생성 모드 글 설명: 분야에 맞는 속성의 영어 설명만 이어 붙인다', () => {
  const r = {
    category: 'hair',
    attributes: [
      { group: 'length', label: '긴머리', label_en: 'long hair that falls well below the shoulders', score: 0.8 },
      { group: 'color', label: '골드블론드', label_en: 'warm golden honey blonde hair', score: 0.7 },
      { group: 'mood', label: '청순', label_en: 'innocent', score: 0.9 },
      { group: 'bangs', label: '풀뱅', label_en: 'full thick blunt bangs', score: 0.2 },
    ],
  };
  const p = promptFromStyleAI(r, 'hair');
  assert.ok(p.includes('long hair') && p.includes('golden') && !p.includes('innocent') && !p.includes('bangs'));
  assert.equal(promptFromStyleAI(r, 'nail'), '');
  assert.equal(promptFromStyleAI(undefined, 'hair'), '');
});

test('타투 자리 탭: 누른 점에 가장 가까운 부위와 위치, 그 자리에 도안 가운데가 온다', () => {
  const pose = armPose();
  // 아래팔(팔꿈치 200 → 손목 300)의 3/4 지점, 축에서 오른쪽으로 10px
  const r = placeFromPoint(pose, { x: 310, y: 275 }, () => 40)!;
  assert.equal(r.place, 'forearmL');
  assert.ok(Math.abs(r.along - 0.75) < 0.01);
  const m = tattooMesh(r.place, pose, { size: 0.2, aspect: 1, width: 40, along: r.along, across: r.across })!;
  const c = (3 * 13 + 6) * 5; // 가운데 정점(7행 중 4번째, 13열 중 7번째)
  assert.ok(Math.abs(m.data[c] - 310) < 0.5 && Math.abs(m.data[c + 1] - 275) < 0.5);
  // 팔에서 먼 곳은 자리가 아니다
  assert.equal(placeFromPoint(pose, { x: 500, y: 250 }, () => 40), null);
});

test('타투 도안 뽑기: 피부 위 선은 잡고, 피부 위 새까만 큰 띠(끈)는 버린다', () => {
  const w = 200;
  const h = 200;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h).fill(1);
  for (let i = 0; i < w * h; i++) d.set([220, 170, 150, 255], i * 4);
  // 도안: (60~100, 60~100) 안의 가는 십자선(회색빛 잉크)
  for (let k = 60; k <= 100; k++) {
    for (const [x, y] of [[80, k], [k, 80], [81, k], [k, 81]]) d.set([90, 70, 70, 255], (y * w + x) * 4);
  }
  // 끈: x 150~165 세로 띠, 새까맣다
  for (let y = 0; y < h; y++) for (let x = 150; x < 166; x++) d.set([10, 10, 12, 255], (y * w + x) * 4);
  const r = extractDesign({ w, h, d, skin })!;
  assert.ok(r);
  assert.ok(r.rect.x < 80 && r.rect.x + r.rect.w > 80 && r.rect.x + r.rect.w < 140, `영역 ${JSON.stringify(r.rect)}`);
  // 잉크 픽셀은 진하고 색(투과율)은 피부보다 어둡다
  const i = (81 - r.rect.y) * r.w + (80 - r.rect.x);
  assert.ok(r.rgba[i * 4 + 3] > 200);
  assert.ok(r.rgba[i * 4] < 140);
});

test('색으로 잡은 피부: 피부색이 넓은 가까운 팔 사진은 팔과 그 위 색 타투(둘러싸인 구멍)까지 피부, 배경은 아님. 피부색이 적으면 null', () => {
  const w = 120, h = 120;
  const d = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const arm = x >= 20 && x < 100;
    const tat = (x - 60) ** 2 + (y - 60) ** 2 < 15 ** 2;
    d.set(tat ? [40, 140, 70, 255] : arm ? [225, 170, 150, 255] : [60, 90, 70, 255], (y * w + x) * 4);
  }
  const s = colorSkin(d, w, h)!;
  assert.ok(s, '피부 있음');
  assert.equal(s[60 * w + 30], 1, '팔');
  assert.equal(s[60 * w + 60], 1, '팔 위 타투(둘러싸인 구멍)');
  assert.equal(s[60 * w + 5], 0, '배경');
  const bg = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) bg.set([60, 90, 70, 255], i * 4);
  assert.equal(colorSkin(bg, w, h), null);
});

test('타투 도안 뽑기: 배율을 곱해 소수가 된 영역도 잉크를 그대로 뽑는다(빈 도안이 되지 않게)', () => {
  const w = 200, h = 200;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h).fill(1);
  for (let i = 0; i < w * h; i++) d.set([220, 170, 150, 255], i * 4);
  for (let k = 60; k <= 100; k++) for (const [x, y] of [[80, k], [k, 80]]) d.set([60, 50, 50, 255], (y * w + x) * 4);
  const r = extractDesign({ w, h, d, skin }, { x: 50.37, y: 49.81, w: 60.5, h: 61.26 })!;
  assert.ok(r);
  assert.ok(Number.isInteger(r.rect.x) && Number.isInteger(r.rect.w));
  let ink = 0;
  for (let i = 3; i < r.rgba.length; i += 4) if (r.rgba[i] > 128) ink++;
  assert.ok(ink > 40, `잉크 ${ink}`);
});

test('타투 영역 고르기: 피부 가장자리를 따라 길게 난 몸 윤곽선 그늘보다 가운데 가는 타투를 고른다', () => {
  const w = 200, h = 200;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 20; x < 200; x++) skin[y * w + x] = 1;
  for (let i = 0; i < w * h; i++) d.set(skin[i] ? [220, 170, 150, 255] : [240, 240, 240, 255], i * 4);
  // 윤곽선: 피부 왼쪽 가장자리 안쪽 2px 짜리 긴 어두운 띠(잉크 양이 타투보다 많다)
  for (let y = 0; y < h; y++) for (let x = 23; x < 25; x++) d.set([150, 110, 95, 255], (y * w + x) * 4);
  // 가운데 가는 십자 타투
  for (let k = 70; k <= 150; k++) for (const [x, y] of [[110, k], [k, 110]]) d.set([90, 70, 70, 255], (y * w + x) * 4);
  const r = proposeRect({ w, h, d, skin })!;
  assert.ok(r && r.x > 60 && r.x < 110 && r.x + r.w > 110, `영역 ${JSON.stringify(r)}`);
});

test('타투 도안: 옷 곁에 떨어진 작은 그늘 조각은 지우고, 도안에 붙은 점·옷에서 먼 점은 남긴다', () => {
  const w = 100, h = 100, full = new Float32Array(w * h * 4);
  const put = (x0: number, y0: number, x1: number, y1: number) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) full[(y * w + x) * 4 + 3] = 1; };
  put(30, 30, 70, 70); // 도안
  put(10, 5, 20, 9); // 옷 밑단 그늘(옷 곁, 떨어짐)
  put(71, 50, 73, 52); // 도안에 붙은 점
  put(85, 85, 88, 88); // 옷에서 먼 떨어진 점(장식)
  const cloth = new Float32Array(w * h);
  for (let y = 0; y < 12; y++) for (let x = 0; x < w; x++) cloth[y * w + x] = 1;
  dropEdgeBits(full, w, h, cloth);
  const a = (x: number, y: number) => full[(y * w + x) * 4 + 3];
  assert.equal(a(15, 7), 0);
  assert.equal(a(50, 50), 1);
  assert.equal(a(72, 51), 1);
  assert.equal(a(86, 86), 1);
});

test('두 색 아이섀도: 바깥 가중치 망은 눈꼬리 1 → 앞머리 0, 섀도 망과 같은 크기', () => {
  const p = Array.from({ length: 478 }, (_, i) => ({ x: 200 + 80 * Math.cos(i), y: 200 + 100 * Math.sin(i * 1.3) }));
  p[234] = { x: 100, y: 200 };
  p[454] = { x: 300, y: 200 };
  p[1] = { x: 200, y: 210 };
  const r = faceRegions(p);
  assert.equal(r.shadowOuter.length, r.shadow.length);
  assert.ok(r.shadowOuter.every((q) => q.v >= 0 && q.v <= 1));
  assert.ok(r.shadowOuter.some((q) => q.v === 1) && r.shadowOuter.some((q) => q.v === 0));
  // 같은 자리의 정점: 섀도 망과 좌표가 같다
  for (let i = 0; i < r.shadow.length; i += 97) assert.deepEqual([r.shadow[i].x, r.shadow[i].y], [r.shadowOuter[i].x, r.shadowOuter[i].y]);
});

test('도안 시트 전체 영역: 위 띠의 가게 이름 글씨(가로로 긴 납작한 줄)와 그 위 장식은 빼고 아래 도안들만 감싼다', () => {
  const w = 300, h = 400;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h).fill(1);
  for (let i = 0; i < w * h; i++) d.set([225, 205, 175, 255], i * 4);
  const box = (x0: number, y0: number, x1: number, y1: number) => { for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if ((x + y) % 3 === 0 || x === x0 || y === y0) d.set([40, 35, 35, 255], (y * w + x) * 4); };
  box(140, 8, 160, 30); // 맨 위 장식(로고)
  for (let y = 45; y < 55; y++) for (let x = 20; x < 280; x += 2) d.set([40, 35, 35, 255], (y * w + x) * 4); // 가게 이름 글씨 줄
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) box(25 + c * 68, 90 + r * 75, 70 + c * 68, 140 + r * 75); // 도안 16개
  const r = sheetRect({ w, h, d, skin })!;
  assert.ok(r, '영역 없음');
  assert.ok(r.y > 55 && r.y < 90, `위 끝 ${r.y}(글씨 줄 55 아래, 첫 도안 90 위)`);
  assert.ok(r.x < 25 && r.x + r.w > 274 && r.y + r.h > 362, `영역 ${JSON.stringify(r)}`);
});

test('도안 시트: 흰 바탕에 도안이 띄엄띄엄 놓여도 시트로, 도안 하나(+작은 점)는 시트가 아니다', () => {
  const w = 300, h = 400;
  const mk = (): Uint8ClampedArray => { const d = new Uint8ClampedArray(w * h * 4); for (let i = 0; i < w * h; i++) d.set([250, 250, 250, 255], i * 4); return d; };
  const blob = (d: Uint8ClampedArray, cx: number, cy: number, r: number) => { for (let y = cy - r; y <= cy + r; y++) for (let x = cx - r; x <= cx + r; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r && (x + y) % 3 === 0) d.set([30, 30, 30, 255], (y * w + x) * 4); };
  const skin = new Float32Array(w * h).fill(1); // 종이 = 바탕
  // 도안 12개가 서로 멀리(넓힌 2% 보다 큰 틈) 떨어져 있다
  const sheet = mk();
  for (let r = 0; r < 4; r++) for (let c = 0; c < 3; c++) blob(sheet, 50 + c * 100, 50 + r * 100, 22);
  assert.equal(isDesignSheet({ w, h, d: sheet, skin }), true);
  // 도안 하나와 둘레 작은 점 몇 개
  const one = mk();
  blob(one, 150, 200, 80);
  for (let k = 0; k < 6; k++) blob(one, 30 + k * 45, 360, 3);
  assert.equal(isDesignSheet({ w, h, d: one, skin }), false);
});

test('빽빽한 소매 잉크 맵: 피부 분할이 거부한 노란 채움도 맨살 대비 투과율(파랑만 낮음)로 옮기고, 맨살은 그대로', () => {
  const w = 200, h = 200;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    const fill = x >= 110 && x < 170 && y >= 60 && y < 140;
    const line = (x === 110 || x === 169 || y === 60 || y === 139) && fill;
    d.set(line ? [40, 35, 35, 255] : fill ? [220, 196, 60, 255] : [220, 170, 150, 255], i * 4);
    skin[i] = fill ? 0.2 : 1; // 채움은 피부로 안 잡힘(옷도 아님)
  }
  const src = { w, h, d, skin, avoid: new Float32Array(w * h) };
  const r = sleeveInk(src, { x: 0, y: 0, w, h })!;
  assert.ok(r, '맵');
  const at = (x: number, y: number) => { const o = (y * r.w + x) * 4; return [r.rgba[o], r.rgba[o + 1], r.rgba[o + 2], r.rgba[o + 3]]; };
  const yel = at(140, 100), bare = at(40, 100);
  assert.ok(yel[3] > 200 && yel[2] < 120 && yel[0] > 230, `노란 채움 ${yel}`);
  assert.ok(bare[0] > 245 && bare[1] > 245 && bare[2] > 245, `맨살 ${bare}`);
  // 팔 영역(오른쪽 절반): 채움이 20%
  const region = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 80; x < w; x++) region[y * w + x] = 1;
  assert.equal(denseSleeve(src, region), true);
  assert.equal(denseSleeve({ ...src, skin: new Float32Array(w * h).fill(1) }, region), false);
});

test('무채색 잉크: 조명 색이 묻은 투과율(보라)을 밝기 그대로 회색으로', () => {
  const rgba = new Uint8ClampedArray([200, 120, 210, 255, 255, 255, 255, 0]);
  const L0 = 0.299 * 200 + 0.587 * 120 + 0.114 * 210;
  neutralInk(rgba);
  assert.equal(rgba[0], rgba[1]);
  assert.equal(rgba[1], rgba[2]);
  assert.ok(Math.abs(rgba[0] - L0) <= 1, `밝기 ${rgba[0]} vs ${L0}`);
  assert.equal(rgba[3], 255);
  assert.equal(rgba[4], 255);
});

test('입술 클로즈업: 얼굴이 안 보이는 립 사진에서 립 색을 읽고, 피부·흰 배경만 있는 사진은 null', () => {
  const w = 100, h = 80, d = new Uint8ClampedArray(w * h * 4);
  // 위 절반 흰 배경, 아래 왼쪽 피부, 가운데 코랄 립(입술 가운데는 더 진함)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const o = (y * w + x) * 4;
    let c = [245, 243, 240];
    if (y > 30) c = [228, 186, 164];
    if (y > 40 && y < 70 && x > 25 && x < 85) c = x > 40 && x < 70 ? [232, 110, 100] : [236, 130, 116];
    d.set([...c, 255], o);
  }
  const lip = lipCloseup(d, w, h)!;
  assert.ok(lip, '립 색');
  assert.ok(Math.abs(lip[0] * 255 - 232) < 8 && Math.abs(lip[1] * 255 - 110) < 12, `립 ${lip.map((v) => Math.round(v * 255))}`);
  const skinOnly = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) skinOnly.set(i % 2 ? [228, 186, 164, 255] : [245, 243, 240, 255], i * 4);
  assert.equal(lipCloseup(skinOnly, w, h), null);
});

test('타투 영역 고르기: 부드러운 팔 윤곽 그늘(굵기 5px, 길게)보다 작은 손목 타투를 고르고, 분할이 타투 둘레에 낸 구멍은 메운다', () => {
  const w = 240, h = 240;
  const d = new Uint8ClampedArray(w * h * 4);
  const skin = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 30; x < 240; x++) skin[y * w + x] = 1;
  for (let i = 0; i < w * h; i++) d.set(skin[i] ? [225, 175, 150, 255] : [235, 220, 205, 255], i * 4);
  // 윤곽 그늘: 피부 왼쪽 가장자리에 붙은 5px 폭의 부드러운 어두운 띠(위에서 아래까지, 가는 선 기준 2.9px 보다 굵다)
  for (let y = 0; y < h; y++) for (let x = 30; x < 35; x++) d.set([175, 130, 112, 255], (y * w + x) * 4);
  // 작은 고리 두 개(무한대) — 왼쪽 고리 둘레는 분할이 피부가 아니라고 본 구멍
  for (let t = 0; t < 360; t += 2) {
    const a = (t * Math.PI) / 180;
    for (const cx of [120, 150]) { const x = Math.round(cx + 13 * Math.cos(a)), y = Math.round(120 + 8 * Math.sin(a)); d.set([60, 45, 45, 255], (y * w + x) * 4); }
  }
  for (let y = 105; y < 136; y++) for (let x = 100; x < 136; x++) skin[y * w + x] = 0.2;
  const r = proposeRect({ w, h, d, skin })!;
  assert.ok(r && r.x > 90 && r.x < 115 && r.x + r.w > 160 && r.w < 90, `영역 ${JSON.stringify(r)}`);
});
