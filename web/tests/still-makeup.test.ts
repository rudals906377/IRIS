/// <reference types="node" />
// 사진 모드 메이크업 옮기기 단위 테스트(가짜 얼굴 점)
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { transferMakeup } from '../src/beauty/still-makeup.ts';
import { rgb2lab } from '../src/beauty/still-hair.ts';
import { FACE_OVAL } from '../src/beauty/face-measure.ts';
import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_INNER, LIPS_OUTER } from '../src/beauty/face-regions.ts';

/** 가운데 (cx, cy), 반지름 R 인 둥근 가짜 얼굴의 점 468개 */
function face(cx: number, cy: number, R: number) {
  const p: { x: number; y: number }[] = [];
  // 나머지 점: 얼굴 안 격자(살짝 흔들어 같은 점이 없게)
  for (let i = 0; i < 468; i++) {
    const gx = (i % 22) / 21 - 0.5, gy = Math.floor(i / 22) / 21 - 0.5;
    p.push({ x: cx + gx * R * 1.2 + (i % 7) * 0.01, y: cy + gy * R * 1.2 + (i % 5) * 0.01 });
  }
  const ring = (ids: number[], ox: number, oy: number, rx: number, ry: number) =>
    ids.forEach((k, j) => { const t = (j / ids.length) * Math.PI * 2; p[k] = { x: cx + ox * R + rx * R * Math.cos(t), y: cy + oy * R + ry * R * Math.sin(t) }; });
  ring(FACE_OVAL, 0, 0, 1, 1.2);
  ring(LIPS_OUTER, 0, 0.6, 0.3, 0.1);
  ring(LIPS_INNER, 0, 0.6, 0.18, 0.02);
  ring([...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], -0.4, -0.2, 0.15, 0.06);
  ring([...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)], 0.4, -0.2, 0.15, 0.06);
  ring([...BROW_R_LOWER, ...BROW_R_UPPER], -0.4, -0.45, 0.2, 0.04);
  ring([...BROW_L_LOWER, ...BROW_L_UPPER], 0.4, -0.45, 0.2, 0.04);
  p[234] = { x: cx - R, y: cy };
  p[454] = { x: cx + R, y: cy };
  return p;
}

test('메이크업: 고른 사진 볼의 블러셔(붉은 기)가 내 볼로, 눈 안과 얼굴 밖은 그대로', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const skin = [200, 170, 150];
  const my = new Uint8ClampedArray(W * H * 4);
  const ref = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) { my[i * 4 + c] = skin[c]; ref[i * 4 + c] = skin[c]; }
    my[i * 4 + 3] = ref[i * 4 + 3] = 255;
  }
  // 고른 사진: 오른볼(이미지 왼쪽 아래)에 블러셔
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if ((x - 70) ** 2 + (y - 135) ** 2 < 15 ** 2) { const o = (y * W + x) * 4; ref[o] = 225; ref[o + 1] = 140; ref[o + 2] = 140; }
  }
  // 내 눈 안: 검은 눈동자
  const eyeO = (y: number, x: number) => (y * W + x) * 4;
  for (let y = 97; y < 100; y++) for (let x = 73; x < 79; x++) { const o = eyeO(y, x); my[o] = my[o + 1] = my[o + 2] = 30; }
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  const cheek = (70 + 135 * W) * 4;
  assert.ok(out[cheek] - out[cheek + 1] > my[cheek] - my[cheek + 1] + 15, `볼 붉은 기 ${out[cheek] - out[cheek + 1]}`);
  assert.equal(out[eyeO(98, 76)], 30, '눈 안 그대로');
  assert.equal(out[eyeO(5, 5)], my[eyeO(5, 5)], '얼굴 밖 그대로');
  // 반대쪽 볼(블러셔 없음)은 거의 그대로
  const other = (130 + 135 * W) * 4;
  assert.ok(Math.abs(out[other + 1] - my[other + 1]) < 8, `반대 볼 ${out[other + 1]}`);
});

test('주근깨 찾기: 둥근 어두운 점은 잡고, 가는 선(머리카락·손가락 가장자리)은 안 잡는다', async () => {
  const { dotMask } = await import('../src/beauty/still-makeup.ts');
  const w = 80, h = 40, L = new Float32Array(w * h).fill(70);
  // 왼쪽: 지름 4px 점, 오른쪽: 세로로 긴 선(폭 2px)
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if ((x - 20) ** 2 + (y - 20) ** 2 <= 4) L[y * w + x] = 50;
    if (x >= 59 && x <= 60) L[y * w + x] = 50;
  }
  const m = dotMask(L, w, h);
  assert.ok(m[20 * w + 20] > 0.5, `점 ${m[20 * w + 20]}`);
  assert.ok(m[20 * w + 60] < 0.2, `선 ${m[20 * w + 60]}`);
});

test('메이크업: 고른 사진의 하얗게 비친 볼(조명)은 내 볼을 잿빛(노란 기 빠짐)으로 만들지 않는다', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const skin = [200, 165, 135];
  const my = new Uint8ClampedArray(W * H * 4);
  const ref = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) { my[i * 4 + c] = skin[c]; ref[i * 4 + c] = skin[c]; }
    my[i * 4 + 3] = ref[i * 4 + 3] = 255;
  }
  // 고른 사진: 볼에 넓게 하얗게 비친 곳(노란 기가 빠진 밝은 피부)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if ((x - 70) ** 2 + (y - 130) ** 2 < 22 ** 2) { const o = (y * W + x) * 4; ref[o] = 228; ref[o + 1] = 218; ref[o + 2] = 214; }
  }
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  const o = (130 * W + 70) * 4, a = new Float32Array(3), b = new Float32Array(3);
  rgb2lab(my[o], my[o + 1], my[o + 2], a);
  rgb2lab(out[o], out[o + 1], out[o + 2], b);
  assert.ok(b[2] - a[2] > -4.5, `노란 기 변화 ${(b[2] - a[2]).toFixed(1)}`);
});

test('메이크업: 맨얼굴의 점 몇 개(점·여드름)는 옮기지 않고, 얼굴에 많이 그린 주근깨는 옮긴다', () => {
  const W = 320, H = 360, cx = 160, cy = 180, R = 110;
  const pts = face(cx, cy, R);
  const skin = [205, 170, 150];
  const make = (spots: [number, number][]) => {
    const ref = new Uint8ClampedArray(W * H * 4);
    for (let i = 0; i < W * H; i++) { ref.set(skin, i * 4); ref[i * 4 + 3] = 255; }
    for (const [x, y] of spots) for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) if (dx * dx + dy * dy <= 5) ref.set([120, 85, 70], ((y + dy) * W + x + dx) * 4);
    return ref;
  };
  const my = make([]);
  const few: [number, number][] = [[cx - 50, cy + 22], [cx + 50, cy + 20], [cx - 40, cy + 30]];
  const many: [number, number][] = [];
  for (let y = 0; y < H; y += 9) for (let x = 0; x < W; x += 9) if (((x - cx) / (0.8 * R)) ** 2 + ((y - cy) / (0.95 * R)) ** 2 < 1) many.push([x, y]);
  const lum = (o: Uint8ClampedArray, x: number, y: number) => { const i = (y * W + x) * 4; return o[i] + o[i + 1] + o[i + 2]; };
  const a = transferMakeup(my, W, H, pts, make(few), W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  assert.ok(lum(my, few[0][0], few[0][1]) - lum(a, few[0][0], few[0][1]) < 30, `점 몇 개 ${lum(my, few[0][0], few[0][1]) - lum(a, few[0][0], few[0][1])}`);
  const dbg: Record<string, unknown> = {};
  const b = transferMakeup(my, W, H, pts, make(many), W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 }, null, null, null, dbg);
  const ds = many.map(([x, y]) => lum(my, x, y) - lum(b, x, y)).sort((p, q) => p - q);
  const p90 = ds[Math.floor(ds.length * 0.9)];
  assert.ok(p90 > 100, `주근깨 ${p90} (점 ${dbg.nDots}개, ${many.length}개 그림)`);
});

test('메이크업(얼굴 바꾸기): 내 얼굴 굴곡 그늘의 따뜻한 색은 화색으로 보고 빼지 않는다(누런 회색 띠 방지)', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const skin = [215, 180, 160];
  const my = new Uint8ClampedArray(W * H * 4);
  const ref = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    for (let c = 0; c < 3; c++) { my[i * 4 + c] = skin[c]; ref[i * 4 + c] = skin[c]; }
    my[i * 4 + 3] = ref[i * 4 + 3] = 255;
  }
  // 내 얼굴(바뀐 얼굴): 코 옆 세로로 따뜻한 그늘. 고른 사진은 같은 자리에 그늘 없음
  for (let y = 100; y < 150; y++) for (let x = 78; x < 88; x++) my.set([180, 130, 110], (y * W + x) * 4);
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 }, null, null, null, undefined, 1);
  const o = (125 * W + 83) * 4, a = new Float32Array(3), b = new Float32Array(3);
  rgb2lab(my[o], my[o + 1], my[o + 2], a);
  rgb2lab(out[o], out[o + 1], out[o + 2], b);
  assert.ok(b[1] - a[1] > -3, `붉은 기 변화 ${(b[1] - a[1]).toFixed(1)}`);
});

test('메이크업: 밝은 피부 사진의 장밋빛 립을 어두운 피부에 옮겨도 누렇고 어두운 벽돌색이 되지 않는다', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const my = new Uint8ClampedArray(W * H * 4), ref = new Uint8ClampedArray(W * H * 4);
  const inLip = (x: number, y: number) => ((x - 100) / 17) ** 2 + ((y - 146) / 5) ** 2 < 1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4, lip = inLip(x, y);
    // 웃는 입: 입술 안쪽 띠(입 안 선 둘레)에 흰 이
    const tooth = lip && Math.abs(y - 145) <= 1 && Math.abs(x - 100) <= 9;
    my.set(tooth ? [236, 232, 224, 255] : lip ? [112, 66, 60, 255] : [125, 82, 56, 255], o);
    ref.set(lip ? [215, 125, 140, 255] : [236, 214, 204, 255], o);
  }
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  // 아랫입술 가운데(입 안 선은 y 146)
  const lab = new Float32Array(3), o = (149 * W + 100) * 4;
  rgb2lab(out[o], out[o + 1], out[o + 2], lab);
  const L = lab[0], b = lab[2];
  rgb2lab(my[o], my[o + 1], my[o + 2], lab);
  const myL = lab[0];
  // 고른 립은 붉고 노란 기가 적다(b ≈ 4): 내 입술(b ≈ 15)보다 노래지면 벽돌색, 내 입술보다 크게 어두우면 탁하다
  assert.ok(b < 14, `입술 b ${b.toFixed(1)}`);
  assert.ok(L > myL - 4, `입술 L ${L.toFixed(1)}, 원래 ${myL.toFixed(1)}`);
  // 이는 분홍으로 물들지 않는다
  const t = (145 * W + 104) * 4;
  rgb2lab(out[t], out[t + 1], out[t + 2], lab);
  assert.ok(lab[1] < 6, `이 a ${lab[1].toFixed(1)}`);
});

test('메이크업: 고른 사진 볼 위의 검은 것(손톱·머리카락)은 화장이 아니다 — 내 볼에 회색 얼룩으로 옮기지 않는다', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  // 콧구멍 둘레 점은 코 밑에(가짜 얼굴 격자에선 흩어져 있다)
  [64, 98, 240, 2, 460, 327, 294].forEach((k, j) => { pts[k] = { x: 92 + j * 3, y: 128 + (j % 2) * 3 }; });
  const skin = [215, 180, 160];
  const my = new Uint8ClampedArray(W * H * 4), ref = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) { my.set([...skin, 255], i * 4); ref.set([...skin, 255], i * 4); }
  // 고른 사진 오른볼에 검은 손톱 하나(지름 10px)
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x - 72) ** 2 + (y - 135) ** 2 < 25) ref.set([25, 22, 28, 255], (y * W + x) * 4);
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  const lab = new Float32Array(3), o = (135 * W + 72) * 4;
  rgb2lab(out[o], out[o + 1], out[o + 2], lab);
  const L = lab[0];
  rgb2lab(my[o], my[o + 1], my[o + 2], lab);
  assert.ok(L > lab[0] - 3, `볼 L ${L.toFixed(1)}, 원래 ${lab[0].toFixed(1)}`);
});

test('메이크업: 고른 사진 아랫입술 밑 붉은 그늘은 립이 아니다 — 내 턱에 분홍 얼룩으로 옮기지 않는다', () => {
  const W = 200, H = 220;
  const pts = face(100, 110, 60);
  const my = new Uint8ClampedArray(W * H * 4), ref = new Uint8ClampedArray(W * H * 4);
  const inLip = (x: number, y: number) => ((x - 100) / 17) ** 2 + ((y - 146) / 5) ** 2 < 1;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4, lip = inLip(x, y);
    my.set(lip ? [190, 130, 125, 255] : [215, 180, 160, 255], o);
    // 고른 사진: 밝은 분홍빛 피부, 장밋빛 립, 아랫입술 바로 밑(y 152~158)에 붉은 그늘
    const shade = !lip && y >= 152 && y <= 158 && Math.abs(x - 100) <= 12;
    ref.set(lip ? [215, 125, 140, 255] : shade ? [200, 150, 150, 255] : [242, 224, 220, 255], o);
  }
  const out = transferMakeup(my, W, H, pts, ref, W, H, pts, FACE_OVAL, { lip: 1, eye: 1, skin: 1 });
  const lab = new Float32Array(3), o = (155 * W + 100) * 4;
  rgb2lab(out[o], out[o + 1], out[o + 2], lab);
  const a = lab[1];
  rgb2lab(my[o], my[o + 1], my[o + 2], lab);
  assert.ok(a < lab[1] + 6, `아랫입술 밑 a ${a.toFixed(1)}, 원래 ${lab[1].toFixed(1)}`);
});
