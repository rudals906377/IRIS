/// <reference types="node" />
// 쇼핑몰 주소 규칙과 상품 사진 분석(합성 윤곽) 단위 테스트
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { upgradeImageUrl } from '../extension/urls.ts';
import { analyzeTop } from '../src/engine/analyze/top.ts';

test('나이키 이미지 주소를 1728px 상세 변환으로 바꾼다', () => {
  const id = '0a1b2c3d-4e5f-6789-abcd-ef0123456789';
  for (const t of ['t_web_pdp_535_v2/f_auto', 't_PDP_864_v1/f_auto,q_auto:eco', 'c_limit,w_592,f_auto/t_product_v1']) {
    assert.equal(
      upgradeImageUrl(`https://static.nike.com/a/images/${t}/${id}/M+NK+DF+TEE.png`),
      `https://static.nike.com/a/images/t_PDP_1728_v1/f_auto,q_auto:eco/${id}/M+NK+DF+TEE.png`,
    );
  }
  // 모르는 형식은 그대로
  const odd = 'https://static.nike.com/a/images/logo.png';
  assert.equal(upgradeImageUrl(odd), odd);
});

test('아디다스 이미지 주소를 가로 1200px로 바꾼다', () => {
  const tail = '68ae7ea7849b43eca70aac1e00f5146d_9366/Tee_Black_IA4845_01_laydown.jpg';
  assert.equal(upgradeImageUrl(`https://assets.adidas.com/images/w_600,f_auto,q_auto/${tail}`), `https://assets.adidas.com/images/w_1200,f_auto,q_auto/${tail}`);
  assert.equal(upgradeImageUrl(`https://assets.adidas.com/images/h_840,f_auto,q_auto,fl_lossy,c_fill,g_auto/${tail}`), `https://assets.adidas.com/images/w_1200,f_auto,q_auto/${tail}`);
  assert.equal(upgradeImageUrl('https://example.com/a.jpg'), 'https://example.com/a.jpg');
});

/** 다각형을 채운 마스크 */
function polyMask(w: number, h: number, pts: [number, number][]): Uint8Array {
  const m = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let inside = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i];
        const [xj, yj] = pts[j];
        if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
      }
      m[y * w + x] = inside ? 1 : 0;
    }
  }
  return m;
}

test('반팔 티셔츠 윤곽에서 겨드랑이·소매를 찾는다', () => {
  // 사진 좌표(가로 300, 세로 340). 착용자 왼쪽 = 사진 오른쪽.
  const tee: [number, number][] = [
    [120, 40], [150, 55], [180, 40], [230, 55], [285, 130], [250, 150], [225, 120],
    [225, 310], [75, 310], [75, 120], [50, 150], [15, 130], [70, 55],
  ];
  const a = analyzeTop(polyMask(300, 340, tee), 300, 340);
  assert.ok(a, '분석 실패');
  assert.equal(a.sleeve, 'short');
  assert.ok(a.confidence > 0.7, `신뢰도 ${a.confidence}`);
  const near = (p: { x: number; y: number }, x: number, y: number, tol: number): boolean => Math.hypot(p.x - x, p.y - y) < tol;
  assert.ok(near(a.keypoints.armpitL, 225, 120, 8), JSON.stringify(a.keypoints.armpitL));
  assert.ok(near(a.keypoints.armpitR, 75, 120, 8), JSON.stringify(a.keypoints.armpitR));
  assert.ok(a.keypoints.hemL.y > 300);
});

test('소매를 몸판 옆에 늘어뜨린 긴팔: 틈 위쪽 겨드랑이를 비율로 추정', () => {
  // 소매가 몸판 옆에 붙어 내려오다 아래쪽에서만 틈이 생긴다.
  const shirt: [number, number][] = [
    [120, 40], [150, 55], [180, 40], [225, 50], [250, 80], [262, 320], [240, 322],
    [232, 200], [228, 330], [72, 330], [68, 200], [60, 322], [38, 320], [50, 80], [75, 50],
  ];
  const a = analyzeTop(polyMask(300, 360, shirt), 300, 360);
  assert.ok(a, '분석 실패');
  assert.equal(a.sleeve, 'long');
  // 겨드랑이는 틈 꼭대기(y≈200)보다 확실히 위
  assert.ok(a.keypoints.armpitL.y < 170, JSON.stringify(a.keypoints.armpitL));
});

test('후드: 모자와 어깨가 만나는 골을 목둘레로 쓰고 모자는 목 뒤 부위로', () => {
  const hoodie: [number, number][] = [
    [110, 70], [105, 30], [125, 8], [175, 8], [195, 30], [190, 70], [230, 80], [290, 300], [262, 308],
    [226, 150], [226, 330], [74, 330], [74, 150], [38, 308], [10, 300], [70, 80],
  ];
  const a = analyzeTop(polyMask(300, 350, hoodie), 300, 350);
  assert.ok(a, '분석 실패');
  assert.ok(a.warnings.some((m) => m.includes('후드')), a.warnings.join());
  assert.ok(Math.abs(a.keypoints.neckL.y - 70) < 8 && Math.abs(a.keypoints.neckL.x - 190) < 10, JSON.stringify(a.keypoints.neckL));
  // 모자 꼭대기는 목 뒤 부위(4)
  assert.equal(a.labels[20 * 300 + 150], 4);
  assert.ok(a.confidence > 0.6, `신뢰도 ${a.confidence}`);
});

test('앞이 열린 재킷: 가운데 틈이 있어도 분석되고 틈은 라벨에서 빠진다', () => {
  const tee: [number, number][] = [
    [120, 40], [150, 55], [180, 40], [230, 55], [285, 130], [250, 150], [225, 120],
    [225, 310], [75, 310], [75, 120], [50, 150], [15, 130], [70, 55],
  ];
  const m = polyMask(300, 340, tee);
  // 가운데 세로 틈(폭 12px), 목부터 밑단까지
  for (let y = 50; y < 340; y++) for (let x = 144; x < 156; x++) m[y * 300 + x] = 0;
  const a = analyzeTop(m, 300, 340);
  assert.ok(a, '분석 실패');
  assert.equal(a.sleeve, 'short');
  assert.ok(a.confidence > 0.6, `신뢰도 ${a.confidence} ${a.warnings.join()}`);
  assert.equal(a.labels[200 * 300 + 150], 0);
  assert.equal(a.labels[200 * 300 + 120], 1);
});

test('모델 착용 사진: 상의와 하의 색이 바뀌는 곳에서 자르고 소매를 팔 쪽으로 나눈다', async () => {
  const { analyzeWorn } = await import('../src/engine/analyze/worn.ts');
  const w = 300;
  const h = 400;
  const clothes = new Float32Array(w * h);
  const rgb = new Uint8ClampedArray(w * h * 4);
  // 관절점(착용자 왼쪽 = 이미지 오른쪽), 팔은 옆으로 약간 벌림
  const lm: { x: number; y: number; visibility: number }[] = Array.from({ length: 33 }, () => ({ x: 0, y: 0, visibility: 0 }));
  const set = (i: number, x: number, y: number): void => void (lm[i] = { x, y, visibility: 0.99 });
  set(11, 200, 80); set(12, 100, 80); set(13, 235, 160); set(14, 65, 160);
  set(15, 250, 240); set(16, 50, 240); set(23, 185, 240); set(24, 115, 240);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const torso = y >= 70 && y < 250 && x >= 95 && x <= 205;
      const pants = y >= 250 && y < 380 && x >= 105 && x <= 195;
      // 반소매: 어깨에서 팔꿈치 절반까지
      const sleeveL = y >= 72 && y < 120 && x > 205 && x < 235;
      const sleeveR = y >= 72 && y < 120 && x < 95 && x > 65;
      if (torso || sleeveL || sleeveR) {
        clothes[i] = 1;
        rgb.set([200, 40, 40, 255], i * 4); // 빨간 상의
      } else if (pants) {
        clothes[i] = 1;
        rgb.set([40, 60, 160, 255], i * 4); // 파란 하의
      } else rgb.set([230, 200, 180, 255], i * 4); // 피부·배경
    }
  }
  const a = analyzeWorn(clothes, rgb, w, h, lm);
  assert.ok(a, '분석 실패');
  assert.ok(Math.abs(a.keypoints.hemL.y - 250) < 8, `밑단 ${a.keypoints.hemL.y}`);
  assert.equal(a.labels[300 * w + 150], 0, '하의는 잘라야 함');
  assert.equal(a.labels[180 * w + 150], 1);
  assert.equal(a.labels[95 * w + 225], 2, '착용자 왼쪽 소매');
  assert.equal(a.labels[95 * w + 75], 3, '착용자 오른쪽 소매');
  assert.equal(a.sleeve, 'short');
  assert.ok((a.widthScale ?? 1) > 1);
});
