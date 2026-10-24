// 사진 따라하기 색 정확도 평가: 참고 사진(testdata/ref/makeup/00~20)을 시험 얼굴에 적용(되먹임 포함)하고
// 사진에서 잰 색과 결과 화면에서 잰 색의 차이를 부위별로 평균한다.
// 사용법: node tools/eval/makeup.mjs [이름] [시험 얼굴 영상] [개수]
//   예) node tools/eval/makeup.mjs 기본 testdata/face/00055_00.webm 21
// 점수(낮을수록 좋음):
//   입술      감마 공간 RGB 거리
//   볼·섀도   밝기로 나눈 색 비율의 거리(색조·채도 차이, 밝기는 조명 차이라 뺀다)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BASE, launch, waitReady } from '../harness/common.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const [label = '기본', src = 'testdata/face/00055_00.webm', countStr = '21'] = process.argv.slice(2);
const count = Number(countStr);
const nz = (r) => {
  if (!r) return null;
  const l = 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2];
  return l > 1e-4 ? r.map((v) => v / l) : null;
};
const dist = (a, b) => (a && b ? Math.hypot(...a.map((v, j) => v - b[j])) / Math.sqrt(3) : null);
const gam = (r) => r && r.map((v) => Math.pow(Math.max(0, v), 1 / 2.2));

const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
const rows = [];
for (let i = 0; i < count; i++) {
  const ref = `testdata/ref/makeup/${String(i).padStart(2, '0')}.png`;
  if (!fs.existsSync(path.join(ROOT, 'web/public', ref))) continue;
  await page.goto(`${BASE}/?src=${src}&delegate=CPU&look=clear&ref=${ref}&refmode=makeup`);
  let r = null;
  try {
    await waitReady(page, 90000);
    await page.waitForFunction(() => window.irisPhoto && window.irisPhoto.last && window.irisPhoto.match.length >= 4, null, { timeout: 150000 });
    await page.waitForTimeout(1500);
    r = await page.evaluate(() => window.irisPhoto.check());
  } catch {
    r = null;
  }
  if (!r) {
    rows.push({ ref, error: '측정 실패' });
    console.error(ref, '측정 실패');
    continue;
  }
  const row = {
    ref,
    lip: dist(gam(r.photo.lip), gam(r.result.lip)),
    apple: dist(nz(r.photo.apple), nz(r.result.apple)),
    mid: dist(nz(r.photo.mid), nz(r.result.mid)),
    shadowOut: dist(nz(r.photo.shadowOut), nz(r.result.shadowOut)),
    shadowIn: dist(nz(r.photo.shadowIn), nz(r.result.shadowIn)),
  };
  rows.push(row);
  console.error(ref, JSON.stringify(row, (k, v) => (typeof v === 'number' ? +v.toFixed(3) : v)));
}
await browser.close();
const mean = (k) => {
  const v = rows.map((r) => r[k]).filter((x) => typeof x === 'number');
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};
const worst = (k) => rows.filter((r) => typeof r[k] === 'number').sort((a, b) => b[k] - a[k]).slice(0, 3).map((r) => `${r.ref.slice(-6, -4)}:${r[k].toFixed(2)}`).join(' ');
const summary = { 이름: label, 얼굴: src, 장수: rows.filter((r) => !r.error).length, 입술: mean('lip'), 볼앞: mean('apple'), 볼중간: mean('mid'), 섀도바깥: mean('shadowOut'), 섀도안: mean('shadowIn') };
console.log(Object.entries(summary).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`).join('  '));
console.log(`  가장 나쁜 볼앞 ${worst('apple')} / 섀도바깥 ${worst('shadowOut')}`);
const outDir = path.join(ROOT, 'tools/eval/results');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `makeup-${label}.json`), JSON.stringify({ summary, rows }, null, 1));
