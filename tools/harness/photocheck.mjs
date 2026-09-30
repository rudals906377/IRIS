// 사진 따라하기 정확도 확인: 참고 사진을 시험 얼굴 영상에 적용하고(되먹임 포함),
// 사진에서 잰 값과 결과 화면에서 잰 값을 나란히 출력한다.
// 사용법: node photocheck.mjs <시험 영상> <출력 접두어> <사진 경로들...>
import { BASE, launch, waitReady } from './common.mjs';

const [, , src, out, ...refs] = process.argv;
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
const nz = (r) => {
  if (!r) return null;
  const l = 0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2];
  return r.map((v) => v / l);
};
const f2 = (r) => (r ? r.map((v) => v.toFixed(2)).join(' ') : '-');
for (let i = 0; i < refs.length; i++) {
  const ref = refs[i];
  await page.goto(`${BASE}/?src=${src}&delegate=CPU&look=clear&ref=${ref}&refmode=makeup`);
  try {
    await waitReady(page, 60000);
    // 되먹임(두 번)이 끝날 때까지
    await page.waitForFunction(() => window.irisPhoto && window.irisPhoto.last && window.irisPhoto.match.length >= 2, null, { timeout: 120000 });
  } catch {
    console.log(JSON.stringify({ ref, error: '시간 초과' }));
    continue;
  }
  await page.waitForTimeout(1500);
  const r = await page.evaluate(() => window.irisPhoto.check());
  if (!r) {
    console.log(JSON.stringify({ ref, error: '측정 실패' }));
    continue;
  }
  const line = (k) => `${k.padEnd(9)} 사진 ${f2(nz(r.photo[k]))} | 결과 ${f2(nz(r.result[k]))}`;
  const lipErr = r.photo.lip && r.result.lip ? Math.hypot(...r.photo.lip.map((v, j) => Math.pow(v, 1 / 2.2) - Math.pow(r.result.lip[j], 1 / 2.2))) / Math.sqrt(3) : null;
  console.log(`== ${ref}`);
  console.log(`  ${line('apple')}\n  ${line('mid')}\n  ${line('shadowOut')}\n  ${line('shadowIn')}`);
  console.log(`  lip       사진 ${f2(r.photo.lip)} | 결과 ${f2(r.result.lip)} | 오차 ${lipErr?.toFixed(3)}`);
  console.log(`  linerL    사진 ${r.photo.linerL?.toFixed(2)} | 결과 ${r.result.linerL?.toFixed(2)}    browL 사진 ${r.photo.browL?.toFixed(2)} | 결과 ${r.result.browL?.toFixed(2)}`);
  console.log(`  설정 ${JSON.stringify(r.settings)}`);
  if (out) await page.locator('#view').screenshot({ path: `${out}_${i}.png` });
}
await browser.close();
