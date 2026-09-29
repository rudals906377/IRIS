// 여러 (영상, 상품) 조합을 차례로 열어 착용 결과를 캡처한다.
// 사용법: node batch.mjs <출력 접두어> '<[{src, product?, img?, settings?}] JSON>' [대기ms]
//   src: web/public 기준 동영상 경로(예: testdata/still_00121_00.webm)
//   product: 카탈로그 상품 id / img: 자동 분석할 상품 사진 경로
//   settings: 엔진 설정 덮어쓰기(예: {"debugGarment":1,"debugLandmarks":true})
import { BASE, launch, query, waitReady } from './common.mjs';

const [, , out, combosJson, waitMs = '5000'] = process.argv;
const combos = JSON.parse(combosJson);
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
for (let i = 0; i < combos.length; i++) {
  const c = combos[i];
  await page.goto(`${BASE}/?${query({ src: c.src, delegate: 'CPU', product: c.product, img: c.img })}`);
  try {
    await waitReady(page, 60000);
  } catch {
    console.log('준비 시간 초과', i);
  }
  if (c.settings) await page.evaluate((v) => Object.assign(window.iris.settings, v), c.settings);
  await page.waitForTimeout(Number(waitMs));
  await page.locator('#view').screenshot({ path: `${out}_${i}.png` });
  const st = await page.evaluate(() => {
    const b = window.iris.lastBody;
    return { status: document.getElementById('status').textContent, body: !!b, w: b ? Math.round(b.shoulderW) : null, axis: b ? Math.round(b.axisLen) : null };
  });
  console.log(i, c.src, c.product ?? c.img, '|', JSON.stringify(st));
}
await browser.close();
