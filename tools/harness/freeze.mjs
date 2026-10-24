// 영상을 지정 시각에 멈추고, 설정 조합마다 같은 프레임을 다시 합성해 캡처한다(정확한 전후 비교용).
// 사용법: node freeze.mjs '<앱 주소 인자>' <출력 접두어> <멈출 시각(초)> '<설정 JSON 배열>' [crop x,y,w,h]
import { BASE, launch, waitReady } from './common.mjs';

const [, , q, out, atSec, variantsJson, crop] = process.argv;
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${BASE}/?${q}`);
await waitReady(page);
await page.evaluate(async (t) => {
  const v = window.iris.source.video;
  v.pause();
  v.currentTime = t;
  await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
}, Number(atSec));
const variants = JSON.parse(variantsJson);
for (let i = 0; i < variants.length; i++) {
  await page.evaluate((v) => {
    Object.assign(window.iris.settings, v);
    for (let k = 0; k < 6; k++) window.iris.redraw();
  }, variants[i]);
  const opts = { path: `${out}_${i}.png` };
  if (crop) {
    const [x, y, width, height] = crop.split(',').map(Number);
    opts.clip = { x, y, width, height };
  }
  await page.screenshot(opts);
}
await browser.close();
