// 참고 사진들을 차례로 분석해 측정값(debug)을 JSON 줄로 출력한다. 사용법: node refdebug.mjs <refmode> <사진 경로들...>
import { BASE, launch, waitReady } from './common.mjs';
const [, , mode, ...refs] = process.argv;
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
for (const ref of refs) {
  await page.goto(`${BASE}/?src=testdata/face/00055_00.webm&delegate=CPU&look=clear&ref=${ref}&refmode=${mode}`);
  try {
    await waitReady(page, 60000);
    await page.waitForFunction(() => window.irisPhoto && window.irisPhoto.last, null, { timeout: 90000 });
  } catch {
    console.log(JSON.stringify({ ref, error: '시간 초과' }));
    continue;
  }
  const r = await page.evaluate(() => {
    const s = window.irisPhoto.last;
    const hx = (c) => c ? '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('') : null;
    const m = s.makeup;
    return { summary: s.summary, debug: m?.debug, lip: m?.lip && { color: hx(m.lip.color), amount: m.lip.amount, gloss: +m.lip.gloss.toFixed(2), style: m.lip.style },
      blush: m?.blush && { color: hx(m.blush.color), amount: m.blush.amount }, shadow: m?.shadow && { color: hx(m.shadow.color), amount: m.shadow.amount },
      liner: m?.liner && m.liner.amount, brow: m?.brow && { color: hx(m.brow.color), amount: m.brow.amount },
      hair: s.hair && { color: hx(s.hair.color), tip: hx(s.hair.tip) }, nail: s.nail && { color: hx(s.nail.color), style: s.nail.style }, tattoo: !!s.tattoo };
  });
  console.log(JSON.stringify({ ref, ...r }));
}
await browser.close();
