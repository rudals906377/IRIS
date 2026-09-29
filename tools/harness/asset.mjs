// 상품 사진 자동 분석 결과(잘라 낸 옷 + 부위 라벨 색 + 기준점)를 그림으로 저장한다.
// 사용법: node asset.mjs <출력 png> <사진 경로(web/public 기준)>...
//   예: node asset.mjs /tmp/a.png testdata/human/00055_00.jpg testdata/cloth/09163_00.jpg
import { BASE, launch, query } from './common.mjs';

const [, , out, ...imgs] = process.argv;
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const tiles = [];
for (const img of imgs) {
  await page.goto(`${BASE}/?${query({ img, delegate: 'CPU' })}`);
  await page.waitForFunction((n) => window.iris?.lastAsset || /인식하지 못|실패|받지 못/.test(document.getElementById('status').textContent), null, { timeout: 120000 }).catch(() => {});
  const r = await page.evaluate(() => {
    const a = window.iris.lastAsset;
    const status = document.getElementById('status').textContent;
    if (!a) return { status };
    const c = document.createElement('canvas');
    c.width = a.width;
    c.height = a.height;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#ddd';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(a.image, 0, 0);
    const im = ctx.getImageData(0, 0, c.width, c.height);
    const col = { 1: [80, 80, 255], 2: [255, 60, 60], 3: [60, 200, 60], 4: [255, 200, 0] };
    for (let i = 0; i < a.labels.length; i++) {
      const k = col[a.labels[i]];
      if (!k) continue;
      for (let j = 0; j < 3; j++) im.data[i * 4 + j] = im.data[i * 4 + j] * 0.6 + k[j] * 0.4;
    }
    ctx.putImageData(im, 0, 0);
    ctx.font = `${Math.round(c.width / 40)}px sans-serif`;
    for (const [n, p] of Object.entries(a.kp)) {
      ctx.fillStyle = '#f0f';
      ctx.beginPath();
      ctx.arc(p.x, p.y, c.width / 150, 0, 7);
      ctx.fill();
      ctx.fillStyle = '#000';
      ctx.fillText(n, p.x + 6, p.y - 4);
    }
    return { status, url: c.toDataURL('image/png') };
  });
  console.log(img, '|', r.status);
  if (r.url) tiles.push(r.url);
}
if (tiles.length) {
  const sheet = await page.evaluate(async (urls) => {
    const ims = await Promise.all(urls.map((u) => new Promise((ok) => { const i = new Image(); i.onload = () => ok(i); i.src = u; })));
    const H = 600;
    const ws = ims.map((i) => Math.round((i.width * H) / i.height));
    const c = document.createElement('canvas');
    c.width = ws.reduce((a, b) => a + b, 0);
    c.height = H;
    const ctx = c.getContext('2d');
    let x = 0;
    ims.forEach((i, k) => { ctx.drawImage(i, x, 0, ws[k], H); x += ws[k]; });
    return c.toDataURL('image/png');
  }, tiles);
  const { writeFileSync } = await import('node:fs');
  writeFileSync(out, Buffer.from(sheet.split(',')[1], 'base64'));
}
await browser.close();
