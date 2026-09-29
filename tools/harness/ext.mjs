// 확장 프로그램을 설치한 크롬으로 쇼핑몰 페이지를 열고, 툴바 클릭을 흉내 내 착용 창까지 확인한다.
// 사용법: node ext.mjs <쇼핑몰 상세페이지 주소> [가짜 웹캠 y4m 경로] [출력 png]
//   먼저: cd web && npm run build:ext   (web/dist-ext 생성)
//   가짜 웹캠: ffmpeg -loop 1 -t 2 -i 인물.jpg -vf "scale=640:-2,fps=15" -pix_fmt yuv420p person.y4m
import { chromium } from 'playwright';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CHROME } from './common.mjs';

const [, , shopUrl, y4m, outPng = 'ext.png'] = process.argv;
const ext = resolve(new URL('../../web/dist-ext', import.meta.url).pathname);
const args = [
  '--headless=new', `--disable-extensions-except=${ext}`, `--load-extension=${ext}`,
  '--ignore-certificate-errors', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
  '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
];
if (y4m) args.push(`--use-file-for-fake-video-capture=${y4m}`);
const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), 'iris-')), {
  executablePath: CHROME, headless: true, args, viewport: { width: 1180, height: 860 },
});
const logs = [];
ctx.on('page', (p) => {
  p.on('console', (m) => { if (m.type() === 'error') logs.push(`[${p.url().slice(0, 50)}] ${m.text()}`); });
  p.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
});
const sw = ctx.serviceWorkers()[0] ?? (await ctx.waitForEvent('serviceworker', { timeout: 20000 }));
const shop = await ctx.newPage();
await shop.goto(shopUrl, { waitUntil: 'load', timeout: 60000 });
await shop.waitForTimeout(3000);
const found = await sw.evaluate(async (url) => {
  const tabs = await chrome.tabs.query({});
  const tab = tabs.find((t) => t.url === url) ?? tabs.find((t) => t.url?.startsWith(new URL(url).origin));
  return globalThis.irisTest.tryTab(tab.id);
}, shop.url());
console.log('수집된 후보:', JSON.stringify(found, null, 1));
const tryon = await ctx.waitForEvent('page', { predicate: (p) => p.url().includes('index.html'), timeout: 20000 });
await tryon.waitForFunction(() => window.iris && window.iris.metrics.framesTotal > 3 && window.iris.product, null, { timeout: 120000 }).catch(() => console.log('착용 준비 시간 초과'));
await tryon.waitForTimeout(6000);
console.log(JSON.stringify(await tryon.evaluate(() => ({
  status: document.getElementById('status').textContent,
  product: window.iris.product?.name,
  selected: document.querySelector('.product.selected')?.dataset.id,
  fit: [...document.querySelectorAll('.product[data-fit]')].map((e) => `${e.dataset.id}:${e.dataset.fit}`),
  body: !!window.iris.lastBody,
})), null, 1));
await tryon.screenshot({ path: outPng });
console.log(logs.slice(-15).join('\n'));
await ctx.close();
