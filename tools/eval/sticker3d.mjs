// 생성 머리 "실시간에 입히기"가 고개를 돌려도 따라오는지 눈으로 확인하는 시험.
// 고개를 돌리는 영상(male-turn.webm)의 정면 장면을 찍어 가짜 '생성 머리'(줄무늬로 칠한 머리 자리)를 만들고,
// 여러 장면에서 3D 휘기와 예전 평면 스티커를 나란히 저장한다.
// 사용법: tools/eval/with-preview.sh node tools/eval/sticker3d.mjs 저장폴더
import fs from 'node:fs';
import { BASE, launch, query, waitReady } from '../harness/common.mjs';

const out = process.argv[2] ?? 'sticker3d';
fs.mkdirSync(out, { recursive: true });
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 800 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
page.on('console', (m) => m.type() === 'warning' && console.log('[warn]', m.text()));
await page.goto(`${BASE}/?${query({ src: 'testdata/male-turn.webm', delegate: 'CPU', look: 'clear' })}`);
await waitReady(page, 120000, true);

const info = await page.evaluate(async () => {
  const iris = window.iris;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const v = iris.source.video;
  v.pause();
  const seek = async (t) => {
    v.currentTime = t;
    await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
    for (let i = 0; i < 6; i++) { iris.redraw(); await sleep(60); }
  };
  window.__seek = seek;
  // 정면에 가장 가까운 장면 찾기(코끝이 양 볼 가운데)
  let best = 0, bestErr = 1e9;
  for (let t = 0.2; t < Math.min(v.duration, 8); t += 0.4) {
    await seek(t);
    const f = iris.lastFace ?? null;
    const cap = await Promise.race([iris.captureNext(), sleep(1500).then(() => null)]);
    iris.redraw();
    const face = cap?.face ?? f;
    if (!face) continue;
    const p = face.p;
    const e = Math.abs((p[1].x - p[234].x) - (p[454].x - p[1].x)) / Math.abs(p[454].x - p[234].x);
    if (e < bestErr) { bestErr = e; best = t; }
  }
  await seek(best);
  const pc = iris.captureNext();
  iris.redraw();
  const cap = await pc;
  const { image, face } = cap;
  const W = image.width, H = image.height;
  const c = new OffscreenCanvas(W, H);
  const ctx = c.getContext('2d');
  ctx.putImageData(image, 0, 0);
  // 줄무늬 빨강으로 칠한 '생성 머리'
  ctx.globalAlpha = 0.6;
  for (let x = -H; x < W; x += 16) { ctx.fillStyle = (x / 16) % 2 ? '#e0306a' : '#ffd040'; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x + 8, 0); ctx.lineTo(x + 8 + H, H); ctx.lineTo(x + H, H); ctx.fill(); }
  ctx.globalAlpha = 1;
  // 머리 자리: 얼굴 위·옆 큰 타원에서 얼굴 윤곽(눈썹 아래)을 뺀 곳
  const p = face.p;
  const fw = Math.hypot(p[454].x - p[234].x, p[454].y - p[234].y);
  const m = new OffscreenCanvas(W, H);
  const mc = m.getContext('2d');
  mc.fillStyle = '#000'; mc.fillRect(0, 0, W, H);
  mc.fillStyle = '#fff';
  mc.beginPath(); mc.ellipse(p[151].x, p[151].y + fw * 0.05, fw * 0.72, fw * 0.75, 0, 0, Math.PI * 2); mc.fill();
  mc.fillStyle = '#000';
  const CUT = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300];
  mc.beginPath(); CUT.forEach((i, k) => (k ? mc.lineTo(p[i].x, p[i].y) : mc.moveTo(p[i].x, p[i].y))); mc.fill();
  const url = async (cv) => { const b = await cv.convertToBlob({ type: 'image/png' }); return await new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(b); }); };
  const facePts = p.map((q, i) => ({ x: q.x, y: q.y, z: face.z ? face.z[i] : undefined }));
  window.irisGen.setLive({ category: 'hair', image: await url(c), mask: await url(m), facePts });
  await window.irisGen.live();
  return { best, bestErr, hasZ: !!face.z, dur: v.duration, mesh: !!window.irisGen.sticker?.mesh };
});
console.log(JSON.stringify(info));

const shot = async (name) => {
  const view = await page.$('#view');
  await view.screenshot({ path: `${out}/${name}.png` });
};
const times = [];
const step = Number(process.env.STEP ?? 2.5);
for (let t = 0.5; t < info.dur - 0.3; t += step) times.push(+t.toFixed(2));
for (const t of times) {
  await page.evaluate((t) => window.__seek(t), t);
  await shot(`t${t}_3d`);
  const mesh = await page.evaluate(() => { const s = window.irisGen.sticker; const m = s.mesh; s.mesh = null; return m; }, null);
  await page.evaluate(() => window.__seek(window.iris.source.video.currentTime));
  await shot(`t${t}_2d`);
  await page.evaluate(() => { window.irisGen.live(); });
  await page.waitForTimeout(400);
}
await browser.close();
console.log('saved', times.length * 2, 'to', out);
