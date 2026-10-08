// 염색 자연스러움·색 정확도 평가: 실제 머리카락 사진(hair_refs.json)의 평균색으로 시험 얼굴을 염색하고,
// 결과 머리카락의 색·명암 분포가 그 실제 사진과 얼마나 같은지 잰다. 머리카락 영역은 평가 정답(eval/hair)의 확실한 안쪽.
// 사용법: node tools/eval/haircolor.mjs [이름] [주소 매개변수 JSON] [--save 폴더]
// 점수(낮을수록 좋음):
//   색오차   결과 평균색과 목표색의 Lab 거리(ΔE)
//   결오차   밝기 변동계수 비(결과/실제)의 로그 절댓값: 0이면 실제 그 색 머리처럼 결·명암이 산다
//   폭오차   밝기 10·90 백분위 폭(그늘~윤기)의 로그 비 절댓값
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BASE, launch, query, waitReady } from '../harness/common.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const si = args.indexOf('--save');
const saveDir = si >= 0 ? args.splice(si, 2)[1] : null;
const [label = '기본', paramsJson = '{}'] = args;
const extra = JSON.parse(paramsJson);
const refs = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools/eval/results/hair_refs.json'), 'utf8'));
const index = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/public/testdata/eval/hair/index.json'), 'utf8'));
// 색이 고루 섞이게: 밝은 금발·회색·주황·빨강·갈색·검정
const REFS = ['01', '22', '29', '07', '27', '20', '13', '06', '24', '12', '15'].filter((k) => refs[k]);
// 원래 머리색이 다른 사람들(검정·갈색·금발)
const FRAMES = ['webcam_10443_0122', 'webcam_4863_0213', 'webcam_11589_0573', 'webcam_15909_0203', 'webcam_6083_0154'].filter((n) => index[n]);

const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
await page.goto(`${BASE}/?${query({ src: `testdata/eval/hair/${FRAMES[0]}.webm`, delegate: 'CPU', look: 'clear', ...extra })}`);
await waitReady(page, 120000, false);
await page.evaluate(() => {
  const iris = window.iris;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const toLab = (c) => {
    const lin = c.map((v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
    const X = (0.4124 * lin[0] + 0.3576 * lin[1] + 0.1805 * lin[2]) / 0.9505;
    const Y = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    const Z = (0.0193 * lin[0] + 0.1192 * lin[1] + 0.9505 * lin[2]) / 1.089;
    const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
    return [116 * f(Y) - 16, 500 * (f(X) - f(Y)), 200 * (f(Y) - f(Z))];
  };
  window.evalColor = {
    async open(url, gtUrl) {
      await iris.source.openFile(url);
      const v = iris.source.video;
      v.pause();
      v.currentTime = 0.04;
      await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
      for (let i = 0; i < 8; i++) {
        iris.redraw();
        await sleep(iris.tracker.hairNet ? 250 : 30);
      }
      // 정답 머리카락의 확실한 안쪽(0.95 이상, 가장자리 3px 깎음)
      const bmp = await createImageBitmap(await (await fetch(gtUrl)).blob());
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext('2d');
      ctx.drawImage(bmp, 0, 0);
      const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const w = bmp.width;
      const h = bmp.height;
      const core = new Uint8Array(w * h);
      const r = 3;
      for (let y = r; y < h - r; y++)
        for (let x = r; x < w - r; x++) {
          let ok = 1;
          for (let dy = -r; dy <= r && ok; dy += r) for (let dx = -r; dx <= r && ok; dx += r) if (d[((y + dy) * w + x + dx) * 4] < 242) ok = 0;
          core[y * w + x] = ok;
        }
      this.core = core;
    },
    async render(color) {
      iris.hair = { color, amount: 1 };
      iris.redraw();
      await sleep(30);
      const p = iris.captureNext();
      iris.redraw();
      const { image } = await p;
      const L = [];
      const sum = [0, 0, 0];
      for (let i = 0; i < this.core.length; i++) {
        if (!this.core[i]) continue;
        const r = image.data[i * 4] / 255;
        const g = image.data[i * 4 + 1] / 255;
        const b = image.data[i * 4 + 2] / 255;
        sum[0] += r;
        sum[1] += g;
        sum[2] += b;
        L.push(0.299 * r + 0.587 * g + 0.114 * b);
      }
      if (L.length < 300) return null;
      const n = L.length;
      const mean = sum.map((v) => v / n);
      const ml = L.reduce((a, b) => a + b, 0) / n;
      const sd = Math.sqrt(L.reduce((a, b) => a + (b - ml) * (b - ml), 0) / n);
      L.sort((a, b) => a - b);
      return { mean, cov: sd / ml, p10: L[Math.floor(n * 0.1)] / ml, p90: L[Math.floor(n * 0.9)] / ml, n, lab: toLab(mean), image };
    },
    labOf: toLab,
    async png(image) {
      const c = new OffscreenCanvas(image.width, image.height);
      c.getContext('2d').putImageData(image, 0, 0);
      const b = await c.convertToBlob({ type: 'image/jpeg', quality: 0.85 });
      return Array.from(new Uint8Array(await b.arrayBuffer()));
    },
  };
});

const rows = [];
for (const fr of FRAMES) {
  await page.evaluate(({ fr }) => window.evalColor.open(`testdata/eval/hair/${fr}.webm`, `testdata/eval/hair/${fr}_gt.png`), { fr });
  for (const k of REFS) {
    const ref = refs[k];
    const r = await page.evaluate(
      async ({ color, save }) => {
        const E = window.evalColor;
        const s = await E.render(color);
        if (!s) return null;
        const out = { mean: s.mean, cov: s.cov, p10: s.p10, p90: s.p90, lab: s.lab, refLab: E.labOf(color) };
        if (save) out.png = await E.png(s.image);
        return out;
      },
      { color: ref.mean, save: !!saveDir },
    );
    if (!r) {
      console.error(fr, k, '머리카락 영역 부족');
      continue;
    }
    const dE = Math.hypot(r.lab[0] - r.refLab[0], r.lab[1] - r.refLab[1], r.lab[2] - r.refLab[2]);
    const covErr = Math.abs(Math.log(r.cov / ref.cov));
    const spanErr = Math.abs(Math.log((r.p90 - r.p10) / (ref.p90 - ref.p10)));
    if (r.png) {
      fs.mkdirSync(saveDir, { recursive: true });
      fs.writeFileSync(path.join(saveDir, `${fr}_${k}.jpg`), Buffer.from(r.png));
    }
    rows.push({ frame: fr, ref: k, dE, covErr, spanErr, cov: r.cov, refCov: ref.cov });
    console.error(fr, k, `ΔE ${dE.toFixed(1)}  결 ${r.cov.toFixed(2)}/${ref.cov.toFixed(2)}  폭 ${(r.p90 - r.p10).toFixed(2)}/${(ref.p90 - ref.p10).toFixed(2)}`);
  }
}
await browser.close();
const mean = (k) => rows.reduce((a, r) => a + r[k], 0) / rows.length;
const summary = { 이름: label, 매개변수: extra, 쌍: rows.length, 색오차: mean('dE'), 결오차: mean('covErr'), 폭오차: mean('spanErr') };
console.log(Object.entries(summary).map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : JSON.stringify(v)}`).join('  '));
const outDir = path.join(ROOT, 'tools/eval/results');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `haircolor-${label}.json`), JSON.stringify({ summary, rows }, null, 1));
