// 머리카락 영역 정확도·떨림 평가. 정답은 make_hair_gt.py 가 만든 testdata/eval/hair/*.
// 앱이 염색 효과에 쓰는 머리카락 확률(정밀화 + 얼굴 피부 위 억제)을 그대로 읽어 정답과 비교한다.
//
// 사용법(미리보기 서버 4173 이 떠 있어야 함, web/dist/testdata 에 eval 폴더가 보여야 함):
//   node tools/eval/hair.mjs [이름] [주소 매개변수 JSON] [--save 폴더] [--flicker]
//   예) node tools/eval/hair.mjs 기본 '{}'
//       node tools/eval/hair.mjs 자체모델 '{"hairmodel":"models/hair-matte-256.onnx"}'
// 점수(머리 주변 정사각형 안):
//   IoU      0.5 기준 겹침(높을수록 좋음)
//   MAE      확률 평균 오차
//   경계MAE  정답 경계 근처 띠의 평균 오차(가장자리 품질, 낮을수록 좋음)
//   번짐     머리카락 아닌 곳(이마·배경)에 칠해진 양 / 머리카락 면적
//   빠짐     확실한 머리카락인데 안 칠해진 양 / 머리카락 면적
//   떨림     (--flicker) 연속 프레임 확률 변화 평균(머리카락 근처, 낮을수록 안정)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { BASE, launch, query, waitReady } from '../harness/common.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const args = process.argv.slice(2);
const flag = (k) => {
  const i = args.indexOf(k);
  if (i < 0) return null;
  const v = args[i + 1];
  args.splice(i, v && !v.startsWith('--') ? 2 : 1);
  return v && !v.startsWith('--') ? v : true;
};
const saveDir = flag('--save');
const flicker = flag('--flicker');
const [label = '기본', paramsJson = '{}'] = args;
const extra = JSON.parse(paramsJson);
const index = JSON.parse(fs.readFileSync(path.join(ROOT, 'web/public/testdata/eval/hair/index.json'), 'utf8'));
const names = Object.keys(index);

const browser = await launch();
const page = await browser.newPage({ viewport: { width: 1000, height: 900 } });
page.on('pageerror', (e) => console.error('[pageerror]', e.message));
const first = `testdata/eval/hair/${names[0]}.webm`;
await page.goto(`${BASE}/?${query({ src: first, delegate: 'CPU', look: 'clear', ...extra })}`);
await waitReady(page, 120000, false);

// 페이지 안 도우미: 영상 바꾸기, 멈춘 프레임 여러 번 다시 처리, 확률 읽기
await page.evaluate(() => {
  const iris = window.iris;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  window.evalHair = {
    async open(url) {
      await iris.source.openFile(url);
      const v = iris.source.video;
      v.pause();
      v.currentTime = 0.04;
      await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
    },
    async settle(n) {
      // 전체 분할은 몇 프레임마다 돌고 자체 모델은 한 프레임 늦게 나오므로 여러 번 다시 처리한다
      for (let i = 0; i < n; i++) {
        iris.redraw();
        await sleep(iris.tracker.hairNet ? 250 : 30);
      }
    },
    async prob() {
      iris.settings.debugSeg = 'prob';
      const p = iris.captureNext();
      iris.redraw();
      const { image } = await p;
      iris.settings.debugSeg = false;
      const out = new Float32Array(image.width * image.height);
      for (let i = 0; i < out.length; i++) out[i] = image.data[i * 4] / 255;
      return { w: image.width, h: image.height, p: out };
    },
    async gt(url) {
      const bmp = await createImageBitmap(await (await fetch(url)).blob());
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext('2d');
      ctx.drawImage(bmp, 0, 0);
      const d = ctx.getImageData(0, 0, bmp.width, bmp.height).data;
      const g = new Float32Array(bmp.width * bmp.height);
      for (let i = 0; i < g.length; i++) g[i] = d[i * 4] / 255;
      return { w: bmp.width, h: bmp.height, g };
    },
    // 상자 흐림(적분 영상)으로 정답 경계 띠를 만든다
    band(bin, w, h, r) {
      const S = new Float64Array((w + 1) * (h + 1));
      for (let y = 0; y < h; y++) {
        let row = 0;
        for (let x = 0; x < w; x++) {
          row += bin[y * w + x];
          S[(y + 1) * (w + 1) + x + 1] = S[y * (w + 1) + x + 1] + row;
        }
      }
      const out = new Uint8Array(w * h);
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - r);
        const y1 = Math.min(h, y + r + 1);
        for (let x = 0; x < w; x++) {
          const x0 = Math.max(0, x - r);
          const x1 = Math.min(w, x + r + 1);
          const s = S[y1 * (w + 1) + x1] - S[y0 * (w + 1) + x1] - S[y1 * (w + 1) + x0] + S[y0 * (w + 1) + x0];
          const m = s / ((y1 - y0) * (x1 - x0));
          out[y * w + x] = m > 0.02 && m < 0.98 ? 1 : 0;
        }
      }
      return out;
    },
    score(P, G, rect) {
      const { w, h, p } = P;
      const g = G.g;
      const [rx, ry, rs] = rect;
      let inter = 0, uni = 0, mae = 0, n = 0, fp = 0, fn = 0, area = 0, bm = 0, bn = 0;
      const bin = new Uint8Array(w * h);
      for (let i = 0; i < g.length; i++) bin[i] = g[i] > 0.5 ? 1 : 0;
      const band = this.band(bin, w, h, Math.max(2, Math.round(rs * 0.02)));
      for (let y = ry; y < Math.min(h, ry + rs); y++) {
        for (let x = rx; x < Math.min(w, rx + rs); x++) {
          const i = y * w + x;
          const a = p[i] > 0.5;
          const b = g[i] > 0.5;
          if (a && b) inter++;
          if (a || b) uni++;
          const e = Math.abs(p[i] - g[i]);
          mae += e;
          n++;
          if (b) area++;
          if (g[i] < 0.05) fp += p[i];
          if (g[i] > 0.95) fn += 1 - p[i];
          if (band[i]) {
            bm += e;
            bn++;
          }
        }
      }
      return { iou: uni ? inter / uni : 1, mae: mae / n, edge: bn ? bm / bn : 0, spill: area ? fp / area : 0, miss: area ? fn / area : 0 };
    },
    png(P, G) {
      // 원본 위에 정답(초록)·예측(빨강)
      const v = iris.source.video;
      const c = new OffscreenCanvas(P.w * 3, P.h);
      const ctx = c.getContext('2d');
      ctx.drawImage(v, 0, 0, P.w, P.h);
      const id = ctx.createImageData(P.w, P.h);
      const id2 = ctx.createImageData(P.w, P.h);
      for (let i = 0; i < P.p.length; i++) {
        id.data[i * 4] = id.data[i * 4 + 1] = id.data[i * 4 + 2] = G.g[i] * 255;
        id.data[i * 4 + 3] = 255;
        id2.data[i * 4] = id2.data[i * 4 + 1] = id2.data[i * 4 + 2] = P.p[i] * 255;
        id2.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(id, P.w, 0);
      ctx.putImageData(id2, P.w * 2, 0);
      return c.convertToBlob({ type: 'image/jpeg', quality: 0.85 }).then((b) => b.arrayBuffer()).then((a) => Array.from(new Uint8Array(a)));
    },
  };
});

const rows = [];
for (const name of names) {
  const it = index[name];
  const r = await page.evaluate(
    async ({ name, rect, save }) => {
      const E = window.evalHair;
      await E.open(`testdata/eval/hair/${name}.webm`);
      await E.settle(8);
      const P = await E.prob();
      const G = await E.gt(`testdata/eval/hair/${name}_gt.png`);
      if (P.w !== G.w || P.h !== G.h) return { error: `크기 다름 ${P.w}x${P.h} vs ${G.w}x${G.h}` };
      const s = E.score(P, G, rect);
      if (save) s.png = await E.png(P, G);
      return s;
    },
    { name, rect: it.rect, save: !!saveDir },
  );
  if (r.png) {
    fs.mkdirSync(saveDir, { recursive: true });
    fs.writeFileSync(path.join(saveDir, `${name}.jpg`), Buffer.from(r.png));
    delete r.png;
  }
  rows.push({ name, ...r });
  console.error(name, JSON.stringify(r, (k, v) => (typeof v === 'number' ? +v.toFixed(3) : v)));
}

let flick = null;
if (flicker) {
  // 거의 가만히 있는 사람 영상 3개에서 연속 24프레임(약 1초)
  const clips = ['eval/flicker_4844.webm', 'eval/flicker_4863.webm', 'eval/flicker_11589.webm'];
  const vals = [];
  for (const clip of clips) {
    const v = await page.evaluate(async (url) => {
      const iris = window.iris;
      const E = window.evalHair;
      await E.open(url);
      const vid = iris.source.video;
      const t0 = 0.2;
      let prev = null;
      let sum = 0;
      let cnt = 0;
      for (let k = 0; k < 30; k++) {
        vid.currentTime = t0 + k / 24;
        await new Promise((r) => vid.addEventListener('seeked', r, { once: true }));
        await E.settle(k === 0 ? 8 : 1);
        const P = await E.prob();
        if (prev && k >= 6) {
          let s = 0;
          let n = 0;
          for (let i = 0; i < P.p.length; i++) {
            if (Math.max(P.p[i], prev[i]) > 0.1) {
              s += Math.abs(P.p[i] - prev[i]);
              n++;
            }
          }
          if (n) {
            sum += s / n;
            cnt++;
          }
        }
        prev = P.p;
      }
      return cnt ? sum / cnt : null;
    }, `testdata/${clip}`);
    vals.push(v);
    console.error('떨림', clip, v?.toFixed(4));
  }
  const ok = vals.filter((x) => x !== null);
  flick = ok.length ? ok.reduce((a, b) => a + b, 0) / ok.length : null;
}
await browser.close();

const good = rows.filter((r) => !r.error);
const mean = (k) => good.reduce((a, r) => a + r[k], 0) / good.length;
const summary = { 이름: label, 매개변수: extra, 장수: good.length, IoU: mean('iou'), MAE: mean('mae'), 경계MAE: mean('edge'), 번짐: mean('spill'), 빠짐: mean('miss'), 떨림: flick };
const fmt = (v) => (typeof v === 'number' ? v.toFixed(3) : v);
console.log(Object.entries(summary).map(([k, v]) => `${k}=${typeof v === 'object' && v !== null ? JSON.stringify(v) : fmt(v)}`).join('  '));
const outDir = path.join(ROOT, 'tools/eval/results');
fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(path.join(outDir, `hair-${label}.json`), JSON.stringify({ summary, rows }, null, 1));
