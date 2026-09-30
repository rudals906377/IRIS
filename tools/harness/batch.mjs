// 여러 (영상, 룩) 조합을 차례로 열어 메이크업 결과를 캡처한다.
// 사용법: node batch.mjs <출력 접두어> '<[{src, look?, hair?, tattoo?, place?, tsize?, nail?, nstyle?, over?, pearl?, lstyle?, settings?, crop?}] JSON>' [대기ms]
//   src: web/public 기준 동영상 경로(예: testdata/female.webm)
//   look: 룩 이름(daily, coral, red, smoky, rose, clear) / hair: 헤어 색상표 번호(0~)
//   settings: 엔진 설정 덮어쓰기(예: {"debugLandmarks":true})
//   crop: true면 얼굴 부분만 잘라 저장
import { BASE, launch, query, waitReady } from './common.mjs';

const [, , out, combosJson, waitMs = '3000'] = process.argv;
const combos = JSON.parse(combosJson);
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
for (let i = 0; i < combos.length; i++) {
  const c = combos[i];
  await page.goto(`${BASE}/?${query({ src: c.src, delegate: 'CPU', look: c.look, hair: c.hair, tattoo: c.tattoo, place: c.place, tsize: c.tsize, nail: c.nail, nstyle: c.nstyle, over: c.over, pearl: c.pearl, lstyle: c.lstyle })}`);
  try {
    await waitReady(page, 60000, !c.nail);
  } catch {
    console.log('준비 시간 초과', i);
  }
  if (c.settings) await page.evaluate((v) => Object.assign(window.iris.settings, v), c.settings);
  await page.waitForTimeout(Number(waitMs));
  const view = page.locator('#view');
  let clip;
  if (c.crop) {
    // 얼굴 폭 기준으로 화면 좌표의 얼굴 영역을 계산
    clip = await page.evaluate(() => {
      const f = window.iris.lastFace;
      const cv = document.getElementById('view').querySelector('canvas');
      if (!f || !cv) return null;
      const r = cv.getBoundingClientRect();
      const sx = r.width / cv.width;
      const sy = r.height / cv.height;
      const xs = f.p.map((p) => p.x);
      const ys = f.p.map((p) => p.y);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      const pad = (x1 - x0) * 0.25;
      const mirrored = getComputedStyle(cv).transform.startsWith('matrix(-1');
      let left = x0 - pad, right = x1 + pad;
      if (mirrored) [left, right] = [cv.width - right, cv.width - left];
      return { x: r.left + left * sx, y: r.top + (y0 - pad) * sy, width: (right - left) * sx, height: (y1 - y0 + 2 * pad) * sy };
    });
  }
  if (clip) await page.screenshot({ path: `${out}_${i}.png`, clip });
  else await view.screenshot({ path: `${out}_${i}.png` });
  const st = await page.evaluate(() => {
    const f = window.iris.lastFace;
    const r = window.iris.lastRegions;
    return { status: document.getElementById('status')?.textContent, hands: window.iris.lastHands.length, face: !!f, conf: f ? +f.confidence.toFixed(2) : null, faceW: r ? Math.round(r.faceW) : null };
  });
  console.log(i, c.src, c.look, '|', JSON.stringify(st));
}
await browser.close();
