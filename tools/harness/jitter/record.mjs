// 합성 영상의 프레임마다 MediaPipe 원본 얼굴 점(필터 전)을 기록한다.
// 사용법: node record.mjs <출력 폴더>   (미리보기 서버에 testdata/motion.webm 이 있어야 함)
import { writeFileSync } from 'node:fs';
import { BASE, launch, waitReady } from '../common.mjs';

const out = process.argv[2];
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 800 } });
await page.goto(`${BASE}/?src=testdata/motion.webm&delegate=CPU&look=clear`);
await waitReady(page, 90000);
const frames = await page.evaluate(async () => {
  const e = window.iris;
  const v = e.source.video;
  v.pause();
  e.source.onFrame(() => {}); // 엔진이 추적기를 같이 부르지 않게
  const res = [];
  let ts = 1e6;
  for (let i = 0; i < 240; i++) {
    v.currentTime = (i + 0.5) / 30;
    await new Promise((r) => v.addEventListener('seeked', r, { once: true }));
    ts += 33.333;
    const r = e.tracker.track(v, ts);
    res.push(r.face ? r.face.map((p) => [+(p.x * v.videoWidth).toFixed(2), +(p.y * v.videoHeight).toFixed(2)]) : null);
  }
  return res;
});
writeFileSync(`${out}/raw.json`, JSON.stringify(frames));
console.log('프레임', frames.length, '놓침', frames.filter((f) => !f).length);
await browser.close();
