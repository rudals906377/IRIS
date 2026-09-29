// 착용 리그 내부 수치(상품 기준점, 소매 곡면 좌표 범위, 메쉬 경계, 관절점)를 JSON으로 출력한다.
// 사용법: node rigprobe.mjs '<앱 주소 인자>'   예: 'src=testdata/still_00121_00.webm&img=testdata/cloth/09266_00.jpg'
import { BASE, launch, waitReady } from './common.mjs';

const [, , q] = process.argv;
const browser = await launch();
const page = await browser.newPage();
await page.goto(`${BASE}/?${q}&delegate=CPU`);
await waitReady(page);
await page.waitForTimeout(3000);
const r = await page.evaluate(() => {
  const e = window.iris;
  const rig = e.garment.rig;
  const b = e.lastBody;
  const R = (v) => Math.round(v);
  const bbox = (m) => {
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (let i = 0; i < m.vertexCount; i++) {
      x0 = Math.min(x0, m.dst[i * 2]); y0 = Math.min(y0, m.dst[i * 2 + 1]);
      x1 = Math.max(x1, m.dst[i * 2]); y1 = Math.max(y1, m.dst[i * 2 + 1]);
    }
    return [x0, y0, x1, y1].map(R);
  };
  return {
    size: [rig.asset.width, rig.asset.height],
    kp: Object.fromEntries(Object.entries(rig.asset.kp).map(([k, v]) => [k, [R(v.x), R(v.y)]])),
    sleeves: rig.sleeves.map((s) => ({ part: s.mesh.partId, sMin: Math.min(...s.s).toFixed(2), sMax: Math.max(...s.s).toFixed(2), vMin: Math.min(...s.v).toFixed(2), vMax: Math.max(...s.v).toFixed(2), bbox: bbox(s.mesh) })),
    torso: rig.torso ? bbox(rig.torso) : null,
    body: { w: R(b.shoulderW), axis: R(b.axisLen), sL: [R(b.p[11].x), R(b.p[11].y)], sR: [R(b.p[12].x), R(b.p[12].y)], eL: [R(b.p[13].x), R(b.p[13].y)], wL: [R(b.p[15].x), R(b.p[15].y)] },
  };
});
console.log(JSON.stringify(r));
await browser.close();
