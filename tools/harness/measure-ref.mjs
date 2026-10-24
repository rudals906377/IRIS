// 참고 사진(web/public/testdata/ref/*)에서 실제 색을 잰다. 저작권 있는 사진은 저장소에 넣지 않고, 뽑은 색 값만 쓴다.
//   메이크업: 입술 색, 볼(블러셔) 색과 이마·턱 피부색 → 앱의 기준 피부(#d1a38a)·조명으로 환산한 목표색
//   헤어: 머리카락 평균색, 위(뿌리)·아래(끝) 색, 밝은 가닥 비율(하이라이트)
//   네일: 손 점으로 손가락 끝 마디 위(손톱 가운데)의 색
// 사용법: node measure-ref.mjs <makeup|hair|nail> <개수> > 결과.json
import { BASE, launch, waitReady } from './common.mjs';

const [, , kind, countStr] = process.argv;
const count = Number(countStr);
const browser = await launch();
const page = await browser.newPage({ viewport: { width: 900, height: 1100 } });
const out = [];
for (let i = 0; i < count; i++) {
  const id = String(i).padStart(2, '0');
  await page.goto(`${BASE}/?src=testdata/ref/${kind}/${id}.webm&delegate=CPU&look=clear&refine=0${kind === 'nail' ? '&nail=0&nstyle=solid' : ''}`);
  try {
    await waitReady(page, 60000, kind === 'makeup');
  } catch {
    out.push({ id, error: kind === 'makeup' ? '얼굴 못 찾음' : '시간 초과' });
    continue;
  }
  await page.waitForTimeout(1500);
  const r = await page.evaluate((kind) => {
    const v = window.iris.source.video;
    const W = v.videoWidth;
    const H = v.videoHeight;
    const cv = document.createElement('canvas');
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(v, 0, 0);
    const img = ctx.getImageData(0, 0, W, H).data;
    const lin = (c) => Math.pow(c / 255, 2.2);
    // 다각형 안 픽셀의 선형 평균(밝기 위아래 10%는 반사광·그늘로 보고 뺌)
    const polyMean = (polys, holes = []) => {
      const m = document.createElement('canvas');
      m.width = W;
      m.height = H;
      const mc = m.getContext('2d', { willReadFrequently: true });
      mc.fillStyle = '#fff';
      for (const p of polys) {
        mc.beginPath();
        p.forEach((q, k) => (k ? mc.lineTo(q.x, q.y) : mc.moveTo(q.x, q.y)));
        mc.closePath();
        mc.fill();
      }
      mc.fillStyle = '#000';
      for (const p of holes) {
        mc.beginPath();
        p.forEach((q, k) => (k ? mc.lineTo(q.x, q.y) : mc.moveTo(q.x, q.y)));
        mc.closePath();
        mc.fill();
      }
      const md = mc.getImageData(0, 0, W, H).data;
      const px = [];
      for (let k = 0; k < W * H; k++) if (md[k * 4] > 128) px.push([lin(img[k * 4]), lin(img[k * 4 + 1]), lin(img[k * 4 + 2])]);
      if (px.length < 20) return null;
      px.sort((a, b) => a[0] + a[1] + a[2] - (b[0] + b[1] + b[2]));
      const a = Math.floor(px.length * 0.1);
      const sel = px.slice(a, px.length - a);
      const s = [0, 0, 0];
      for (const q of sel) for (let c = 0; c < 3; c++) s[c] += q[c];
      return s.map((x) => x / sel.length);
    };
    const circle = (c, r) => Array.from({ length: 16 }, (_, k) => ({ x: c.x + Math.cos((k / 16) * 6.283) * r, y: c.y + Math.sin((k / 16) * 6.283) * r }));
    if (kind === 'makeup') {
      const f = window.iris.lastFace;
      if (!f) return { error: '얼굴 없음' };
      const p = f.p;
      const fw = Math.hypot(p[234].x - p[454].x, p[234].y - p[454].y);
      const OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
      const IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191];
      const lip = polyMean([OUT.map((i) => p[i])], [IN.map((i) => p[i])]);
      // 볼 가장 붉은 곳(광대 앞쪽, 눈 밑) 두 곳, 기준 피부는 이마 가운데·코 옆 볼 안쪽이 아닌 턱 끝
      const cheek = polyMean([circle(p[118], fw * 0.06), circle(p[347], fw * 0.06)]);
      const skin = polyMean([circle(p[151], fw * 0.05), circle(p[199], fw * 0.035)]);
      return { fw: Math.round(fw), lip, cheek, skin, yaw: (p[1].x - (p[234].x + p[454].x) / 2) / fw };
    }
    if (kind === 'nail') {
      const hands = window.iris.lastHands;
      if (!hands.length) return { error: '손 없음' };
      const nails = [];
      for (const h of hands)
        for (const [t, d] of [[8, 7], [12, 11], [16, 15], [20, 19], [4, 3]]) {
          const seg = Math.hypot(h.p[t].x - h.p[d].x, h.p[t].y - h.p[d].y);
          const c = { x: h.p[d].x + (h.p[t].x - h.p[d].x) * 0.62, y: h.p[d].y + (h.p[t].y - h.p[d].y) * 0.62 };
          const m = polyMean([circle(c, seg * 0.22)]);
          if (m) nails.push(m);
        }
      return { hands: hands.length, nails };
    }
    // 헤어: 분할 R(머리카락) 확률 0.8 이상인 곳
    const seg = window.iris.lastTrack?.seg;
    if (!seg) return { error: '분할 없음' };
    const sx = seg.width / W;
    const sy = seg.height / H;
    const px = [];
    for (let y = 0; y < H; y += 2)
      for (let x = 0; x < W; x += 2) {
        const si = (Math.floor(y * sy) * seg.width + Math.floor(x * sx)) * 4;
        if (seg.flags[si] < 204) continue;
        const k = (y * W + x) * 4;
        px.push([lin(img[k]), lin(img[k + 1]), lin(img[k + 2]), y]);
      }
    if (px.length < 200) return { error: '머리카락 거의 없음', n: px.length };
    const mean = (arr) => [0, 1, 2].map((c) => arr.reduce((s, q) => s + q[c], 0) / arr.length);
    const Y = px.map((q) => q[3]).sort((a, b) => a - b);
    const y1 = Y[Math.floor(Y.length / 3)];
    const y2 = Y[Math.floor((Y.length * 2) / 3)];
    const L = (q) => 0.2126 * q[0] + 0.7152 * q[1] + 0.0722 * q[2];
    const Ls = px.map(L).sort((a, b) => a - b);
    const med = Ls[Math.floor(Ls.length / 2)];
    return {
      n: px.length,
      all: mean(px),
      top: mean(px.filter((q) => q[3] <= y1)),
      bottom: mean(px.filter((q) => q[3] >= y2)),
      // 밝은 가닥 비율: 가운데 밝기의 2배 넘는 픽셀
      bright: px.filter((q) => L(q) > med * 2).length / px.length,
      lo: Ls[Math.floor(Ls.length * 0.1)],
      hi: Ls[Math.floor(Ls.length * 0.9)],
    };
  }, kind);
  out.push({ id, ...r });
  process.stderr.write(`${id} ${r.error ?? 'ok'}\n`);
}
console.log(JSON.stringify(out));
await browser.close();
