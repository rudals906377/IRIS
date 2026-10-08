// 말하기(입 벌림) 합성: 머리는 가만히, 아랫입술·턱이 초당 3회 벌어졌다 닫힘. 움직이는 입술 점 오차를 잰다.
// 사용법(web 폴더에서): node --experimental-strip-types ../tools/harness/jitter/talk.ts <폴더(raw.json)> [벌림 px]
import { readFileSync } from 'node:fs';
import { FaceTracker } from '../../../web/src/engine/face.ts';
import { OneEuro, PointFilter, type OneEuroParams } from '../../../web/src/engine/filters.ts';
import { LIPS_OUTER, LIPS_INNER } from '../../../web/src/beauty/face-regions.ts';
const raw: number[][][] = JSON.parse(readFileSync(`${process.argv[2]}/raw.json`, 'utf8'));
const L0 = raw[0].map((_, j) => { let x = 0, y = 0; for (let i = 5; i < 55; i++) { x += raw[i][j][0]; y += raw[i][j][1]; } return [x / 50, y / 50]; });
const mouthY = (L0[13][1] + L0[14][1]) / 2, chinY = L0[152][1];
const N = 240, A = Number(process.argv[3] ?? 12);
const gt = (i: number) => { const t = i / 30; const open = t > 2 && t < 6 ? A * (0.5 - 0.5 * Math.cos(2 * Math.PI * 3 * (t - 2))) : 0;
  return L0.map(([x, y]) => [x, y + open * Math.min(1, Math.max(0, (y - mouthY + 1) / (0.2 * (chinY - mouthY))))]); };
let seed = 11; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const noisy = Array.from({ length: N }, (_, i) => { const g = gt(i); const tx = gauss() * 0.7, ty = gauss() * 0.7; return g.map(([x, y]) => [x + tx + gauss() * 0.5, y + ty + gauss() * 0.5]); });
const LIP = [...LIPS_OUTER, ...LIPS_INNER].filter((j) => { const a = gt(75)[j][1] - L0[j][1]; return a > 6; });
console.log('움직이는 입술 점', LIP.length);
function evaluate(name: string, run: (f: number[][], t: number) => number[][]) {
  let e = 0, n = 0, jit = 0;
  const out = noisy.map((f, i) => run(f, 1000 + i * 33.333));
  for (let i = 60; i < 180; i++) { const g = gt(i); for (const j of LIP) { e += Math.hypot(out[i][j][0] - g[j][0], out[i][j][1] - g[j][1]); n++; } }
  for (let i = 20; i < 58; i++) { for (const j of LIP) jit += Math.hypot(out[i][j][0] - out[i - 1][j][0], out[i][j][1] - out[i - 1][j][1]) / LIP.length; }
  console.log(name.padEnd(30), '말할 때 입술 오차', (e / n).toFixed(2), 'px | 정지 입술 떨림', (jit / 38).toFixed(3));
}
evaluate('원본', (f) => f);
{ const pf = Array.from({ length: 478 }, () => new PointFilter({ minCutoff: 1.2, beta: 0.01, dCutoff: 1 })); evaluate('이전: 점별 One Euro', (f, t) => f.map(([x, y], j) => { const q = pf[j].filter(x, y, t); return [q.x, q.y]; })); }
{ const ft = new FaceTracker(); evaluate('앱 FaceTracker', (f, t) => ft.update(f.map(([x, y]) => ({ x: x / 640, y: y / 480 })), t, 640, 480)!.p.map((q) => [q.x, q.y])); }
const RIGID = [33, 133, 362, 263, 1, 4, 5, 6, 168, 197, 195, 10, 151, 9, 234, 454, 127, 356, 93, 323];
for (const [lm, lb, ld] of [[0.5, 0.03, 1], [0.8, 0.03, 1], [1.0, 0.05, 1], [0.8, 0.08, 1.5], [1.0, 0.1, 2], [0.6, 0.1, 2]]) {
  const lf = L0.map(() => new PointFilter({ minCutoff: lm, beta: lb, dCutoff: ld }));
  // 머리 자세는 고정이므로 모양 필터만 비교
  evaluate(`모양 ${lm}/${lb}/${ld}`, (f, t) => f.map(([x, y], j) => { const q = lf[j].filter(x, y, t); return [q.x, q.y]; }));
}

class DispFilter {
  x: number[] | null = null; lastT = -1; d0: number; d1: number; aMin: number;
  constructor(d0: number, d1: number, aMin: number) { this.d0 = d0; this.d1 = d1; this.aMin = aMin; }
  filter(v: number[], t: number): number[] {
    if (!this.x) { this.x = [...v]; this.lastT = t; return this.x; }
    const dt = Math.max(1, t - this.lastT); this.lastT = t;
    const e = Math.hypot(...v.map((q, i) => q - this.x![i]));
    const a1 = Math.min(1, Math.max(this.aMin, (e - this.d0) / (this.d1 - this.d0)));
    const a = 1 - Math.pow(1 - a1, dt / 33.333);
    this.x = this.x.map((q, i) => q + a * (v[i] - q));
    return this.x;
  }
}
for (const [d0, d1, am] of [[1.0, 4, 0.08], [1.2, 5, 0.06], [1.5, 5, 0.06], [1.0, 3, 0.1], [0.8, 4, 0.1]]) {
  const lf = L0.map(() => new DispFilter(d0, d1, am));
  evaluate(`모양(변위) ${d0}/${d1}/${am}`, (f, t) => f.map(([x, y], j) => lf[j].filter([x, y], t)));
}
