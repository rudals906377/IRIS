// 손 점 필터 비교(합성 영상의 손 2개): 이전 점별 One Euro vs 앱의 묶음 필터(HAND_FILTER).
// 사용법(web 폴더에서): node --experimental-strip-types ../tools/harness/jitter/hands.ts <폴더(raw-hands.json·poses.json)> [손 전체 잡음 px] [점별 잡음 px]
import { readFileSync } from 'node:fs';
import { PointFilter, RigidShapeFilter } from '../../../web/src/engine/filters.ts';
import { HAND_FILTER } from '../../../web/src/engine/engine.ts';

const DIR = process.argv[2];
const raw: number[][][][] = JSON.parse(readFileSync(`${DIR}/raw-hands.json`, 'utf8'));
const poses: number[][] = JSON.parse(readFileSync(`${DIR}/poses.json`, 'utf8')).poses;
const N = raw.length;
const [RIG, PT] = [Number(process.argv[3] ?? 0.6), Number(process.argv[4] ?? 0.6)];
let seed = 5;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
// 프레임 사이 손 짝 맞추기: 앞 프레임 손과 손목·손가락 뿌리 가운데가 가까운 쪽끼리
const ctr = (hd: number[][]) => [0, 5, 9, 13, 17].reduce((a, j) => [a[0] + hd[j][0] / 5, a[1] + hd[j][1] / 5], [0, 0]);
for (let i = 1; i < raw.length; i++) {
  const [a, b] = raw[i];
  const prev = raw[i - 1];
  if (!a || !b || prev.length < 2) continue;
  const d = (x: number[][], y: number[][]) => Math.hypot(ctr(x)[0] - ctr(y)[0], ctr(x)[1] - ctr(y)[1]);
  if (d(a, prev[0]) + d(b, prev[1]) > d(a, prev[1]) + d(b, prev[0])) raw[i] = [b, a];
}
for (let h = 0; h < 2; h++) {
  const ok = raw.map((f) => f.length === 2);
  const still = raw.slice(5, 55).filter((f) => f.length === 2).map((f) => f[h]);
  const L0 = still[0].map((_, j) => [still.reduce((a, f) => a + f[j][0], 0) / still.length, still.reduce((a, f) => a + f[j][1], 0) / still.length]);
  const gt = (i: number) => { const [dx, dy, th, s] = poses[i]; const t = th * Math.PI / 180; const c = Math.cos(t), sn = Math.sin(t);
    return L0.map(([x, y]) => { const u = x - 260, v = y - 240; return [260 + dx + s * (u * c + v * sn), 240 + dy + s * (-u * sn + v * c)]; }); };
  const noisy = raw.map((f, i) => { const g = ok[i] ? f[h] : gt(i); const tx = gauss() * RIG, ty = gauss() * RIG; return g.map(([x, y]) => [x + tx + gauss() * PT, y + ty + gauss() * PT]); });
  const size = 2 * Math.hypot(L0[9][0] - L0[0][0], L0[9][1] - L0[0][1]);
  const evaluate = (name: string, run: (f: number[][], t: number) => number[][]) => {
    const out = noisy.map((f, i) => run(f, 1000 + i * 33.333));
    const still = [...Array.from({ length: 45 }, (_, k) => 10 + k), ...Array.from({ length: 30 }, (_, k) => 208 + k)];
    let jit = 0; for (const i of still) jit += out[i].reduce((a, q, j) => a + Math.hypot(q[0] - out[i - 1][j][0], q[1] - out[i - 1][j][1]), 0) / 21;
    const err = (a: number, b: number) => { let e = 0; for (let i = a; i < b; i++) { const g = gt(i); e += out[i].reduce((s, q, j) => s + Math.hypot(q[0] - g[j][0], q[1] - g[j][1]), 0) / 21; } return e / (b - a); };
    console.log(`손${h} ${name}`.padEnd(26), '정지 떨림', (jit / still.length).toFixed(3), '| 느린 이동', err(60, 120).toFixed(2), '| 빠른 흔들기', err(120, 180).toFixed(2), '| 급정지 후', err(180, 208).toFixed(2), `(손 크기 ${size.toFixed(0)}px)`);
  };
  evaluate('원본', (f) => f);
  { const pf = Array.from({ length: 21 }, () => new PointFilter({ minCutoff: 1.5, beta: 0.03, dCutoff: 1 })); evaluate('이전 점별 One Euro', (f, t) => f.map(([x, y], j) => { const q = pf[j].filter(x, y, t); return [q.x, q.y]; })); }
  { const rf = new RigidShapeFilter(HAND_FILTER); evaluate('앱 묶음 필터', (f, t) => rf.filter(f.map(([x, y]) => ({ x, y })), size, t).map((q) => [q.x, q.y])); }
}
