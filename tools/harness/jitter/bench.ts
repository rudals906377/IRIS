// 얼굴 점 필터 비교(머리 움직임): 원본 vs 앱의 FaceTracker vs 실험 변형들.
// 사용법(web 폴더에서): node --experimental-strip-types ../tools/harness/jitter/bench.ts <폴더(raw.json·poses.json)> [얼굴 전체 잡음 px] [점별 잡음 px]
import { readFileSync } from 'node:fs';
import { FaceTracker } from '../../../web/src/engine/face.ts';
import { OneEuro, PointFilter, type OneEuroParams } from '../../../web/src/engine/filters.ts';
const DIR = process.argv[2];
const raw: number[][][] = JSON.parse(readFileSync(`${DIR}/raw.json`, 'utf8'));
const poses: number[][] = JSON.parse(readFileSync(`${DIR}/poses.json`, 'utf8')).poses;
const N = raw.length;
// 정답
const L0 = raw[0].map((_, j) => { let x = 0, y = 0; for (let i = 5; i < 55; i++) { x += raw[i][j][0]; y += raw[i][j][1]; } return [x / 50, y / 50]; });
const gt = (i: number) => { const [dx, dy, th, s] = poses[i]; const t = th * Math.PI / 180; const c = Math.cos(t), sn = Math.sin(t);
  return L0.map(([x, y]) => { const u = x - 260, v = y - 240; return [260 + dx + s * (u * c + v * sn), 240 + dy + s * (-u * sn + v * c)]; }); };
// 잡음(재현 가능)
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
const [RIG, PT] = [Number(process.argv[3] ?? 0.5), Number(process.argv[4] ?? 0.4)];
const noisy = raw.map((f) => { const tx = gauss() * RIG, ty = gauss() * RIG, sc = 1 + gauss() * RIG * 0.004, rt = gauss() * RIG * 0.002;
  let cx = 0, cy = 0; for (const [x, y] of f) { cx += x / f.length; cy += y / f.length; }
  return f.map(([x, y]) => { const u = x - cx, v = y - cy; return [cx + tx + sc * (u * Math.cos(rt) - v * Math.sin(rt)) + gauss() * PT, cy + ty + sc * (u * Math.sin(rt) + v * Math.cos(rt)) + gauss() * PT]; }); });

function evaluate(name: string, run: (f: number[][], t: number) => number[][]) {
  const out: number[][][] = [];
  for (let i = 0; i < N; i++) out.push(run(noisy[i], 1000 + i * 33.333));
  const still = [...Array.from({ length: 45 }, (_, k) => 10 + k), ...Array.from({ length: 30 }, (_, k) => 208 + k)];
  let jit = 0; for (const i of still) { let s = 0; for (let j = 0; j < 478; j++) s += Math.hypot(out[i][j][0] - out[i - 1][j][0], out[i][j][1] - out[i - 1][j][1]); jit += s / 478; }
  const err = (a: number, b: number) => { let e = 0; for (let i = a; i < b; i++) { const g = gt(i); let s = 0; for (let j = 0; j < 478; j++) s += Math.hypot(out[i][j][0] - g[j][0], out[i][j][1] - g[j][1]); e += s / 478; } return e / (b - a); };
  // 얼굴 모양 일그러짐: 정답과의 오차에서 가장 잘 맞는 닮음 변환을 뺀 나머지(모양만)
  console.log(name.padEnd(28), '정지 떨림', (jit / still.length).toFixed(3), 'px/프레임 | 느린 이동 오차', err(60, 120).toFixed(2), '| 빠른 흔들기 오차', err(120, 180).toFixed(2), '| 급정지 후', err(180, 208).toFixed(2));
}
evaluate('원본(필터 없음)', (f) => f);
{ const pf = Array.from({ length: 478 }, () => new PointFilter({ minCutoff: 1.2, beta: 0.01, dCutoff: 1 })); evaluate('이전: 점별 One Euro', (f, t) => f.map(([x, y], j) => { const q = pf[j].filter(x, y, t); return [q.x, q.y]; })); }
{ const ft = new FaceTracker(); evaluate('앱 FaceTracker', (f, t) => ft.update(f.map(([x, y]) => ({ x: x / 640, y: y / 480 })), t, 640, 480)!.p.map((q) => [q.x, q.y])); }

// 2단 필터
const RIGID = [33, 133, 362, 263, 1, 4, 5, 6, 168, 197, 195, 10, 151, 9, 234, 454, 127, 356, 93, 323];
function twoStage(poseP: OneEuroParams, localP: OneEuroParams) {
  let ref: number[][] | null = null; let refScale = 1;
  const pf = [0, 1, 2, 3].map(() => new OneEuro(poseP));
  const lf: PointFilter[] = [];
  return (f: number[][], t: number) => {
    let cx = 0, cy = 0; for (const j of RIGID) { cx += f[j][0] / RIGID.length; cy += f[j][1] / RIGID.length; }
    if (!ref) { ref = f.map(([x, y]) => [x - cx, y - cy]); refScale = Math.sqrt(RIGID.reduce((a, j) => a + ref![j][0] ** 2 + ref![j][1] ** 2, 0) / RIGID.length); for (let j = 0; j < f.length; j++) lf.push(new PointFilter(localP)); }
    // 닮음 변환(Procrustes): 기준 모양 → 현재
    let sxx = 0, sxy = 0, ss = 0;
    for (const j of RIGID) { const [rx, ry] = ref[j]; const x = f[j][0] - cx, y = f[j][1] - cy; sxx += rx * x + ry * y; sxy += rx * y - ry * x; ss += rx * rx + ry * ry; }
    const th = Math.atan2(sxy, sxx); const s = Math.hypot(sxx, sxy) / ss;
    // 자세를 px 단위로 걸러낸다(회전·크기는 얼굴 크기를 곱해 px로 환산)
    const fcx = pf[0].filter(cx, t), fcy = pf[1].filter(cy, t);
    const fs = pf[2].filter(s * refScale, t) / refScale, fth = pf[3].filter(th * refScale, t) / refScale;
    const c = Math.cos(th), sn = Math.sin(th), fc = Math.cos(fth), fsn = Math.sin(fth);
    return f.map(([x, y], j) => {
      // 얼굴 안 좌표(원래 자세로 되돌림)
      const u = x - cx, v = y - cy; const lx = (u * c + v * sn) / s, ly = (-u * sn + v * c) / s;
      const q = lf[j].filter(lx, ly, t);
      return [fcx + fs * (q.x * fc - q.y * fsn), fcy + fs * (q.x * fsn + q.y * fc)];
    });
  };
}

// 변위 적응 필터: 차이가 d0 이하(잡음)면 aMin으로 강하게, d1 이상이면 바로 따라감. 2차원 벡터 단위로
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
function twoStageD(d0: number, d1: number, aMin: number, localP: OneEuroParams) {
  let ref: number[][] | null = null; let refScale = 1;
  const tf = new DispFilter(d0, d1, aMin), rf = new DispFilter(d0, d1, aMin);
  const lf: PointFilter[] = [];
  return (f: number[][], t: number) => {
    let cx = 0, cy = 0; for (const j of RIGID) { cx += f[j][0] / RIGID.length; cy += f[j][1] / RIGID.length; }
    if (!ref) { ref = f.map(([x, y]) => [x - cx, y - cy]); refScale = Math.sqrt(RIGID.reduce((a, j) => a + ref![j][0] ** 2 + ref![j][1] ** 2, 0) / RIGID.length); for (let j = 0; j < f.length; j++) lf.push(new PointFilter(localP)); }
    let sxx = 0, sxy = 0, ss = 0;
    for (const j of RIGID) { const [rx, ry] = ref[j]; const x = f[j][0] - cx, y = f[j][1] - cy; sxx += rx * x + ry * y; sxy += rx * y - ry * x; ss += rx * rx + ry * ry; }
    const th = Math.atan2(sxy, sxx); const s = Math.hypot(sxx, sxy) / ss;
    const [fcx, fcy] = tf.filter([cx, cy], t);
    const [fsR, fthR] = rf.filter([s * refScale, th * refScale], t);
    const fs = fsR / refScale, fth = fthR / refScale;
    const c = Math.cos(th), sn = Math.sin(th), fc = Math.cos(fth), fsn = Math.sin(fth);
    return f.map(([x, y], j) => {
      const u = x - cx, v = y - cy; const lx = (u * c + v * sn) / s, ly = (-u * sn + v * c) / s;
      const q = lf[j].filter(lx, ly, t);
      return [fcx + fs * (q.x * fc - q.y * fsn), fcy + fs * (q.x * fsn + q.y * fc)];
    });
  };
}
for (const [d0, d1, am, lm, lb] of [[0.5, 4, 0.08, 0.5, 0.03], [0.8, 5, 0.06, 0.5, 0.03], [1.0, 6, 0.05, 0.5, 0.03], [0.6, 3, 0.08, 0.5, 0.03], [1.0, 4, 0.06, 0.4, 0.03], [1.5, 6, 0.05, 0.5, 0.03]])
  evaluate(`변위 ${d0}/${d1}/${am} 모양${lm}/${lb}`, twoStageD(d0, d1, am, { minCutoff: lm, beta: lb, dCutoff: 1 }));
