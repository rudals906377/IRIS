// One Euro 필터(Casiez et al., CHI 2012).
// 느리게 움직일 때는 떨림을 강하게 줄이고, 빠르게 움직일 때는 거의 거르지 않아 지연을 적게 더한다.

function alpha(dtSec: number, cutoffHz: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dtSec);
}

export interface OneEuroParams {
  /** 정지 상태의 차단 주파수(Hz). 낮을수록 떨림이 줄고 지연이 늘어난다. */
  minCutoff: number;
  /** 속도에 따른 차단 주파수 증가율. 클수록 빠른 동작을 덜 늦춘다. */
  beta: number;
  /** 속도 추정용 차단 주파수(Hz). */
  dCutoff: number;
}

export const DEFAULT_ONE_EURO: OneEuroParams = { minCutoff: 1.7, beta: 0.004, dCutoff: 1.0 };

export class OneEuro {
  private x = 0;
  private dx = 0;
  private lastT = -1;
  params: OneEuroParams;

  constructor(params: OneEuroParams = DEFAULT_ONE_EURO) {
    this.params = params;
  }

  /** tMs: 밀리초 단위 시각. */
  filter(value: number, tMs: number): number {
    if (this.lastT < 0) {
      this.x = value;
      this.dx = 0;
      this.lastT = tMs;
      return value;
    }
    if (tMs <= this.lastT) return this.x;
    const dt = Math.min(0.2, (tMs - this.lastT) / 1000);
    this.lastT = tMs;
    const rawDx = (value - this.x) / dt;
    this.dx += alpha(dt, this.params.dCutoff) * (rawDx - this.dx);
    const cutoff = this.params.minCutoff + this.params.beta * Math.abs(this.dx);
    this.x += alpha(dt, cutoff) * (value - this.x);
    return this.x;
  }

  reset(): void {
    this.lastT = -1;
  }
}

/** 2D 점용 One Euro 필터 묶음. */
export class PointFilter {
  private readonly fx: OneEuro;
  private readonly fy: OneEuro;

  constructor(params?: OneEuroParams) {
    this.fx = new OneEuro(params);
    this.fy = new OneEuro(params);
  }

  filter(x: number, y: number, tMs: number): { x: number; y: number } {
    return { x: this.fx.filter(x, tMs), y: this.fy.filter(y, tMs) };
  }

  setParams(p: OneEuroParams): void {
    this.fx.params = p;
    this.fy.params = p;
  }

  reset(): void {
    this.fx.reset();
    this.fy.reset();
  }
}

/**
 * 변위 적응 필터(2차원). 걸러 둔 값과의 차이가 d0 이하(잡음 수준)면 aMin으로 강하게 누르고,
 * d1 이상이면 바로 따라간다. 속도(미분)를 추정하지 않으므로 잡음에 흔들리지 않고, 큰 움직임엔 지연이 거의 없다.
 * aMin은 30fps 한 프레임 기준이며 프레임 간격에 맞춰 환산한다.
 */
export class AdaptiveFilter2 {
  private x = 0;
  private y = 0;
  private lastT = -1;
  d0: number;
  d1: number;
  aMin: number;

  constructor(d0: number, d1: number, aMin: number) {
    this.d0 = d0;
    this.d1 = d1;
    this.aMin = aMin;
  }

  filter(x: number, y: number, tMs: number): { x: number; y: number } {
    if (this.lastT < 0) {
      this.x = x;
      this.y = y;
      this.lastT = tMs;
      return { x, y };
    }
    if (tMs <= this.lastT) return { x: this.x, y: this.y };
    const dt = Math.min(200, tMs - this.lastT);
    this.lastT = tMs;
    const e = Math.hypot(x - this.x, y - this.y);
    const a1 = Math.min(1, Math.max(this.aMin, (e - this.d0) / (this.d1 - this.d0)));
    const a = 1 - Math.pow(1 - a1, dt / 33.333);
    this.x += a * (x - this.x);
    this.y += a * (y - this.y);
    return { x: this.x, y: this.y };
  }

  reset(): void {
    this.lastT = -1;
  }
}

export interface RigidShapeParams {
  /** 자세(닮음 변환)를 잴 때 쓰는, 모양이 잘 안 변하는 점 번호 */
  rigid: number[];
  /** 기준값은 모두 size(얼굴 폭·손 크기 px)에 곱하는 비율 */
  poseD0: number;
  poseD1: number;
  poseAMin: number;
  shapeD0: number;
  shapeD1: number;
  shapeAMin: number;
}

/**
 * 2단 점 묶음 필터(Procrustes 분해).
 *  ① 전체 움직임(이동·회전·크기)을 rigid 점으로 재어 변위 적응 필터로 거르고,
 *  ② 그 움직임을 되돌린 좌표에서 점마다(표정·손가락 움직임·잔떨림) 변위 적응 필터로 거른 뒤 다시 합친다.
 * 점마다 따로 거르면 빠르게 움직일 때 크게 늦고 점마다 늦는 정도가 달라 모양이 일그러진다.
 */
export class RigidShapeFilter {
  private readonly params: RigidShapeParams;
  private readonly pose = [new AdaptiveFilter2(0, 1, 1), new AdaptiveFilter2(0, 1, 1)];
  private shape: AdaptiveFilter2[] = [];
  private ref: { x: number; y: number }[] | null = null;
  private refScale = 1;

  constructor(params: RigidShapeParams) {
    this.params = params;
  }

  /** raw: 픽셀 좌표, size: 기준 크기(px) */
  filter(raw: { x: number; y: number }[], size: number, tMs: number): { x: number; y: number }[] {
    const P = this.params;
    const rigid = P.rigid;
    let cx = 0;
    let cy = 0;
    for (const j of rigid) {
      cx += raw[j].x / rigid.length;
      cy += raw[j].y / rigid.length;
    }
    if (!this.ref || this.shape.length !== raw.length) {
      const ref = raw.map((q) => ({ x: q.x - cx, y: q.y - cy }));
      this.ref = ref;
      this.refScale = Math.sqrt(rigid.reduce((a, j) => a + ref[j].x ** 2 + ref[j].y ** 2, 0) / rigid.length) || 1;
      this.shape = raw.map(() => new AdaptiveFilter2(0, 1, 1));
      for (const f of this.pose) f.reset();
    }
    // ① 기준 모양 → 현재의 닮음 변환(회전 th, 크기 s)
    let sxx = 0;
    let sxy = 0;
    let ss = 0;
    for (const j of rigid) {
      const r = this.ref[j];
      const x = raw[j].x - cx;
      const y = raw[j].y - cy;
      sxx += r.x * x + r.y * y;
      sxy += r.x * y - r.y * x;
      ss += r.x * r.x + r.y * r.y;
    }
    const th = Math.atan2(sxy, sxx);
    const s = Math.hypot(sxx, sxy) / ss || 1;
    // 이동은 px, 회전·크기는 기준 크기를 곱해 px로 환산해 같은 기준으로 거른다
    const k = this.refScale;
    for (const f of this.pose) {
      f.d0 = P.poseD0 * size;
      f.d1 = P.poseD1 * size;
      f.aMin = P.poseAMin;
    }
    const pc = this.pose[0].filter(cx, cy, tMs);
    const pr = this.pose[1].filter(s * k, th * k, tMs);
    const fs = pr.x / k;
    const fth = pr.y / k;
    // ② 움직임을 되돌린 좌표(기준 크기 px)에서 점마다
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const fc = Math.cos(fth);
    const fsn = Math.sin(fth);
    const refSize = size / s;
    return raw.map((q, j) => {
      const f = this.shape[j];
      f.d0 = P.shapeD0 * refSize;
      f.d1 = P.shapeD1 * refSize;
      f.aMin = P.shapeAMin;
      const u = q.x - cx;
      const v = q.y - cy;
      const l = f.filter((u * c + v * sn) / s, (-u * sn + v * c) / s, tMs);
      return { x: pc.x + fs * (l.x * fc - l.y * fsn), y: pc.y + fs * (l.x * fsn + l.y * fc) };
    });
  }

  reset(): void {
    this.ref = null;
    this.shape = [];
    for (const f of this.pose) f.reset();
  }
}
