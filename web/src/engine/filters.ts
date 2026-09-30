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
