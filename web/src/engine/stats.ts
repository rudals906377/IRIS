// 지연·처리 시간 통계. FPS 하나로 판단하지 않도록 분위수(p50/p95)와 최댓값을 함께 본다.

/** 최근 N개 샘플의 분위수를 계산하는 고정 크기 버퍼. */
export class Rolling {
  private readonly buf: Float64Array;
  private next = 0;
  private size = 0;

  constructor(capacity = 300) {
    this.buf = new Float64Array(capacity);
  }

  push(v: number): void {
    if (!Number.isFinite(v)) return;
    this.buf[this.next] = v;
    this.next = (this.next + 1) % this.buf.length;
    if (this.size < this.buf.length) this.size++;
  }

  get count(): number {
    return this.size;
  }

  /** q는 0~1. 샘플이 없으면 NaN. */
  quantile(q: number): number {
    if (this.size === 0) return NaN;
    const sorted = Array.from(this.buf.subarray(0, this.size)).sort((a, b) => a - b);
    return quantileSorted(sorted, q);
  }

  max(): number {
    if (this.size === 0) return NaN;
    let m = -Infinity;
    for (let i = 0; i < this.size; i++) m = Math.max(m, this.buf[i]);
    return m;
  }

  clear(): void {
    this.next = 0;
    this.size = 0;
  }
}

export function quantileSorted(sorted: readonly number[], q: number): number {
  if (sorted.length === 0) return NaN;
  const pos = (sorted.length - 1) * Math.min(1, Math.max(0, q));
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** 최근 windowMs 동안 발생한 이벤트 수로 초당 빈도를 계산한다. */
export class RateMeter {
  private readonly times: number[] = [];
  private readonly windowMs: number;

  constructor(windowMs = 1000) {
    this.windowMs = windowMs;
  }

  tick(t: number): void {
    this.times.push(t);
    const cutoff = t - this.windowMs;
    while (this.times.length > 0 && this.times[0] < cutoff) this.times.shift();
  }

  rate(now: number): number {
    const cutoff = now - this.windowMs;
    let n = 0;
    for (const t of this.times) if (t >= cutoff) n++;
    return (n * 1000) / this.windowMs;
  }

  clear(): void {
    this.times.length = 0;
  }
}

export interface Summary {
  n: number;
  p50: number;
  p95: number;
  max: number;
}

export function summarize(values: readonly number[]): Summary {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return {
    n: sorted.length,
    p50: quantileSorted(sorted, 0.5),
    p95: quantileSorted(sorted, 0.95),
    max: sorted.length ? sorted[sorted.length - 1] : NaN,
  };
}

export function fmt(v: number, digits = 1): string {
  return Number.isFinite(v) ? v.toFixed(digits) : '–';
}
