// 실시간성 측정: FPS 하나로 판단하지 않도록 단계별 처리 시간, 프레임 나이(촬영→그리기 완료),
// 건너뛴 프레임, 1분 단위 추세(지연 누적 확인)를 함께 기록한다.

import { RateMeter, Rolling, fmt, summarize } from './stats.ts';

export interface FrameRecord {
  /** 프레임 수신 시각(ms, performance.now). */
  t: number;
  /** 수신 → 그리기 제출까지 처리 시간(ms). */
  proc: number;
  pose: number;
  seg: number;
  hands: number;
  rig: number;
  render: number;
  /** 촬영 → 그리기 제출(ms). 브라우저가 촬영 시각을 줄 때만. */
  age?: number;
  /** 이번 콜백 전까지 처리하지 못하고 지나간 프레임 수. */
  skipped: number;
  person: boolean;
}

export interface MinuteSummary {
  minute: number;
  frames: number;
  fps: number;
  procP50: number;
  procP95: number;
  ageP50: number;
  ageP95: number;
  skipped: number;
}

const MAX_RECORDS = 60 * 60 * 30; // 30fps 기준 약 1시간

export class Metrics {
  readonly proc = new Rolling(300);
  readonly pose = new Rolling(300);
  readonly seg = new Rolling(300);
  readonly hands = new Rolling(300);
  readonly rig = new Rolling(300);
  readonly render = new Rolling(300);
  readonly age = new Rolling(300);
  readonly frameRate = new RateMeter(1000);
  readonly records: FrameRecord[] = [];
  readonly minutes: MinuteSummary[] = [];
  skippedTotal = 0;
  framesTotal = 0;
  private startT = -1;
  private minuteBuf: FrameRecord[] = [];
  private lastPresented = -1;

  /** presentedFrames 차이로 건너뛴 프레임 수를 계산한다. */
  skippedSince(presented: number | undefined): number {
    if (presented === undefined) return 0;
    const skipped = this.lastPresented >= 0 ? Math.max(0, presented - this.lastPresented - 1) : 0;
    this.lastPresented = presented;
    return skipped;
  }

  push(r: FrameRecord): void {
    if (this.startT < 0) this.startT = r.t;
    this.framesTotal++;
    this.skippedTotal += r.skipped;
    this.frameRate.tick(r.t);
    this.proc.push(r.proc);
    if (r.pose) this.pose.push(r.pose);
    if (r.seg) this.seg.push(r.seg);
    if (r.hands) this.hands.push(r.hands);
    this.rig.push(r.rig);
    this.render.push(r.render);
    if (r.age !== undefined) this.age.push(r.age);
    if (this.records.length >= MAX_RECORDS) this.records.shift();
    this.records.push(r);

    this.minuteBuf.push(r);
    const minute = Math.floor((r.t - this.startT) / 60000);
    const first = this.minuteBuf[0];
    if (Math.floor((first.t - this.startT) / 60000) !== minute) {
      const done = this.minuteBuf.slice(0, -1);
      this.minuteBuf = [r];
      const proc = summarize(done.map((d) => d.proc));
      const age = summarize(done.filter((d) => d.age !== undefined).map((d) => d.age!));
      const span = (done[done.length - 1].t - done[0].t) / 1000 || 1;
      this.minutes.push({
        minute: minute - 1,
        frames: done.length,
        fps: done.length / span,
        procP50: proc.p50,
        procP95: proc.p95,
        ageP50: age.p50,
        ageP95: age.p95,
        skipped: done.reduce((a, d) => a + d.skipped, 0),
      });
    }
  }

  reset(): void {
    for (const r of [this.proc, this.pose, this.seg, this.hands, this.rig, this.render, this.age]) r.clear();
    this.frameRate.clear();
    this.records.length = 0;
    this.minutes.length = 0;
    this.minuteBuf = [];
    this.skippedTotal = 0;
    this.framesTotal = 0;
    this.startT = -1;
    this.lastPresented = -1;
  }

  hudLines(now: number): string[] {
    const q = (r: Rolling): string => `${fmt(r.quantile(0.5))} / ${fmt(r.quantile(0.95))}`;
    const lines = [
      `프레임 처리  ${fmt(this.frameRate.rate(now), 0)} fps`,
      `처리 시간    ${q(this.proc)} ms (중앙/95%)`,
      `  자세 ${fmt(this.pose.quantile(0.5))} · 분할 ${fmt(this.seg.quantile(0.5))} · 변형 ${fmt(this.rig.quantile(0.5))} · 그리기 ${fmt(this.render.quantile(0.5))}`,
    ];
    if (this.hands.count) lines.push(`  손 ${fmt(this.hands.quantile(0.5))} ms`);
    lines.push(
      this.age.count
        ? `촬영→그리기  ${q(this.age)} ms`
        : '촬영→그리기  이 브라우저는 촬영 시각 미제공',
    );
    const skipRate = this.framesTotal ? (this.skippedTotal / (this.framesTotal + this.skippedTotal)) * 100 : 0;
    lines.push(`건너뛴 프레임 ${this.skippedTotal} (${fmt(skipRate)}%)`);
    return lines;
  }
}
