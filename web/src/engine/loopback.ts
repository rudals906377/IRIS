// 거울 루프백 지연 측정.
//
// 화면을 검정↔흰색으로 바꾸고, 거울로 화면을 비춘 웹캠이 그 변화를 받을 때까지의 시간을 잰다.
// 측정 구간 = (그리기 → 화면 표시) + (화면 → 카메라 촬영 → 브라우저 수신).
// 아무 처리 없이 카메라 영상을 그대로 보여줄 때의 "움직임 → 화면" 지연과 같은 구성 요소다.
// DOM에 의존하지 않는 순수 로직이라 단위 테스트로 검증할 수 있다.

export type LoopbackPhase = 'idle' | 'calib-black' | 'calib-white' | 'wait' | 'measuring' | 'done' | 'failed';

export interface LoopbackConfig {
  trials: number;
  maxFailures: number;
  calibSettleMs: number;
  calibWindowMs: number;
  minContrast: number;
  timeoutMs: number;
  gapMinMs: number;
  gapMaxMs: number;
}

export const DEFAULT_LOOPBACK_CONFIG: LoopbackConfig = {
  trials: 30,
  maxFailures: 10,
  calibSettleMs: 400,
  calibWindowMs: 500,
  minContrast: 15,
  timeoutMs: 1500,
  gapMinMs: 350,
  gapMaxMs: 650,
};

export class LoopbackTest {
  phase: LoopbackPhase = 'idle';
  /** 지금 화면에 그려야 할 색. 0 = 검정, 1 = 흰색. */
  color: 0 | 1 = 0;
  readonly samples: number[] = [];
  failures = 0;
  message = '';
  blackLevel = NaN;
  whiteLevel = NaN;

  private phaseStart = 0;
  private calib: number[] = [];
  private nextSwitchAt = 0;
  private switchTime = 0;
  private target: 0 | 1 = 0;
  private readonly cfg: LoopbackConfig;
  private readonly random: () => number;

  constructor(cfg: LoopbackConfig = DEFAULT_LOOPBACK_CONFIG, random: () => number = Math.random) {
    this.cfg = cfg;
    this.random = random;
  }

  get running(): boolean {
    return this.phase !== 'idle' && this.phase !== 'done' && this.phase !== 'failed';
  }

  get threshold(): number {
    return (this.blackLevel + this.whiteLevel) / 2;
  }

  start(now: number): void {
    this.samples.length = 0;
    this.failures = 0;
    this.message = '';
    this.blackLevel = NaN;
    this.whiteLevel = NaN;
    this.enter('calib-black', now);
    this.color = 0;
  }

  cancel(): void {
    this.phase = 'idle';
    this.message = '취소됨';
  }

  /**
   * 화면 그리기 직전(requestAnimationFrame)에 호출한다. 이번 프레임에 그릴 색을 결정하고,
   * 색을 바꾸는 프레임이면 그 시각을 전환 시각으로 기록한다.
   */
  onDraw(now: number): 0 | 1 {
    if (this.phase === 'wait' && now >= this.nextSwitchAt) {
      this.target = this.color === 0 ? 1 : 0;
      this.color = this.target;
      this.switchTime = now;
      this.enter('measuring', now);
    }
    return this.color;
  }

  /** 카메라 프레임을 받을 때마다 호출한다. luma는 프레임 중앙부 평균 밝기(0~255). */
  onCameraFrame(now: number, luma: number): void {
    switch (this.phase) {
      case 'calib-black':
      case 'calib-white':
        this.calibrate(now, luma);
        break;
      case 'measuring': {
        const reached = this.target === 1 ? luma > this.threshold : luma < this.threshold;
        if (reached) {
          this.samples.push(now - this.switchTime);
          this.afterTrial(now);
        } else if (now - this.switchTime > this.cfg.timeoutMs) {
          this.failures++;
          this.afterTrial(now);
        }
        break;
      }
      default:
        break;
    }
  }

  private calibrate(now: number, luma: number): void {
    const elapsed = now - this.phaseStart;
    if (elapsed >= this.cfg.calibSettleMs) this.calib.push(luma);
    if (elapsed < this.cfg.calibSettleMs + this.cfg.calibWindowMs) return;

    const level = median(this.calib);
    if (this.phase === 'calib-black') {
      this.blackLevel = level;
      this.color = 1;
      this.enter('calib-white', now);
      return;
    }
    this.whiteLevel = level;
    if (!(this.whiteLevel - this.blackLevel >= this.cfg.minContrast)) {
      this.phase = 'failed';
      this.message = `카메라가 화면 깜빡임을 보지 못했습니다(밝기 차이 ${(this.whiteLevel - this.blackLevel).toFixed(1)}). 거울 위치를 조정하세요.`;
      return;
    }
    this.enter('wait', now);
    this.scheduleNext(now);
  }

  private afterTrial(now: number): void {
    if (this.samples.length >= this.cfg.trials) {
      this.phase = 'done';
      this.message = '측정 완료';
      return;
    }
    if (this.failures >= this.cfg.maxFailures) {
      this.phase = 'failed';
      this.message = `실패 ${this.failures}회로 중단했습니다. 거울 위치와 조명을 확인하세요.`;
      return;
    }
    this.enter('wait', now);
    this.scheduleNext(now);
  }

  private scheduleNext(now: number): void {
    const gap = this.cfg.gapMinMs + (this.cfg.gapMaxMs - this.cfg.gapMinMs) * this.random();
    this.nextSwitchAt = now + gap;
  }

  private enter(phase: LoopbackPhase, now: number): void {
    this.phase = phase;
    this.phaseStart = now;
    this.calib = [];
  }
}

function median(values: number[]): number {
  if (values.length === 0) return NaN;
  const s = [...values].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
