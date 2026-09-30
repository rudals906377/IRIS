// 실시간 뷰티 엔진: 영상 입력 → 추적(얼굴·분할) → 효과(메이크업 등) → 합성 → 측정을 한 프레임 안에서 처리한다.
// 원칙: 가장 최신 프레임만 처리하고(밀린 프레임은 건너뜀), 추적한 바로 그 프레임 위에 효과를 합성한다.
// 영상은 이 기기 안에서만 처리되며 어디에도 전송하지 않는다.

import { faceRegions, type FaceRegions } from '../beauty/face-regions.ts';
import { HairColorRenderer, type HairLook } from '../beauty/hair.ts';
import { MakeupRenderer, type MakeupLook } from '../beauty/makeup.ts';
import { FaceTracker, type FaceFrame } from './face.ts';
import { LoopbackTest } from './loopback.ts';
import { Metrics } from './metrics.ts';
import { Renderer, type FrameTextures } from './renderer.ts';
import { VideoSource, type FrameInfo } from './source.ts';
import { Tracker, type TrackResult } from './tracker.ts';

export interface EngineSettings {
  /** 분할(머리카락·피부) 사용: 손이 얼굴을 가릴 때 효과를 숨기는 데 쓴다 */
  useSeg: boolean;
  /** 머리카락·피부 경계를 원본 해상도로 정밀화 */
  refine: boolean;
  debugSeg: boolean;
  /** 개발용: 얼굴 점 표시 */
  debugLandmarks: boolean;
}

export const DEFAULT_SETTINGS: EngineSettings = {
  useSeg: true,
  refine: true,
  debugSeg: false,
  debugLandmarks: false,
};

export class BeautyEngine {
  readonly source: VideoSource;
  readonly tracker: Tracker;
  readonly face = new FaceTracker();
  readonly renderer: Renderer;
  readonly metrics = new Metrics();
  settings: EngineSettings = { ...DEFAULT_SETTINGS };
  /** 지금 적용 중인 메이크업 */
  look: MakeupLook = {};
  /** 지금 적용 중인 헤어 컬러(없으면 null) */
  hair: HairLook | null = null;
  loopback: LoopbackTest | null = null;
  lastFace: FaceFrame | null = null;
  lastRegions: FaceRegions | null = null;
  lastTrack: TrackResult | null = null;
  private readonly makeup: MakeupRenderer;
  private readonly hairFx: HairColorRenderer;
  private readonly overlay: CanvasRenderingContext2D;
  private readonly lumaCtx: CanvasRenderingContext2D;
  private loopRaf = 0;

  constructor(opts: { video: HTMLVideoElement; canvas: HTMLCanvasElement; overlay: HTMLCanvasElement; wasmBase: string }) {
    this.source = new VideoSource(opts.video);
    this.renderer = new Renderer(opts.canvas);
    this.makeup = new MakeupRenderer(this.renderer.gl);
    this.hairFx = new HairColorRenderer(this.renderer.gl);
    this.tracker = new Tracker(opts.wasmBase);
    this.overlay = opts.overlay.getContext('2d')!;
    const luma = document.createElement('canvas');
    luma.width = 48;
    luma.height = 27;
    this.lumaCtx = luma.getContext('2d', { willReadFrequently: true })!;
  }

  start(): void {
    this.metrics.reset();
    this.face.reset();
    this.source.onFrame((info) => this.onFrame(info));
  }

  /** 시험용: 멈춘 영상의 현재 프레임을 설정만 바꿔 다시 처리·합성한다. */
  redraw(): void {
    this.onFrame({ now: performance.now(), mediaTime: this.source.video.currentTime });
  }

  private onFrame(info: FrameInfo): void {
    const video = this.source.video;
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) return;

    if (this.loopback?.running) {
      this.loopback.onCameraFrame(info.now, this.sampleLuma(video));
      return;
    }

    const t0 = performance.now();
    const skipped = this.metrics.skippedSince(info.presentedFrames);
    this.renderer.resize(w, h);

    let track: TrackResult | null = null;
    if (!this.tracker.loading) {
      try {
        track = this.tracker.track(video, info.now);
      } catch (err) {
        console.error('추적 오류', err);
      }
    }
    this.lastTrack = track;
    const face = this.face.update(track?.face ?? null, info.now, w, h);
    this.lastFace = face;

    this.renderer.uploadCamera(video);
    if (track?.seg) this.renderer.uploadSeg(track.seg.flags, track.seg.width, track.seg.height);

    const t1 = performance.now();
    const regions = face ? faceRegions(face.p) : null;
    this.lastRegions = regions;
    const effects: ((t: FrameTextures) => void)[] = [];
    const hair = this.hair;
    if (hair && this.settings.useSeg) {
      effects.push((t) => this.hairFx.draw(hair, t, hairSpan(face, h)));
    }
    if (face && regions && hasAnyMakeup(this.look)) {
      effects.push((t) => this.makeup.draw(regions, this.look, t.cam, t.seg, t, face.confidence));
    }
    const t2 = performance.now();
    this.renderer.draw({ useSeg: this.settings.useSeg, refine: this.settings.refine, debugSeg: this.settings.debugSeg, effects });
    this.drawOverlay(face, w, h);
    const t3 = performance.now();

    this.metrics.push({
      t: info.now,
      proc: t3 - t0,
      pose: track?.timings.pose ?? 0,
      seg: track?.timings.seg ?? 0,
      hands: track?.timings.hands ?? 0,
      face: track?.timings.face ?? 0,
      rig: t2 - t1,
      render: t3 - t2,
      age: info.captureTime !== undefined && info.captureTime > 0 ? t3 - info.captureTime : undefined,
      skipped,
      person: face !== null,
    });
  }

  private drawOverlay(face: FaceFrame | null, w: number, h: number): void {
    const ctx = this.overlay;
    if (ctx.canvas.width !== w || ctx.canvas.height !== h) {
      ctx.canvas.width = w;
      ctx.canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (!this.settings.debugLandmarks || !face) return;
    ctx.fillStyle = 'rgba(0,255,180,0.8)';
    const r = Math.max(1, w / 900);
    for (const p of face.p) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ---- 거울 루프백 지연 측정 ----

  startLoopback(onDone: (lb: LoopbackTest) => void): void {
    this.stopLoopback();
    const lb = new LoopbackTest();
    this.loopback = lb;
    lb.start(performance.now());
    const tick = (t: number): void => {
      if (this.loopback !== lb) return;
      if (!lb.running) {
        onDone(lb);
        return;
      }
      this.renderer.drawSolid(lb.onDraw(t) === 1);
      this.loopRaf = requestAnimationFrame(tick);
    };
    this.loopRaf = requestAnimationFrame(tick);
  }

  stopLoopback(): void {
    cancelAnimationFrame(this.loopRaf);
    this.loopback?.cancel();
    this.loopback = null;
  }

  private sampleLuma(video: HTMLVideoElement): number {
    const ctx = this.lumaCtx;
    const { width: w, height: h } = ctx.canvas;
    ctx.drawImage(video, 0, 0, w, h);
    // 가장자리는 거울 테두리 등이 섞이므로 가운데 절반만 사용
    const d = ctx.getImageData(w / 4, h / 4, w / 2, h / 2).data;
    let sum = 0;
    for (let i = 0; i < d.length; i += 4) sum += 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    return sum / (d.length / 4);
  }

  stop(): void {
    this.stopLoopback();
    this.source.close();
  }
}

/** 머리카락 세로 범위(uv): 뿌리 = 이마 위, 끝 = 턱 아래. 얼굴이 없으면 화면 기준 기본값 */
function hairSpan(face: FaceFrame | null, h: number): [number, number] {
  if (!face) return [0.1, 0.8];
  const top = face.p[10].y;
  const chin = face.p[152].y;
  const fh = chin - top;
  return [(top - fh * 0.35) / h, (chin + fh * 0.9) / h];
}

function hasAnyMakeup(look: MakeupLook): boolean {
  return !!(look.lip || look.shadow || look.blush || look.liner || look.brow);
}
