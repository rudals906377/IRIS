// 가상 착용 엔진: 영상 입력 → 추적 → 신체 상태 → 옷 변형 → 합성 → 측정을 한 프레임 안에서 처리한다.
// 원칙: 가장 최신 프레임만 처리하고(밀린 프레임은 건너뜀), 자세를 계산한 바로 그 프레임 위에 옷을 합성한다.

import { BodyTracker, LM, type BodyFrame } from './body.ts';
import { loadGarment, type GarmentAsset, type ProductInfo } from './garment.ts';
import { LoopbackTest } from './loopback.ts';
import { Metrics } from './metrics.ts';
import { buildArmOccluders } from './occluders.ts';
import { Renderer, type GarmentGPU } from './renderer.ts';
import { DEFAULT_TOP_FIT, TopRig, type TopFit } from './rig-top.ts';
import { VideoSource, type FrameInfo } from './source.ts';
import { Tracker, type TrackResult } from './tracker.ts';

export interface EngineSettings {
  /** 셰이딩 전이 강도 0~1 */
  shade: number;
  /** 분할 마스크(머리카락·피부·옷)를 가림과 셰이딩에 사용 */
  useSeg: boolean;
  /** 머리카락·피부 경계를 원본 해상도로 정밀화 */
  refine: boolean;
  debugSeg: boolean;
  debugOcc: boolean;
  debugLandmarks: boolean;
  /** 개발용: 0 정상, 1 라벨 색, 2 통과 픽셀, 3 가림 버퍼, 4 가림 원인, 5 소매만 */
  debugGarment: number;
  fit: TopFit;
}

export const DEFAULT_SETTINGS: EngineSettings = {
  shade: 0.5,
  useSeg: true,
  refine: true,
  debugSeg: false,
  debugOcc: false,
  debugLandmarks: false,
  debugGarment: 0,
  fit: { ...DEFAULT_TOP_FIT },
};

interface ActiveGarment {
  info: ProductInfo;
  gpu: GarmentGPU;
  rig: TopRig;
}

export class TryOnEngine {
  readonly source: VideoSource;
  readonly tracker: Tracker;
  readonly body = new BodyTracker();
  readonly renderer: Renderer;
  readonly metrics = new Metrics();
  settings: EngineSettings = { ...DEFAULT_SETTINGS };
  loopback: LoopbackTest | null = null;
  lastBody: BodyFrame | null = null;
  lastTrack: TrackResult | null = null;
  private garment: ActiveGarment | null = null;
  private readonly overlay: CanvasRenderingContext2D;
  private readonly lumaCtx: CanvasRenderingContext2D;
  private loopRaf = 0;
  private readonly productBase: string;
  private loadingId: string | null = null;

  constructor(opts: { video: HTMLVideoElement; canvas: HTMLCanvasElement; overlay: HTMLCanvasElement; wasmBase: string; productBase: string }) {
    this.source = new VideoSource(opts.video);
    this.renderer = new Renderer(opts.canvas);
    this.tracker = new Tracker(opts.wasmBase);
    this.overlay = opts.overlay.getContext('2d')!;
    const luma = document.createElement('canvas');
    luma.width = 48;
    luma.height = 27;
    this.lumaCtx = luma.getContext('2d', { willReadFrequently: true })!;
    this.productBase = opts.productBase;
  }

  get product(): ProductInfo | null {
    return this.garment?.info ?? null;
  }

  async setProduct(info: ProductInfo | null): Promise<void> {
    if (!info) {
      this.clearGarment();
      return;
    }
    this.loadingId = info.id;
    const asset = await loadGarment(info, this.productBase);
    if (this.loadingId !== info.id) return; // 그 사이 다른 상품이 선택됨
    this.setAsset(asset);
  }

  /** 이미 만들어진 자산(자동 분석한 상품 사진 등)을 입힌다. */
  setAsset(asset: GarmentAsset): void {
    this.loadingId = asset.info.id;
    this.clearGarment();
    this.garment = { info: asset.info, gpu: this.renderer.createGarment(asset), rig: new TopRig(asset) };
  }

  private clearGarment(): void {
    if (this.garment) this.renderer.deleteGarment(this.garment.gpu);
    this.garment = null;
  }

  start(): void {
    this.metrics.reset();
    this.body.reset();
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
    const body = this.body.update(track?.pose ?? null, info.now, w, h);
    this.lastBody = body;

    this.renderer.uploadCamera(video);
    if (track?.seg) this.renderer.uploadSeg(track.seg.flags, track.seg.width, track.seg.height);

    const t1 = performance.now();
    const layers = [];
    let capsules: ReturnType<typeof buildArmOccluders> = [];
    if (body && this.garment) {
      this.garment.rig.update(body, this.settings.fit);
      const order =
        this.settings.debugGarment === 5 ? this.garment.rig.order.filter((m) => m.partId === 2 || m.partId === 3) : this.garment.rig.order;
      layers.push({ gpu: this.garment.gpu, order, alpha: body.confidence * frontFacing(body) });
      capsules = buildArmOccluders(body);
    }
    const t2 = performance.now();
    this.renderer.draw({
      layers,
      capsules,
      shade: this.settings.shade,
      useSeg: this.settings.useSeg,
      refine: this.settings.refine,
      debugSeg: this.settings.debugSeg,
      debugOcc: this.settings.debugOcc,
      debugGarment: this.settings.debugGarment === 5 ? 0 : this.settings.debugGarment,
    });
    this.drawOverlay(body, w, h);
    const t3 = performance.now();

    this.metrics.push({
      t: info.now,
      proc: t3 - t0,
      pose: track?.timings.pose ?? 0,
      seg: track?.timings.seg ?? 0,
      hands: track?.timings.hands ?? 0,
      rig: t2 - t1,
      render: t3 - t2,
      age: info.captureTime !== undefined && info.captureTime > 0 ? t3 - info.captureTime : undefined,
      skipped,
      person: body !== null,
    });
  }

  private drawOverlay(body: BodyFrame | null, w: number, h: number): void {
    const ctx = this.overlay;
    if (ctx.canvas.width !== w || ctx.canvas.height !== h) {
      ctx.canvas.width = w;
      ctx.canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (!this.settings.debugLandmarks || !body) return;
    ctx.lineWidth = Math.max(2, w / 400);
    ctx.strokeStyle = 'rgba(0,255,180,0.8)';
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    const pairs: [number, number][] = [
      [LM.leftShoulder, LM.rightShoulder], [LM.leftShoulder, LM.leftElbow], [LM.leftElbow, LM.leftWrist],
      [LM.rightShoulder, LM.rightElbow], [LM.rightElbow, LM.rightWrist], [LM.leftShoulder, LM.leftHip],
      [LM.rightShoulder, LM.rightHip], [LM.leftHip, LM.rightHip],
    ];
    ctx.beginPath();
    for (const [a, b] of pairs) {
      if (body.vis[a] < 0.3 || body.vis[b] < 0.3) continue;
      ctx.moveTo(body.p[a].x, body.p[a].y);
      ctx.lineTo(body.p[b].x, body.p[b].y);
    }
    ctx.stroke();
    for (let i = 0; i < 33; i++) {
      if (body.vis[i] < 0.3) continue;
      ctx.beginPath();
      ctx.arc(body.p[i].x, body.p[i].y, ctx.lineWidth * 1.5, 0, Math.PI * 2);
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

/**
 * 정면에서 많이 돌아서면(상품 사진에 없는 옆·뒷면) 옷을 서서히 흐리게 한다.
 * 깊이(z) 추정은 흔들림이 커서 쓰지 않고, 2D 어깨 폭 ÷ 몸통 길이로 판단한다(정면 ≈ 0.7).
 * 등을 보이면(왼쪽 어깨가 화면 왼쪽) 바로 사라진다.
 */
function frontFacing(body: BodyFrame): number {
  const sL = body.p[LM.leftShoulder];
  const sR = body.p[LM.rightShoulder];
  if (sL.x - sR.x < body.shoulderW * 0.1) return 0;
  const hipsSeen = Math.min(body.vis[LM.leftHip], body.vis[LM.rightHip]) > 0.6;
  if (!hipsSeen) return 1;
  const r = body.shoulderW / body.axisLen;
  return Math.min(1, Math.max(0, (r - 0.25) / 0.2));
}
