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
import { measureTorso, TorsoTracker } from './silhouette.ts';

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
  /** 새 옷에 덮이지 않은 원래 옷(후드 모자, 긴 소매 등)을 배경·맨살로 지운다 */
  removeOriginal: boolean;
  /** 개발용: 지울 곳을 분홍색으로 표시 */
  debugRemove: boolean;
  /** 개발용: 지운 자리에 채울 색을 화면 전체에 표시 */
  debugRemoveFill: boolean;
  /** 개발용: 몸통 회전 각도(라디안)를 강제로 지정(null이면 추적값) */
  forceTurn: number | null;
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
  removeOriginal: true,
  debugRemove: false,
  debugRemoveFill: false,
  forceTurn: null,
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
  /** 착용자 몸통 윤곽(분할에서 측정, 시간 평활) */
  readonly torso = new TorsoTracker();
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

  /** 마지막으로 입힌 자산(시험 도구에서 분석 결과 확인용) */
  lastAsset: GarmentAsset | null = null;

  /** 이미 만들어진 자산(자동 분석한 상품 사진 등)을 입힌다. */
  setAsset(asset: GarmentAsset): void {
    this.loadingId = asset.info.id;
    this.lastAsset = asset;
    this.clearGarment();
    const rig = new TopRig(asset);
    this.garment = { info: asset.info, gpu: this.renderer.createGarment(asset, rig.backTorso ? [rig.backTorso] : []), rig };
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
    if (!body) this.torso.reset();
    else if (track?.seg && this.settings.useSeg) this.torso.update(measureTorso(track.seg, body), body.shoulderW);
    if (body && this.garment) {
      const rigBody = this.settings.forceTurn !== null ? { ...body, turn: this.settings.forceTurn } : body;
      this.garment.rig.update(rigBody, this.settings.fit, this.settings.useSeg ? this.torso : undefined);
      const order =
        this.settings.debugGarment === 5 ? this.garment.rig.order.filter((m) => m.partId === 2 || m.partId === 3) : this.garment.rig.order;
      layers.push({ gpu: this.garment.gpu, order, alpha: body.confidence });
      capsules = buildArmOccluders(body);
    }
    const t2 = performance.now();
    // 합성기에 넘길 몸 정보(목 보존·원래 옷 지우기)
    const bodyInfo =
      body && this.garment
        ? (() => {
            const axis = { x: body.hipMid.x - body.shoulderMid.x, y: body.hipMid.y - body.shoulderMid.y };
            const l = Math.hypot(axis.x, axis.y) || 1;
            const mid = this.torso.at(0.6, body.shoulderW);
            return {
              sm: body.shoulderMid,
              lat: body.u,
              axis: { x: axis.x / l, y: axis.y / l },
              axisLen: body.axisLen,
              torsoHalf: (mid.left + mid.right) / 2,
              pitAx: body.axisLen * 0.33,
              shoulderW: body.shoulderW,
            };
          })()
        : null;
    this.renderer.draw({
      layers,
      capsules,
      shade: this.settings.shade,
      useSeg: this.settings.useSeg,
      refine: this.settings.refine,
      debugSeg: this.settings.debugSeg,
      debugOcc: this.settings.debugOcc,
      debugGarment: this.settings.debugGarment === 5 ? 0 : this.settings.debugGarment,
      removal: bodyInfo && this.settings.removeOriginal ? { ...bodyInfo, debug: this.settings.debugRemove, debugFill: this.settings.debugRemoveFill } : undefined,
      body: bodyInfo ?? undefined,
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
