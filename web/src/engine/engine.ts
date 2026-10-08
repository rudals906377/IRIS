// 실시간 뷰티 엔진: 영상 입력 → 추적(얼굴·분할) → 효과(메이크업 등) → 합성 → 측정을 한 프레임 안에서 처리한다.
// 원칙: 가장 최신 프레임만 처리하고(밀린 프레임은 건너뜀), 추적한 바로 그 프레임 위에 효과를 합성한다.
// 영상은 이 기기 안에서만 처리되며 어디에도 전송하지 않는다.

import { faceRegions, type FaceRegions } from '../beauty/face-regions.ts';
import { HairColorRenderer, type HairLook } from '../beauty/hair.ts';
import { MakeupRenderer, type MakeupLook } from '../beauty/makeup.ts';
import { NailRenderer, type NailLook } from '../beauty/nail.ts';
import { nailQuads, type HandPoints, type NailQuad } from '../beauty/nail-place.ts';
import { HairNet } from './hairnet.ts';
import { TattooRenderer } from '../beauty/tattoo.ts';
import { FACE_FOR_POSE, measureLimbWidth, placeAxis, placeFromPoint, tattooMesh, type PosePoints, type TattooPlace } from '../beauty/tattoo-place.ts';
import { PointFilter, RigidShapeFilter, type RigidShapeParams } from './filters.ts';
import { FaceTracker, type FaceFrame } from './face.ts';
import { LoopbackTest } from './loopback.ts';
import { Metrics } from './metrics.ts';
import { Renderer, type FrameTextures } from './renderer.ts';
import { VideoSource, type FrameInfo } from './source.ts';
import { Tracker, type TrackResult } from './tracker.ts';


/** 손 점 떨림 필터: 손바닥(손목·손가락 뿌리)을 기준 강체로, 손가락 움직임은 얼굴 안 모양처럼 따로 */
export const HAND_FILTER: RigidShapeParams = {
  rigid: [0, 1, 5, 9, 13, 17],
  poseD0: 0.008,
  poseD1: 0.035,
  poseAMin: 0.06,
  shapeD0: 0.01,
  shapeD1: 0.035,
  shapeAMin: 0.08,
};
export interface EngineSettings {
  /** 분할(머리카락·피부) 사용: 손이 얼굴을 가릴 때 효과를 숨기는 데 쓴다 */
  useSeg: boolean;
  /** 머리카락·피부 경계를 원본 해상도로 정밀화 */
  refine: boolean;
  /** 개발용: true = 분할 결과를 색으로 덧칠, 'prob' = 확률 그대로(R 머리카락, G 몸 피부, B 얼굴 피부; 평가 도구가 읽는다) */
  debugSeg: boolean | 'prob';
  /** 개발용: 얼굴 점 표시 */
  debugLandmarks: boolean;
}

export const DEFAULT_SETTINGS: EngineSettings = {
  useSeg: true,
  refine: true,
  debugSeg: false,
  debugLandmarks: false,
};

export interface TattooSettings {
  design: { canvas: TexImageSource; aspect: number; multiply?: boolean };
  place: TattooPlace;
  /** 0~1 */
  size: number;
  /** 0~1 잉크 진하기 */
  amount: number;
  /** 검은 도안의 잉크 색 */
  ink: [number, number, number];
  /** 부위 안 위치(사용자가 화면을 눌러 고른 자리). 없으면 가운데 */
  along?: number;
  across?: number;
}

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
  /** 지금 적용 중인 타투(없으면 null). 켜려면 추적기의 자세 추적(pose)이 켜져 있어야 한다 */
  tattoo: TattooSettings | null = null;
  lastPose: PosePoints | null = null;
  /** 지금 적용 중인 네일(없으면 null). 켜려면 추적기의 손 추적(hands)이 켜져 있어야 한다 */
  nail: NailLook | null = null;
  lastHands: HandPoints[] = [];
  loopback: LoopbackTest | null = null;
  lastFace: FaceFrame | null = null;
  lastRegions: FaceRegions | null = null;
  lastTrack: TrackResult | null = null;
  private readonly makeup: MakeupRenderer;
  private readonly hairFx: HairColorRenderer;
  private readonly tattooFx: TattooRenderer;
  private readonly nailFx: NailRenderer;
  /** 자체 손톱 모델(선택): 손마다 하나씩 실행기 */
  nailNets: HairNet[] = [];
  private nailCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private nailCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private nailModelUrl = '';
  /** 손 종류('Left'/'Right')별 21점 필터 */
  private handTracks: { filter: RigidShapeFilter; c: { x: number; y: number } }[] = [];
  private readonly poseFilters = new Map<number, PointFilter>();
  private poseSeen = -1;
  private limbWidth: { place: TattooPlace; w: number } | null = null;
  private readonly overlay: CanvasRenderingContext2D;
  private readonly lumaCtx: CanvasRenderingContext2D;
  private loopRaf = 0;
  private captureWaiters: ((r: { image: ImageData; face: FaceFrame | null }) => void)[] = [];

  constructor(opts: { video: HTMLVideoElement; canvas: HTMLCanvasElement; overlay: HTMLCanvasElement; wasmBase: string }) {
    this.source = new VideoSource(opts.video);
    this.renderer = new Renderer(opts.canvas);
    this.makeup = new MakeupRenderer(this.renderer.gl);
    this.hairFx = new HairColorRenderer(this.renderer.gl);
    this.tattooFx = new TattooRenderer(this.renderer.gl);
    this.nailFx = new NailRenderer(this.renderer.gl);
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
    const regions = face ? faceRegions(face.p, { overlip: this.look.lip?.over, lipStyle: this.look.lip?.style, blushPos: this.look.blush?.pos, blushSize: this.look.blush?.size }) : null;
    this.lastRegions = regions;
    const effects: ((t: FrameTextures) => void)[] = [];
    const hands = this.updateHands(track, info.now, w, h);
    const nail = this.nail;
    if (nail && hands.length > 0) {
      const quads = hands.flatMap((hp, hi) => nailQuads(hp).map((q) => ({ ...q, hand: hi })));
      if (this.nailNets.length) this.runNailNets(video, quads);
      else this.nailFx.setMasks([]);
      effects.push((t) => this.nailFx.draw(quads, nail, t));
    }
    const pose = this.updatePose(track, info.now, w, h, face);
    const tat = this.tattoo;
    if (tat && pose) {
      const width = this.updateLimbWidth(tat.place, pose, track, w, h);
      const mesh = tattooMesh(tat.place, pose, { size: tat.size, aspect: tat.design.aspect, width, along: tat.along, across: tat.across });
      if (mesh) {
        this.tattooFx.setDesign(tat.design.canvas);
        effects.push((t) => this.tattooFx.draw(mesh, t, tat.amount, tat.ink, tat.design.multiply));
      }
    }
    const hair = this.hair;
    if (hair && this.settings.useSeg) {
      effects.push((t) => this.hairFx.draw(hair, t, hairSpan(face, h)));
    }
    if (face && regions && hasAnyMakeup(this.look)) {
      effects.push((t) => this.makeup.draw(regions, this.look, t.cam, t.seg, t, face.confidence));
    }
    const t2 = performance.now();
    this.renderer.draw({ useSeg: this.settings.useSeg, refine: this.settings.refine, debugSeg: this.settings.debugSeg, effects });
    if (this.captureWaiters.length > 0) {
      // 합성 결과를 읽어 달라는 요청(사진 따라하기의 되먹임 측정용)
      const image = this.renderer.readFrame();
      const ws = this.captureWaiters;
      this.captureWaiters = [];
      for (const cb of ws) cb({ image, face });
    }
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

  /** 다음에 그리는 합성 화면과 그때의 얼굴 점을 받는다. */
  captureNext(): Promise<{ image: ImageData; face: FaceFrame | null }> {
    return new Promise((res) => this.captureWaiters.push(res));
  }

  /** 몸 관절점(0~16번)을 픽셀 좌표로 바꾸고 떨림을 줄인다. 잠깐 놓치면 마지막 값을 유지. */
  private updatePose(track: TrackResult | null, t: number, w: number, h: number, face: FaceFrame | null): PosePoints | null {
    const lm = track?.pose;
    const p: PosePoints['p'] = {};
    const vis: PosePoints['vis'] = {};
    if (lm) {
      for (let i = 0; i <= 16 && i < lm.length; i++) {
        let f = this.poseFilters.get(i);
        if (!f) {
          f = new PointFilter({ minCutoff: 1.0, beta: 0.02, dCutoff: 1.0 });
          this.poseFilters.set(i, f);
        }
        p[i] = f.filter(lm[i].x * w, lm[i].y * h, t);
        vis[i] = lm[i].visibility ?? 1;
      }
      this.poseSeen = t;
    } else if (this.lastPose && t - this.poseSeen <= 300) {
      // 잠깐 놓치면 마지막 관절점을 유지
      for (let i = 0; i <= 16; i++) {
        if (this.lastPose.p[i]) {
          p[i] = this.lastPose.p[i];
          vis[i] = this.lastPose.vis[i];
        }
      }
    }
    // 얼굴 점(이미 떨림을 줄인 값)에서 목 위치용 보조점: 어깨가 화면 밖이어도 목 타투는 된다
    if (face && face.confidence > 0.5) {
      for (const [k, fi] of Object.entries(FACE_FOR_POSE)) {
        p[Number(k)] = face.p[fi];
        vis[Number(k)] = face.confidence;
      }
    }
    this.lastPose = Object.keys(p).length > 0 ? { p, vis } : null;
    return this.lastPose;
  }

  /**
   * 손 점을 픽셀 좌표로 바꾸고 떨림을 줄인다.
   * 필터는 '왼손/오른손' 판정이 아니라 위치로 짝짓는다(판정이 프레임마다 뒤바뀌면 다른 손의 필터를 쓰게 되므로).
   */
  private updateHands(track: TrackResult | null, t: number, w: number, h: number): HandPoints[] {
    const r = track?.hands;
    if (!r) return this.lastHands;
    const out: HandPoints[] = [];
    const prev = this.handTracks;
    const next: typeof prev = [];
    r.landmarks.forEach((lm, i) => {
      const handed = r.handedness[i]?.[0]?.categoryName ?? 'Right';
      const raw = lm.map((q) => ({ x: q.x * w, y: q.y * h }));
      // 손 크기: 손목 → 가운데 손가락 뿌리 거리의 2배(얼굴 폭과 비슷한 크기)
      const size = 2 * Math.hypot(raw[9].x - raw[0].x, raw[9].y - raw[0].y) || 1;
      const c = [0, 5, 9, 13, 17].reduce((a, j) => ({ x: a.x + raw[j].x / 5, y: a.y + raw[j].y / 5 }), { x: 0, y: 0 });
      // 앞 프레임에서 가장 가까운(손 크기 안) 손의 필터를 이어 쓴다
      let best = -1;
      let bestD = size;
      prev.forEach((pt, k) => {
        const d = Math.hypot(pt.c.x - c.x, pt.c.y - c.y);
        if (d < bestD && !next.includes(pt)) {
          best = k;
          bestD = d;
        }
      });
      const tr = best >= 0 ? prev[best] : { filter: new RigidShapeFilter(HAND_FILTER), c };
      tr.c = c;
      next.push(tr);
      out.push({ p: tr.filter.filter(raw, size, t), z: lm.map((q) => q.z), handed });
    });
    // 사라진 손의 필터는 버려서 다시 나타날 때 옛 위치에서 끌려오지 않게
    this.handTracks = next;
    this.lastHands = out;
    return out;
  }

  /** 팔·목 굵기를 분할에서 재고 천천히 따라가게 평활한다(분할이 없으면 null → 길이 비율로 추정). */
  private updateLimbWidth(place: TattooPlace, pose: PosePoints, track: TrackResult | null, w: number, h: number): number | null {
    if (this.limbWidth && this.limbWidth.place !== place) this.limbWidth = null;
    const seg = track?.seg;
    const ax = placeAxis(place, pose);
    if (seg && ax) {
      const m = measureLimbWidth(seg.flags, seg.width, seg.height, w, h, ax.a, ax.b);
      if (m !== null) {
        const len = Math.hypot(ax.b.x - ax.a.x, ax.b.y - ax.a.y);
        // 말이 안 되는 값(옷·배경 섞임)은 버린다
        if (m > len * 0.12 && m < len * 1.2) {
          this.limbWidth = this.limbWidth ? { place, w: this.limbWidth.w * 0.85 + m * 0.15 } : { place, w: m };
        }
      }
    }
    return this.limbWidth?.w ?? null;
  }

  /** 화면(영상 px)에서 누른 점 → 타투 부위와 그 안의 위치. 몸 관절이 안 보이거나 부위 밖이면 null */
  tattooPlaceAt(x: number, y: number): { place: TattooPlace; along: number; across: number } | null {
    const pose = this.lastPose;
    if (!pose) return null;
    return placeFromPoint(pose, { x, y }, (place) => (this.limbWidth?.place === place ? this.limbWidth.w : null));
  }

  /** 자체 손톱 모델 켜기/끄기(주소가 비면 끈다). 두 손 분량의 실행기를 만든다 */
  async setNailModel(url: string, onStatus?: (s: string) => void): Promise<void> {
    if (url === this.nailModelUrl) return;
    this.nailModelUrl = url;
    for (const n of this.nailNets) n.close();
    this.nailNets = [];
    if (!url) return;
    const nets = [new HairNet(url), new HairNet(url)];
    try {
      await nets[0].load(onStatus);
      await nets[1].load();
      this.nailNets = nets;
      onStatus?.(`손톱 모델 준비(${nets[0].provider})`);
    } catch (e) {
      console.warn(e);
      onStatus?.('손톱 모델을 못 불러와 손 점 방식을 씁니다');
    }
  }

  /** 손마다 손톱들을 감싸는 정사각형(학습 때와 같은 비율: 손톱 상자의 2.6배)을 잘라 모델에 넣고, 최근 결과를 셰이더에 준다 */
  private runNailNets(video: HTMLVideoElement, quads: NailQuad[]): void {
    const W = video.videoWidth;
    const H = video.videoHeight;
    const out: { data: Uint8ClampedArray; size: number; rect: { x: number; y: number; size: number } }[] = [];
    for (let hi = 0; hi < 2; hi++) {
      const net = this.nailNets[hi];
      const qs = quads.filter((q) => q.hand === hi);
      if (!net || qs.length === 0) continue;
      let x0 = Infinity;
      let y0 = Infinity;
      let x1 = -Infinity;
      let y1 = -Infinity;
      for (const q of qs) {
        x0 = Math.min(x0, q.c.x - q.width);
        y0 = Math.min(y0, q.c.y - q.len);
        x1 = Math.max(x1, q.c.x + q.width);
        y1 = Math.max(y1, q.c.y + q.len);
      }
      const size = Math.max(Math.max(x1 - x0, y1 - y0) * 2.6, Math.min(W, H) * 0.25);
      const cx = (x0 + x1) / 2;
      const cy = (y0 + y1) / 2;
      const rect = { x: cx - size / 2, y: cy - size / 2, size };
      const N = net.size;
      if (!this.nailCanvas || this.nailCanvas.width !== N) {
        this.nailCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(N, N) : document.createElement('canvas');
        this.nailCanvas.width = N;
        this.nailCanvas.height = N;
        this.nailCtx = this.nailCanvas.getContext('2d', { willReadFrequently: true }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
      }
      // 영상 밖으로 나간 부분은 검게(모델 입력 크기 고정)
      this.nailCtx!.fillStyle = '#000';
      this.nailCtx!.fillRect(0, 0, N, N);
      this.nailCtx!.drawImage(video, rect.x, rect.y, size, size, 0, 0, N, N);
      net.submit(this.nailCtx!, rect);
      if (net.last) out[hi] = { data: net.last.alpha, size: net.last.size, rect: (net.last.tag as typeof rect) ?? rect };
    }
    this.nailFx.setMasks(out);
  }

  private drawOverlay(face: FaceFrame | null, w: number, h: number): void {
    const ctx = this.overlay;
    if (ctx.canvas.width !== w || ctx.canvas.height !== h) {
      ctx.canvas.width = w;
      ctx.canvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);
    if (!this.settings.debugLandmarks) return;
    const r = Math.max(1, w / 900);
    if (face) {
      ctx.fillStyle = 'rgba(0,255,180,0.8)';
      for (const p of face.p) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // 손 점 21개: 점과 번호(손톱 위치 맞추기용)
    for (const hd of this.lastHands) {
      ctx.strokeStyle = 'rgba(255,220,0,0.9)';
      ctx.lineWidth = r;
      const chains = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
      for (const ch of chains) {
        ctx.beginPath();
        ch.forEach((i, k) => (k === 0 ? ctx.moveTo(hd.p[i].x, hd.p[i].y) : ctx.lineTo(hd.p[i].x, hd.p[i].y)));
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(255,60,60,0.95)';
      ctx.font = `${Math.round(r * 10)}px sans-serif`;
      hd.p.forEach((p, i) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillText(String(i), p.x + r * 3, p.y - r * 3);
      });
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
  return !!(look.base || look.contour || look.lip || look.shadow || look.blush || look.liner || look.brow);
}
