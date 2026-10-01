// MediaPipe Tasks(웹) 추적기 모음: 자세(Pose), 다중 클래스 분할(머리카락·피부), 손, 얼굴(선택).
// GPU(WebGL) 위임을 먼저 시도하고 실패하면 CPU로 전환한다.

import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  ImageSegmenter,
  PoseLandmarker,
  type HandLandmarkerResult,
  type NormalizedLandmark,
} from '@mediapipe/tasks-vision';

const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';

export const POSE_MODELS = {
  lite: `${MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  full: `${MODEL_BASE}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  heavy: `${MODEL_BASE}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
} as const;
export type PoseModel = keyof typeof POSE_MODELS;

const SEG_MODEL = `${MODEL_BASE}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite`;
const HAND_MODEL = `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`;
const FACE_MODEL = `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;

export type Delegate = 'GPU' | 'CPU';

export interface SegOutput {
  /** RGBA8 확률(0~255): R 머리카락, G 몸 피부(손·팔·목), B 얼굴 피부, A 사람(배경 아님) */
  flags: Uint8ClampedArray;
  width: number;
  height: number;
  /** 머리 주변을 따로 분할해 합쳤으면 그 영역(0~1 비율) */
  headRect?: { x: number; y: number; w: number; h: number };
}

export interface TrackTimings {
  pose: number;
  seg: number;
  hands: number;
  face: number;
}

export interface TrackResult {
  pose: NormalizedLandmark[] | null;
  seg: SegOutput | null;
  hands: HandLandmarkerResult | null;
  /** 얼굴 점 478개(정규화 좌표). 얼굴 추적을 켰고 얼굴이 보일 때만 */
  face: NormalizedLandmark[] | null;
  timings: TrackTimings;
}

export interface TrackerConfig {
  /** 몸 관절점 추적(타투 등 몸 효과) */
  pose: boolean;
  poseModel: PoseModel;
  delegate: Delegate;
  segmentation: boolean;
  /** 분할을 몇 프레임마다 돌릴지(1 = 매 프레임). */
  segEvery: number;
  /** 머리 주변을 따로 잘라 다시 분할해 머리카락 경계를 세밀하게(얼굴이 보일 때만) */
  headSeg: boolean;
  hands: boolean;
  /** 얼굴 점 추적(메이크업) */
  face: boolean;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  pose: false,
  poseModel: 'full',
  delegate: 'GPU',
  segmentation: true,
  segEvery: 1,
  headSeg: true,
  hands: false,
  face: true,
};

// 다중 클래스 셀피 모델의 클래스 번호(모델 카드 기준)
const CLS = { background: 0, hair: 1, bodySkin: 2, faceSkin: 3, clothes: 4, others: 5 } as const;
const SEG_W = 256;
const SEG_H = 144;
/** 머리 주변 잘라 분할하는 크기(모델 입력과 같은 정사각형)와 합친 결과의 폭 */
const HEAD_W = 256;
const HI_W = 512;

export class Tracker {
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  private pose: PoseLandmarker | null = null;
  private seg: ImageSegmenter | null = null;
  /** 머리 주변만 잘라 다시 분할하는 두 번째 분할기(머리카락 올 해상도를 높인다) */
  private segHead: ImageSegmenter | null = null;
  private headCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private headCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private lastFaceLm: NormalizedLandmark[] | null = null;
  private lastGlobal: { flags: Uint8ClampedArray; w: number; h: number } | null = null;
  private headTs = 0;
  private hands: HandLandmarker | null = null;
  private face: FaceLandmarker | null = null;
  private configured = false;
  private config: TrackerConfig = { ...DEFAULT_TRACKER_CONFIG };
  private activeDelegate: Delegate = 'GPU';
  private segCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private segCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;
  private frameNo = 0;
  private lastSeg: SegOutput | null = null;
  private lastTs = 0;
  private readonly wasmBase: string;
  loading = false;

  constructor(wasmBase: string) {
    this.wasmBase = wasmBase;
  }

  get delegate(): Delegate {
    return this.activeDelegate;
  }

  get current(): TrackerConfig {
    return this.config;
  }

  /** 설정을 적용한다. 모델·위임이 바뀌면 해당 작업만 다시 만든다. */
  async configure(next: TrackerConfig, onStatus?: (msg: string) => void): Promise<void> {
    this.loading = true;
    try {
      if (!this.fileset) {
        onStatus?.('추적 엔진(WASM) 불러오는 중…');
        this.fileset = await FilesetResolver.forVisionTasks(this.wasmBase);
      }
      const prev = this.config;
      const delegateChanged = next.delegate !== prev.delegate || !this.configured;
      this.configured = true;
      if (!next.pose && this.pose) {
        this.pose.close();
        this.pose = null;
      } else if (next.pose && (delegateChanged || next.poseModel !== prev.poseModel || !this.pose)) {
        onStatus?.(`자세 모델(${next.poseModel}) 불러오는 중…`);
        this.pose?.close();
        this.pose = await this.create((delegate) =>
          PoseLandmarker.createFromOptions(this.fileset!, {
            baseOptions: { modelAssetPath: POSE_MODELS[next.poseModel], delegate },
            runningMode: 'VIDEO',
            numPoses: 1,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
          next.delegate,
        );
      }
      if (next.segmentation && (!this.seg || delegateChanged)) {
        onStatus?.('분할 모델 불러오는 중…');
        this.seg?.close();
        this.segHead?.close();
        const makeSeg = (delegate: Delegate): Promise<ImageSegmenter> =>
          ImageSegmenter.createFromOptions(this.fileset!, {
            baseOptions: { modelAssetPath: SEG_MODEL, delegate },
            runningMode: 'VIDEO',
            // 확률(0~1) 마스크를 받아야 경계가 계단 없이 부드럽다.
            outputCategoryMask: false,
            outputConfidenceMasks: true,
          });
        this.seg = await this.create(makeSeg, next.delegate);
        // 같은 모델을 하나 더: 머리 주변 잘라낸 그림 전용(시각이 따로 흐르므로 분리)
        this.segHead = await this.create(makeSeg, next.delegate);
      } else if (!next.segmentation && this.seg) {
        this.seg.close();
        this.seg = null;
        this.segHead?.close();
        this.segHead = null;
        this.lastSeg = null;
        this.lastGlobal = null;
      }
      if (next.hands && (!this.hands || delegateChanged)) {
        onStatus?.('손 모델 불러오는 중…');
        this.hands?.close();
        this.hands = await this.create((delegate) =>
          HandLandmarker.createFromOptions(this.fileset!, {
            baseOptions: { modelAssetPath: HAND_MODEL, delegate },
            runningMode: 'VIDEO',
            numHands: 2,
          }),
          next.delegate,
        );
      } else if (!next.hands && this.hands) {
        this.hands.close();
        this.hands = null;
      }
      if (next.face && (!this.face || delegateChanged)) {
        onStatus?.('얼굴 모델 불러오는 중…');
        this.face?.close();
        this.face = await this.create((delegate) =>
          FaceLandmarker.createFromOptions(this.fileset!, {
            baseOptions: { modelAssetPath: FACE_MODEL, delegate },
            runningMode: 'VIDEO',
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
          }),
          next.delegate,
        );
      } else if (!next.face && this.face) {
        this.face.close();
        this.face = null;
      }
      this.config = { ...next };
      onStatus?.('');
    } finally {
      this.loading = false;
    }
  }

  private async create<T>(make: (d: Delegate) => Promise<T>, want: Delegate): Promise<T> {
    if (want === 'GPU') {
      try {
        const t = await make('GPU');
        this.activeDelegate = 'GPU';
        return t;
      } catch (err) {
        console.warn('GPU 위임 실패, CPU로 전환합니다.', err);
      }
    }
    this.activeDelegate = 'CPU';
    return make('CPU');
  }

  /** 한 프레임을 처리한다. ts는 단조 증가하는 밀리초 시각이어야 한다. */
  track(video: HTMLVideoElement, ts: number): TrackResult {
    const timings: TrackTimings = { pose: 0, seg: 0, hands: 0, face: 0 };
    // MediaPipe VIDEO 모드는 시각이 엄격히 증가해야 하고, 내부적으로 ms 단위로 잘린다.
    // 같은 1ms 안에 두 프레임이 오면 그래프가 오류로 멈추므로 최소 1ms씩 증가시킨다.
    const t = Math.max(Math.round(ts), this.lastTs + 1);
    this.lastTs = t;
    this.frameNo++;

    let pose: NormalizedLandmark[] | null = null;
    if (this.pose) {
      const t0 = performance.now();
      const r = this.pose.detectForVideo(video, t);
      timings.pose = performance.now() - t0;
      pose = r.landmarks.length > 0 ? r.landmarks[0] : null;
    }

    let seg: SegOutput | null = this.lastSeg;
    if (this.seg) {
      const t0 = performance.now();
      const head = this.config.headSeg ? this.headRect(video.videoWidth, video.videoHeight) : null;
      // 전체 분할은 segEvery 프레임마다(머리 분할을 할 때는 3배로 띄엄띄엄: 손·몸 피부는 천천히 변한다),
      // 머리 주변 분할은 얼굴이 보이면 매 프레임
      const every = Math.max(1, this.config.segEvery) * (head ? 3 : 1);
      if (!this.lastGlobal || this.frameNo % every === 0) this.runSeg(video, t);
      if (this.lastGlobal) {
        const hair = head ? this.runHeadSeg(video, head) : null;
        seg = this.merge(this.lastGlobal, hair, head, video.videoWidth, video.videoHeight);
        this.lastSeg = seg;
      }
      timings.seg = performance.now() - t0;
    }

    let hands: HandLandmarkerResult | null = null;
    if (this.hands) {
      const t0 = performance.now();
      hands = this.hands.detectForVideo(video, t);
      timings.hands = performance.now() - t0;
    }
    let face: NormalizedLandmark[] | null = null;
    if (this.face) {
      const t0 = performance.now();
      const r = this.face.detectForVideo(video, t);
      timings.face = performance.now() - t0;
      face = r.faceLandmarks.length > 0 ? r.faceLandmarks[0] : null;
    }
    this.lastFaceLm = face;
    return { pose, seg, hands, face, timings };
  }

  /** 앞 프레임 얼굴 점으로 머리 전체를 감싸는 정사각형(픽셀)을 정한다. 얼굴이 없으면 null */
  private headRect(W: number, H: number): { x: number; y: number; size: number } | null {
    const lm = this.lastFaceLm;
    if (!lm || !W || !H) return null;
    let x0 = 1;
    let y0 = 1;
    let x1 = 0;
    let y1 = 0;
    for (const q of lm) {
      if (q.x < x0) x0 = q.x;
      if (q.y < y0) y0 = q.y;
      if (q.x > x1) x1 = q.x;
      if (q.y > y1) y1 = q.y;
    }
    const fw = (x1 - x0) * W;
    const fh = (y1 - y0) * H;
    // 머리카락은 얼굴보다 위·옆으로 넓다: 얼굴 폭의 2.4배 정사각형, 중심은 얼굴 중심보다 위
    const size = Math.max(fw * 2.4, fh * 2.0);
    if (size < 64) return null;
    const cx = ((x0 + x1) / 2) * W;
    const cy = ((y0 + y1) / 2) * H - fh * 0.35;
    let x = cx - size / 2;
    let y = cy - size / 2;
    const sz = Math.min(size, W, H);
    x = Math.max(0, Math.min(W - sz, x));
    y = Math.max(0, Math.min(H - sz, y));
    return { x, y, size: sz };
  }

  /** 머리 주변을 잘라 분할하고 머리카락 확률(0~255, HEAD_W²)을 돌려준다 */
  private runHeadSeg(video: HTMLVideoElement, head: { x: number; y: number; size: number }): Uint8ClampedArray | null {
    if (!this.segHead) return null;
    if (!this.headCanvas) {
      this.headCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(HEAD_W, HEAD_W) : document.createElement('canvas');
      this.headCanvas.width = HEAD_W;
      this.headCanvas.height = HEAD_W;
      this.headCtx = this.headCanvas.getContext('2d', { willReadFrequently: false }) as CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
    }
    this.headCtx!.drawImage(video, head.x, head.y, head.size, head.size, 0, 0, HEAD_W, HEAD_W);
    this.headTs = Math.max(this.headTs + 1, Math.round(performance.now()));
    let out: Uint8ClampedArray | null = null;
    this.segHead.segmentForVideo(this.headCanvas as unknown as HTMLCanvasElement, this.headTs, (result) => {
      const masks = result.confidenceMasks;
      if (!masks || masks.length < 5) return;
      const hair = masks[CLS.hair].getAsFloat32Array();
      const w = masks[0].width;
      const h = masks[0].height;
      const arr = new Uint8ClampedArray(HEAD_W * HEAD_W);
      if (w === HEAD_W && h === HEAD_W) {
        for (let i = 0; i < arr.length; i++) arr[i] = hair[i] * 255;
      } else {
        for (let y = 0; y < HEAD_W; y++) {
          const sy = Math.min(h - 1, Math.floor((y * h) / HEAD_W));
          for (let x = 0; x < HEAD_W; x++) arr[y * HEAD_W + x] = hair[sy * w + Math.min(w - 1, Math.floor((x * w) / HEAD_W))] * 255;
        }
      }
      out = arr;
    });
    return out;
  }

  /**
   * 전체 분할(저해상도)을 HI_W 폭으로 키우고, 머리 영역의 머리카락(R)은 잘라 분할한 고해상도 값으로 바꾼다.
   * 영역 가장자리는 서서히 섞어 이음새가 보이지 않게 한다.
   */
  private merge(g: { flags: Uint8ClampedArray; w: number; h: number }, hair: Uint8ClampedArray | null, head: { x: number; y: number; size: number } | null, frameW: number, frameH: number): SegOutput {
    if (!hair || !head) return { flags: g.flags, width: g.w, height: g.h };
    // 잘라 분할한 결과는 머리만 크게 보여 배경(벽 무늬 등)을 머리카락으로 착각할 때가 있다.
    // 전체 분할의 '사람(A)·머리카락(R)'을 한 칸 넓힌 값으로 걸러, 사람 밖에서는 쓰지 않는다
    const gate = new Uint8ClampedArray(g.w * g.h);
    for (let y = 0; y < g.h; y++) {
      for (let x = 0; x < g.w; x++) {
        let m = 0;
        for (let dy = -1; dy <= 1; dy++) {
          const yy = Math.max(0, Math.min(g.h - 1, y + dy));
          for (let dx = -1; dx <= 1; dx++) {
            const i = (yy * g.w + Math.max(0, Math.min(g.w - 1, x + dx))) * 4;
            const v = Math.max(g.flags[i], g.flags[i + 3]);
            if (v > m) m = v;
          }
        }
        gate[y * g.w + x] = m;
      }
    }
    const W = HI_W;
    const H = Math.round((HI_W * g.h) / g.w);
    const out = new Uint8ClampedArray(W * H * 4);
    const sx = g.w / W;
    const sy = g.h / H;
    // 전체 분할을 이중선형으로 키운다
    for (let y = 0; y < H; y++) {
      const fy = Math.min(g.h - 1.001, y * sy);
      const y0 = fy | 0;
      const ty = fy - y0;
      for (let x = 0; x < W; x++) {
        const fx = Math.min(g.w - 1.001, x * sx);
        const x0 = fx | 0;
        const tx = fx - x0;
        const i00 = (y0 * g.w + x0) * 4;
        const i10 = i00 + 4;
        const i01 = i00 + g.w * 4;
        const i11 = i01 + 4;
        const o = (y * W + x) * 4;
        for (let c = 0; c < 4; c++) {
          out[o + c] = (g.flags[i00 + c] * (1 - tx) + g.flags[i10 + c] * tx) * (1 - ty) + (g.flags[i01 + c] * (1 - tx) + g.flags[i11 + c] * tx) * ty;
        }
      }
    }
    // 머리 영역: 잘라 분할한 머리카락 확률을 덮어쓴다(가장자리 8%는 섞음)
    const rx0 = (head.x / frameW) * W;
    const ry0 = (head.y / frameH) * H;
    const rw = (head.size / frameW) * W;
    const rh = (head.size / frameH) * H;
    const feather = Math.max(2, rw * 0.08);
    const xs = Math.max(0, Math.floor(rx0));
    const ys = Math.max(0, Math.floor(ry0));
    const xe = Math.min(W - 1, Math.ceil(rx0 + rw));
    const ye = Math.min(H - 1, Math.ceil(ry0 + rh));
    for (let y = ys; y <= ye; y++) {
      const v = ((y - ry0) / rh) * HEAD_W;
      const vy = Math.max(0, Math.min(HEAD_W - 1.001, v));
      const y0 = vy | 0;
      const ty = vy - y0;
      const wy = Math.min(1, (y - ry0) / feather, (ry0 + rh - y) / feather);
      if (wy <= 0) continue;
      for (let x = xs; x <= xe; x++) {
        const u = ((x - rx0) / rw) * HEAD_W;
        const ux = Math.max(0, Math.min(HEAD_W - 1.001, u));
        const x0 = ux | 0;
        const tx = ux - x0;
        const wx = Math.min(1, (x - rx0) / feather, (rx0 + rw - x) / feather);
        const wgt = Math.min(wx, wy);
        if (wgt <= 0) continue;
        const h00 = hair[y0 * HEAD_W + x0];
        const h10 = hair[y0 * HEAD_W + x0 + 1];
        const h01 = hair[(y0 + 1) * HEAD_W + x0];
        const h11 = hair[(y0 + 1) * HEAD_W + x0 + 1];
        const hp = (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty;
        const o = (y * W + x) * 4;
        // 전체 분할 기준 사람 밖(넓힌 값 < 약 0.15)이면 잘라 분할한 값을 쓰지 않는다
        const gx = Math.min(g.w - 1.001, x * sx);
        const gy = Math.min(g.h - 1.001, y * sy);
        const gx0 = gx | 0;
        const gy0 = gy | 0;
        const gtx = gx - gx0;
        const gty = gy - gy0;
        const gi = gy0 * g.w + gx0;
        const gv = (gate[gi] * (1 - gtx) + gate[gi + 1] * gtx) * (1 - gty) + (gate[gi + g.w] * (1 - gtx) + gate[gi + g.w + 1] * gtx) * gty;
        const keep = Math.max(0, Math.min(1, (gv - 25) / 60));
        out[o] = out[o] * (1 - wgt) + hp * keep * wgt;
      }
    }
    return { flags: out, width: W, height: H, headRect: { x: head.x / frameW, y: head.y / frameH, w: head.size / frameW, h: head.size / frameH } };
  }

  private runSeg(video: HTMLVideoElement, t: number): void {
    if (!this.segCanvas) {
      this.segCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(SEG_W, SEG_H) : document.createElement('canvas');
      this.segCanvas.width = SEG_W;
      this.segCanvas.height = SEG_H;
      this.segCtx = this.segCanvas.getContext('2d', { willReadFrequently: false }) as
        | CanvasRenderingContext2D
        | OffscreenCanvasRenderingContext2D;
    }
    // 작은 캔버스로 줄여서 넣어야 결과 마스크 읽기(GPU→CPU) 비용이 작다.
    this.segCtx!.drawImage(video, 0, 0, SEG_W, SEG_H);
    let out: SegOutput | null = null;
    this.seg!.segmentForVideo(this.segCanvas as unknown as HTMLCanvasElement, t, (result) => {
      const masks = result.confidenceMasks;
      if (!masks || masks.length < 5) return;
      const w = masks[0].width;
      const h = masks[0].height;
      const bg = masks[CLS.background].getAsFloat32Array();
      const hair = masks[CLS.hair].getAsFloat32Array();
      const body = masks[CLS.bodySkin].getAsFloat32Array();
      const face = masks[CLS.faceSkin].getAsFloat32Array();
      const n = w * h;
      // Uint8ClampedArray: 0~255 범위 밖 값은 잘리고 반올림된다.
      const flags = new Uint8ClampedArray(n * 4);
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const person = 1 - bg[i];
        flags[o] = hair[i] * 255;
        flags[o + 1] = body[i] * 255;
        flags[o + 2] = face[i] * 255;
        flags[o + 3] = person * 255;
      }
      out = { flags, width: w, height: h };
    });
    const o = out as SegOutput | null;
    if (o) this.lastGlobal = { flags: o.flags, w: o.width, h: o.height };
  }

  close(): void {
    this.pose?.close();
    this.seg?.close();
    this.segHead?.close();
    this.segHead = null;
    this.hands?.close();
    this.face?.close();
    this.face = null;
    this.pose = null;
    this.seg = null;
    this.hands = null;
  }
}
