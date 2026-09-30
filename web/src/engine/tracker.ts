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
  hands: false,
  face: true,
};

// 다중 클래스 셀피 모델의 클래스 번호(모델 카드 기준)
const CLS = { background: 0, hair: 1, bodySkin: 2, faceSkin: 3, clothes: 4, others: 5 } as const;
const SEG_W = 256;
const SEG_H = 144;

export class Tracker {
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  private pose: PoseLandmarker | null = null;
  private seg: ImageSegmenter | null = null;
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
        this.seg = await this.create((delegate) =>
          ImageSegmenter.createFromOptions(this.fileset!, {
            baseOptions: { modelAssetPath: SEG_MODEL, delegate },
            runningMode: 'VIDEO',
            // 확률(0~1) 마스크를 받아야 경계가 계단 없이 부드럽다.
            outputCategoryMask: false,
            outputConfidenceMasks: true,
          }),
          next.delegate,
        );
      } else if (!next.segmentation && this.seg) {
        this.seg.close();
        this.seg = null;
        this.lastSeg = null;
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
    if (this.seg && this.frameNo % Math.max(1, this.config.segEvery) === 0) {
      const t0 = performance.now();
      seg = this.runSeg(video, t);
      timings.seg = performance.now() - t0;
      this.lastSeg = seg;
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
    return { pose, seg, hands, face, timings };
  }

  private runSeg(video: HTMLVideoElement, t: number): SegOutput | null {
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
    return out;
  }

  close(): void {
    this.pose?.close();
    this.seg?.close();
    this.hands?.close();
    this.face?.close();
    this.face = null;
    this.pose = null;
    this.seg = null;
    this.hands = null;
  }
}
