// 영상 입력: 웹캠 또는 테스트용 동영상 파일. 둘 다 <video> 요소로 다룬다.
// 새 프레임이 도착할 때마다(requestVideoFrameCallback) 콜백을 부른다.

export interface FrameInfo {
  /** 콜백 호출 시각(performance.now 기준, ms). */
  now: number;
  /** 카메라가 프레임을 촬영한 시각(브라우저가 제공할 때만). */
  captureTime?: number;
  /** 누적 표시 프레임 수(건너뛴 프레임 계산용). */
  presentedFrames?: number;
  mediaTime: number;
}

export type SourceKind = 'camera' | 'file' | 'image';

/** 처리에 넣는 그림: 영상(<video>) 또는 사진 캔버스 */
export type FrameSource = HTMLVideoElement | HTMLCanvasElement;
/** 그림의 픽셀 크기(영상은 videoWidth, 캔버스는 width) */
export function frameSize(v: FrameSource): { w: number; h: number } {
  return v instanceof HTMLVideoElement ? { w: v.videoWidth, h: v.videoHeight } : { w: v.width, h: v.height };
}


export interface CameraRequest {
  width: number;
  height: number;
  fps: number;
  /** 'user' 앞(셀피) 카메라, 'environment' 뒤 카메라(휴대폰) */
  facing?: 'user' | 'environment';
}

export class VideoSource {
  readonly video: HTMLVideoElement;
  kind: SourceKind | null = null;
  private stream: MediaStream | null = null;
  private stopLoop: (() => void) | null = null;
  private objectUrl: string | null = null;
  private stillTimer = 0;
  private stillDraw: ((img: ImageBitmap) => void) | null = null;
  /** 사진을 열었을 때 그 그림이 그려진 캔버스(처리 입력으로 바로 쓴다) */
  private stillCanvas: HTMLCanvasElement | null = null;
  /** 사진(정지 영상)을 열었을 때 그 원본(최대 변 제한 전) */
  still: ImageBitmap | null = null;

  constructor(video: HTMLVideoElement) {
    this.video = video;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
  }

  get width(): number {
    return this.stillCanvas?.width ?? this.video.videoWidth;
  }

  get height(): number {
    return this.stillCanvas?.height ?? this.video.videoHeight;
  }

  /**
   * 처리·합성에 넣을 현재 그림. 사진은 캔버스 그대로(스트림을 거친 <video> 가 아니라):
   * 큰 사진(1500×2000)에서 브라우저 자원이 모자라면 캔버스 스트림이 조용히 멈춰(currentTime 고정) 새 프레임 콜백이 안 와
   * 엔진이 통째로 멈췄고, 구운 결과도 <video> 에 안 비쳤다
   */
  get frame(): FrameSource {
    return this.stillCanvas ?? this.video;
  }

  get settings(): MediaTrackSettings | null {
    return this.stream?.getVideoTracks()[0]?.getSettings() ?? null;
  }

  async openCamera(req: CameraRequest): Promise<void> {
    this.close();
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('이 브라우저에서는 카메라를 사용할 수 없습니다(HTTPS 필요).');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: req.facing ?? 'user',
        width: { ideal: req.width },
        height: { ideal: req.height },
        frameRate: { ideal: req.fps },
      },
    });
    this.video.srcObject = this.stream;
    this.kind = 'camera';
    await this.video.play();
    await this.waitForSize();
  }

  async openFile(fileOrUrl: File | string): Promise<void> {
    this.close();
    if (typeof fileOrUrl === 'string') {
      this.video.src = fileOrUrl;
    } else {
      this.objectUrl = URL.createObjectURL(fileOrUrl);
      this.video.src = this.objectUrl;
    }
    this.video.loop = true;
    this.kind = 'file';
    await this.video.play();
    await this.waitForSize();
  }

  /**
   * 사진 한 장을 '움직이지 않는 카메라'처럼 연다: 캔버스에 그려 스트림으로 내보낸다.
   * 그러면 얼굴 찾기·분할·화장·생성 등 영상용 처리가 모두 그대로 사진에 적용된다.
   * maxSide: 처리 속도를 위해 긴 변을 줄인다(원본은 still 에 남김).
   */
  async openImage(img: ImageBitmap, maxSide = 1600): Promise<void> {
    this.close();
    const s = Math.min(1, maxSide / Math.max(img.width, img.height));
    // 크기를 짝수로 맞추지 않는다: 홀수 변(465px) 사진을 466 으로 1px 늘리면 사진 전체가 반 픽셀씩 재샘플돼
    // 얼굴·옷까지 뿌옇게 됐음(라플라시안 분산 0.38배). 캔버스 스트림은 홀수 크기도 되고 엔진은 캔버스를 직접 읽는다
    const w = Math.max(2, Math.round(img.width * s));
    const h = Math.max(2, Math.round(img.height * s));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    let cur = img;
    ctx.drawImage(cur, 0, 0, w, h);
    this.stream = canvas.captureStream(15);
    // 그림이 그대로면 새 프레임이 안 나오는 브라우저가 있어 주기적으로 다시 그린다
    this.stillTimer = window.setInterval(() => ctx.drawImage(cur, 0, 0, w, h), 66);
    this.stillDraw = (next) => {
      cur = next;
      ctx.drawImage(cur, 0, 0, w, h);
    };
    this.still = img;
    this.stillCanvas = canvas;
    this.video.srcObject = this.stream;
    this.kind = 'image';
    await this.video.play();
    await this.waitForSize();
  }

  /** 사진 모드: 처리 중인 정지 영상을 (같은 크기의) 다른 그림으로 바꾼다. 엔진은 멈추지 않는다 */
  replaceStill(img: ImageBitmap): void {
    this.stillDraw?.(img);
  }

  private async waitForSize(): Promise<void> {
    if (this.video.videoWidth > 0) return;
    await new Promise<void>((resolve) => {
      const on = (): void => {
        this.video.removeEventListener('loadedmetadata', on);
        resolve();
      };
      this.video.addEventListener('loadedmetadata', on);
    });
  }

  /** 새 프레임마다 cb 호출. 반환 함수로 중지한다. */
  onFrame(cb: (info: FrameInfo) => void): void {
    this.stopLoop?.();
    const video = this.video;
    let stopped = false;
    if (this.stillCanvas) {
      // 사진: 스트림 프레임을 기다리지 않고 직접 돈다(스트림이 멈춰도 엔진은 계속).
      // 한 프레임 처리가 끝난 뒤 66ms 쉬고 다음(고정 간격이면 처리가 긴 큰 사진에서 메인 스레드를 다 먹어 사진 분석이 굶었다)
      let timer = 0;
      const tick = (): void => {
        if (stopped) return;
        cb({ now: performance.now(), mediaTime: performance.now() / 1000 });
        timer = window.setTimeout(tick, 66);
      };
      timer = window.setTimeout(tick, 0);
      this.stopLoop = () => {
        stopped = true;
        clearTimeout(timer);
      };
      return;
    }
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      let handle = 0;
      const step = (now: number, md: VideoFrameCallbackMetadata): void => {
        if (stopped) return;
        cb({ now, captureTime: md.captureTime, presentedFrames: md.presentedFrames, mediaTime: md.mediaTime });
        handle = video.requestVideoFrameCallback(step);
      };
      handle = video.requestVideoFrameCallback(step);
      this.stopLoop = () => {
        stopped = true;
        video.cancelVideoFrameCallback(handle);
      };
      return;
    }
    // 대체 경로: 화면 갱신마다 currentTime 변화를 확인한다.
    let last = -1;
    let raf = 0;
    const tick = (now: number): void => {
      if (stopped) return;
      if (video.currentTime !== last) {
        last = video.currentTime;
        cb({ now, mediaTime: video.currentTime });
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    this.stopLoop = () => {
      stopped = true;
      cancelAnimationFrame(raf);
    };
  }

  close(): void {
    this.stopLoop?.();
    this.stopLoop = null;
    if (this.stillTimer) clearInterval(this.stillTimer);
    this.stillTimer = 0;
    this.still = null;
    this.stillDraw = null;
    this.stillCanvas = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.srcObject = null;
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = null;
    this.kind = null;
  }
}
