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

export type SourceKind = 'camera' | 'file';

export interface CameraRequest {
  width: number;
  height: number;
  fps: number;
}

export class VideoSource {
  readonly video: HTMLVideoElement;
  kind: SourceKind | null = null;
  private stream: MediaStream | null = null;
  private stopLoop: (() => void) | null = null;
  private objectUrl: string | null = null;

  constructor(video: HTMLVideoElement) {
    this.video = video;
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
  }

  get width(): number {
    return this.video.videoWidth;
  }

  get height(): number {
    return this.video.videoHeight;
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
        facingMode: 'user',
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
