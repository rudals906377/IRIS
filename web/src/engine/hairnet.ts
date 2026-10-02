// 머리카락 전용 매팅 모델(자체 학습, ONNX)을 브라우저에서 돌린다(onnxruntime-web, WebGPU → 안 되면 WASM).
// 입력: 머리 주변 정사각형 그림(캔버스) → 출력: 머리카락 알파(0~255, N×N). tracker.ts가 머리 분할 대신 쓴다.
// 모델 파일은 tools/train/hair/train.py 가 만든 hair-matte-256.onnx / hair-matte-384.onnx.

const ORT_VERSION = '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

type Ort = {
  env: { wasm: { wasmPaths: string; numThreads?: number } };
  InferenceSession: {
    create(url: string, opts: { executionProviders: string[] }): Promise<{
      inputNames: string[];
      outputNames: string[];
      run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array; dims: number[] }>>;
    }>;
  };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
};

export class HairNet {
  private ort: Ort | null = null;
  private session: Awaited<ReturnType<Ort['InferenceSession']['create']>> | null = null;
  private busy = false;
  private input: Float32Array | null = null;
  /** 마지막 결과(알파 0~255)와 그때의 입력 크기 */
  last: { alpha: Uint8ClampedArray; size: number } | null = null;
  /** 모델 입력 한 변(ONNX 파일 이름의 숫자). 기본 256 */
  readonly size: number;
  provider = '';
  lastMs = 0;

  readonly url: string;

  constructor(url: string) {
    this.url = url;
    const m = /(\d{3})\.onnx/.exec(url);
    this.size = m ? Number(m[1]) : 256;
  }

  async load(onStatus?: (s: string) => void): Promise<void> {
    if (this.session) return;
    onStatus?.('머리카락 모델 불러오는 중…');
    const ort = (await import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`)) as Ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    this.ort = ort;
    for (const ep of ['webgpu', 'wasm']) {
      try {
        this.session = await ort.InferenceSession.create(this.url, { executionProviders: [ep] });
        this.provider = ep;
        break;
      } catch (e) {
        console.warn(`머리카락 모델 ${ep} 실패`, e);
      }
    }
    if (!this.session) throw new Error('머리카락 모델을 불러오지 못했습니다');
    this.input = new Float32Array(3 * this.size * this.size);
  }

  /** 바쁘지 않으면 추론을 시작한다(비동기). 결과는 다음 프레임부터 last 에 들어온다 */
  submit(ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D): void {
    if (!this.session || !this.ort || this.busy || !this.input) return;
    const N = this.size;
    const d = ctx.getImageData(0, 0, N, N).data;
    const x = this.input;
    const plane = N * N;
    for (let i = 0; i < plane; i++) {
      x[i] = d[i * 4] / 255;
      x[plane + i] = d[i * 4 + 1] / 255;
      x[2 * plane + i] = d[i * 4 + 2] / 255;
    }
    this.busy = true;
    const t0 = performance.now();
    const feeds = { [this.session.inputNames[0]]: new this.ort.Tensor('float32', x, [1, 3, N, N]) };
    this.session
      .run(feeds)
      .then((out) => {
        const logit = out[this.session!.outputNames[0]].data;
        const alpha = this.last && this.last.size === N ? this.last.alpha : new Uint8ClampedArray(plane);
        for (let i = 0; i < plane; i++) alpha[i] = 255 / (1 + Math.exp(-logit[i]));
        this.last = { alpha, size: N };
        this.lastMs = performance.now() - t0;
      })
      .catch((e) => console.warn('머리카락 모델 추론 실패', e))
      .finally(() => {
        this.busy = false;
      });
  }

  close(): void {
    this.session = null;
    this.last = null;
  }
}
