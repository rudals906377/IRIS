// 사진(정지 영상)용 정밀 얼굴 분할: BiSeNet 얼굴 파싱(zllrunning/face-parsing.PyTorch, MIT, CelebAMask-HQ 19부위).
// 실시간 분할은 이마로 내려온 머리카락을 '얼굴 피부'로, 귀걸이·귀 주변을 머리카락으로 헷갈려 염색에 구멍·얼룩이 생긴다.
// 사진은 한 번만 처리하면 되므로 머리 주변을 이 모델로 한 번 나눈다.
// 모델: public/models/face-parse-512.onnx(26MB, 가중치 fp16 저장·계산 fp32). 입력 RGB 0~1(정규화는 모델 안),
// 출력 [머리카락, 얼굴(피부·눈·눈썹·코·입·귀), 목] 확률 512×512. 만든 방법은 tools/train/face_parse_export.py.
// onnxruntime-web(WASM)으로 기기 안에서 돌린다.

const ORT_VERSION = '1.30.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const MODEL_URL = 'models/face-parse-512.onnx';
const N = 512;

export interface ParseResult {
  /** 0~255, size×size, 잘라 낸 정사각형 기준 */
  hair: Uint8ClampedArray;
  face: Uint8ClampedArray;
  body: Uint8ClampedArray;
  size: number;
  /** 모델에 넣은 잘라 낸 그림(RGBA, size×size) */
  rgba: Uint8ClampedArray;
}

type Session = {
  inputNames: string[];
  outputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, { data: Float32Array; dims: number[] }>>;
};
type Ort = {
  env: { wasm: { wasmPaths: string } };
  InferenceSession: { create(url: string, opts: { executionProviders: string[] }): Promise<Session> };
  Tensor: new (type: 'float32', data: Float32Array, dims: number[]) => unknown;
};

let loading: Promise<{ ort: Ort; session: Session }> | null = null;

function load(onStatus?: (s: string) => void) {
  loading ??= (async () => {
    onStatus?.('정밀 얼굴 분할 모델 불러오는 중… (처음 한 번 26MB)');
    const ort = (await import(/* @vite-ignore */ `${ORT_BASE}ort.webgpu.bundle.min.mjs`)) as Ort;
    ort.env.wasm.wasmPaths = ORT_BASE;
    const url = new URL(MODEL_URL, document.baseURI).href;
    // 사진 한 장에 한 번만 돌리므로 WASM 으로 충분하다(512 한 장 약 1.5초). WebGPU 는 소프트웨어 GPU 환경에서 화면이 멈춘 적이 있어 쓰지 않는다
    return { ort, session: await ort.InferenceSession.create(url, { executionProviders: ['wasm'] }) };
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** 그림의 정사각형 영역(rect, 그림 픽셀. 그림 밖은 검게)을 512 로 잘라 부위 확률을 구한다 */
export async function parseFace(img: ImageBitmap | HTMLCanvasElement, rect: { x: number; y: number; size: number }, onStatus?: (s: string) => void): Promise<ParseResult> {
  const { ort, session } = await load(onStatus);
  const c = document.createElement('canvas');
  c.width = N;
  c.height = N;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, N, N);
  ctx.drawImage(img, rect.x, rect.y, rect.size, rect.size, 0, 0, N, N);
  const px = ctx.getImageData(0, 0, N, N).data;
  const plane = N * N;
  const x = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    x[i] = px[i * 4] / 255;
    x[plane + i] = px[i * 4 + 1] / 255;
    x[2 * plane + i] = px[i * 4 + 2] / 255;
  }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, N, N]) });
  const y = out[session.outputNames[0]].data;
  const ch = (k: number): Uint8ClampedArray => {
    const o = new Uint8ClampedArray(plane);
    for (let i = 0; i < plane; i++) o[i] = y[k * plane + i] * 255;
    return o;
  };
  return { hair: ch(0), face: ch(1), body: ch(2), size: N, rgba: px };
}
