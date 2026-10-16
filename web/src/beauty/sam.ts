// 눌러서 오려 내기(Segment Anything, SlimSAM-77, Apache-2.0, 양자화 약 14MB): 사진 모드에서 손톱을 찾는다.
// 그림 한 장의 특징을 한 번 계산(수 초)해 두고, 점 하나마다 0.3초 안팎으로 그 물체의 마스크를 얻는다.
// Transformers.js 가 Hugging Face 에서 받아 브라우저에 캐시한다(사진은 기기 밖으로 나가지 않음).

const MODEL = 'Xenova/slimsam-77-uniform';

type Tensor = { data: ArrayLike<number> & { [i: number]: number }; dims: number[] };
type TF = {
  SamModel: { from_pretrained(id: string, o?: Record<string, unknown>): Promise<SamModelT> };
  AutoProcessor: { from_pretrained(id: string): Promise<ProcessorT> };
  RawImage: { fromCanvas(c: HTMLCanvasElement | OffscreenCanvas): Promise<unknown> };
};
type ProcessorT = ((image: unknown, o?: Record<string, unknown>) => Promise<Record<string, Tensor>>) & {
  post_process_masks(masks: Tensor, orig: Tensor, reshaped: Tensor): Promise<Tensor[]>;
};
type SamModelT = ((inputs: Record<string, unknown>) => Promise<{ pred_masks: Tensor; iou_scores: Tensor }>) & {
  get_image_embeddings(inputs: Record<string, Tensor>): Promise<Record<string, unknown>>;
};

let loading: Promise<{ tf: TF; model: SamModelT; processor: ProcessorT }> | null = null;
function load() {
  loading ??= (async () => {
    const id = '@huggingface/transformers';
    const tf = (await import(/* @vite-ignore */ id)) as unknown as TF;
    const [model, processor] = await Promise.all([tf.SamModel.from_pretrained(MODEL, { dtype: 'q8' }), tf.AutoProcessor.from_pretrained(MODEL)]);
    return { tf, model, processor };
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

export interface SamImage {
  w: number;
  h: number;
  /** 점(그림 픽셀) → 마스크(0/1, w×h)와 점수 */
  segment(x: number, y: number): Promise<{ mask: Uint8Array; score: number }>;
}

/** 그림(캔버스) 하나의 특징을 계산해 둔다 */
export async function samImage(canvas: HTMLCanvasElement): Promise<SamImage> {
  const { tf, model, processor } = await load();
  const image = await tf.RawImage.fromCanvas(canvas);
  const first = await processor(image, { input_points: [[[canvas.width / 2, canvas.height / 2]]] });
  const emb = await model.get_image_embeddings(first);
  const w = canvas.width;
  const h = canvas.height;
  return {
    w,
    h,
    async segment(x: number, y: number) {
      const inputs = await processor(image, { input_points: [[[x, y]]] });
      const out = await model({ ...emb, input_points: inputs.input_points, input_labels: inputs.input_labels });
      const masks = await processor.post_process_masks(out.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
      const sc = Array.from(out.iou_scores.data as ArrayLike<number>);
      // 세 후보 중: 점수가 높고, 너무 크지 않은 것(손 전체가 아니라 손톱 하나)
      const m = masks[0].data;
      const n = w * h;
      let best = -1;
      let bestScore = -1;
      for (let k = 0; k < sc.length; k++) {
        let area = 0;
        for (let i = 0; i < n; i++) if (m[k * n + i]) area++;
        const frac = area / n;
        const s = sc[k] - (frac > 0.08 ? 0.5 : 0);
        if (area > 20 && s > bestScore) {
          bestScore = s;
          best = k;
        }
      }
      const mask = new Uint8Array(n);
      if (best >= 0) for (let i = 0; i < n; i++) mask[i] = m[best * n + i] ? 1 : 0;
      return { mask, score: bestScore };
    },
  };
}
