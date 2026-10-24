// 뷰티 스타일 AI 분류기(별도 저장소 rudals906377/AI) 불러오기.
// 분류기 코드와 사전 계산 파일은 그 저장소의 특정 커밋에서 CDN(jsDelivr)으로 받고,
// 비전 모델(약 94MB)은 Transformers.js가 Hugging Face에서 받아 브라우저 캐시에 둔다.
// 사진은 기기 밖으로 나가지 않는다(모델 파일만 내려받는다).
// index.html의 importmap이 '@huggingface/transformers'를 CDN 주소로 연결해야 한다.

import type { StyleAIResult } from './style-attributes.ts';

/** 분류기 저장소의 커밋(PR #1 시점). 저장소가 합쳐지면 main의 커밋으로 바꾼다 */
const AI_REPO = 'rudals906377/AI';
const AI_COMMIT = '1e98992';
const AI_BASE = `https://cdn.jsdelivr.net/gh/${AI_REPO}@${AI_COMMIT}`;
const MODEL = 'Marqo/marqo-fashionSigLIP';

interface Analyzer {
  analyze(source: ImageBitmap | HTMLCanvasElement | HTMLImageElement | Blob, opts?: { category?: string; topk?: number }): Promise<StyleAIResult>;
  dispose(): void;
}

let loading: Promise<Analyzer> | null = null;

export interface AIProgress {
  status?: string;
  file?: string;
  progress?: number;
}

/** 분류기를 (한 번만) 불러온다. onProgress로 내려받기 진행률을 알린다. */
export function loadStyleAI(onProgress?: (p: AIProgress) => void): Promise<Analyzer> {
  loading ??= (async () => {
    const [mod, labelEmbeddings, heads] = await Promise.all([
      import(/* @vite-ignore */ `${AI_BASE}/analyzer.js`) as Promise<{ createAnalyzer: (o: Record<string, unknown>) => Promise<Analyzer> }>,
      fetch(`${AI_BASE}/embeddings/marqo-fashionSigLIP.json`).then((r) => (r.ok ? r.json() : undefined)),
      fetch(`${AI_BASE}/heads/marqo-fashionSigLIP.json`).then((r) => (r.ok ? r.json() : undefined)),
    ]);
    return mod.createAnalyzer({ model: MODEL, labelEmbeddings, heads, onProgress });
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

/** 사진 한 장을 분류한다. category를 주면 그 분야로 고정. */
export async function analyzeStyle(source: ImageBitmap | Blob, category: 'auto' | 'hair' | 'nail' | 'makeup' | 'tattoo' = 'auto', onProgress?: (p: AIProgress) => void): Promise<StyleAIResult> {
  const a = await loadStyleAI(onProgress);
  // RawImage.read는 Blob·URL·캔버스를 받는다. ImageBitmap은 캔버스로 바꿔 넘긴다
  let src: HTMLCanvasElement | Blob = source as Blob;
  if (source instanceof ImageBitmap) {
    const c = document.createElement('canvas');
    c.width = source.width;
    c.height = source.height;
    c.getContext('2d')!.drawImage(source, 0, 0);
    src = c;
  }
  return a.analyze(src, { category });
}
