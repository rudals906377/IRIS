// 임의의 상품 사진(쇼핑몰·업로드) → 착용 엔진용 자산(투명 배경 텍스처, 부위 라벨, 기준점, 메쉬).
//
// 1) 빠른 규칙 방식(배경색·경계 기반 flood fill)으로 옷을 분리하고 기준점을 찾는다.
// 2) 신뢰도가 낮으면 MediaPipe 대화형 분할(magic touch: 사진 가운데를 찍으면 그 물체를 분리)로
//    다시 분리해 보고, 더 신뢰도가 높은 결과를 쓴다.
// 모든 처리는 브라우저 안에서 이뤄지며 사진은 어디에도 전송되지 않는다.

import { FilesetResolver, InteractiveSegmenterLegacy } from '@mediapipe/tasks-vision';
import { buildMeshes, type GarmentAsset, type ProductInfo } from '../garment.ts';
import type { Vec2 } from '../math.ts';
import { largestComponent, open, removeBackground, type RGBAImage } from './background.ts';
import { analyzeTop, type TopAnalysis } from './top.ts';

const WORK_SIZE = 512;
const TEX_SIZE = 1024;
const MAGIC_MODEL =
  'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite';

export interface AnalyzeReport {
  method: 'rule' | 'magic';
  confidence: number;
  sleeve: TopAnalysis['sleeve'];
  warnings: string[];
  ms: number;
}

export interface AnalyzeOptions {
  wasmBase: string;
  name?: string;
  id?: string;
  onStatus?: (msg: string) => void;
}

let magic: InteractiveSegmenterLegacy | null = null;
let magicLoading: Promise<InteractiveSegmenterLegacy> | null = null;

async function getMagic(wasmBase: string): Promise<InteractiveSegmenterLegacy> {
  if (magic) return magic;
  magicLoading ??= (async () => {
    const fs = await FilesetResolver.forVisionTasks(wasmBase);
    magic = await InteractiveSegmenterLegacy.createFromOptions(fs, {
      baseOptions: { modelAssetPath: MAGIC_MODEL, delegate: 'CPU' },
      outputCategoryMask: false,
      outputConfidenceMasks: true,
    });
    return magic;
  })();
  return magicLoading;
}

function toCanvas(src: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  return c;
}

function sizeOf(src: HTMLImageElement | ImageBitmap | HTMLCanvasElement): { w: number; h: number } {
  if (src instanceof HTMLImageElement) return { w: src.naturalWidth, h: src.naturalHeight };
  return { w: src.width, h: src.height };
}

function fit(w: number, h: number, max: number): { w: number; h: number; s: number } {
  const s = Math.min(1, max / Math.max(w, h));
  return { w: Math.round(w * s), h: Math.round(h * s), s };
}

/** 대화형 분할로 옷 마스크를 얻고, 테두리에 붙은 배경색 조각을 지운다. */
function magicMask(seg: InteractiveSegmenterLegacy, work: HTMLCanvasElement, rule: ReturnType<typeof removeBackground>, img: RGBAImage): Uint8Array {
  const { width: w, height: h } = work;
  // 규칙 방식이 찾은 옷의 중심(없으면 사진 가운데)을 찍는다.
  let sx = 0;
  let sy = 0;
  let n = 0;
  for (let i = 0; i < rule.mask.length; i++) {
    if (!rule.mask[i]) continue;
    sx += i % w;
    sy += Math.floor(i / w);
    n++;
  }
  const kp = n > w * h * 0.02 ? { x: sx / n / w, y: sy / n / h } : { x: 0.5, y: 0.55 };
  let mask: Uint8Array = new Uint8Array(w * h);
  seg.segment(work, { keypoint: kp }, (r) => {
    const m = r.confidenceMasks?.[0];
    if (!m) return;
    const f = m.getAsFloat32Array();
    // 결과 마스크 해상도가 입력과 다를 수 있어 가까운 픽셀로 맞춘다.
    for (let y = 0; y < h; y++) {
      const my = Math.min(m.height - 1, Math.floor((y * m.height) / h));
      for (let x = 0; x < w; x++) {
        const mx = Math.min(m.width - 1, Math.floor((x * m.width) / w));
        mask[y * w + x] = f[my * m.width + mx] > 0.5 ? 1 : 0;
      }
    }
  });
  // 배경색과 거의 같은 픽셀은 제외(분할이 배경 조각을 붙여 오는 경우)
  const [br, bg, bb] = rule.bg;
  for (let i = 0; i < w * h; i++) {
    if (!mask[i]) continue;
    const d = Math.hypot(img.data[i * 4] - br, img.data[i * 4 + 1] - bg, img.data[i * 4 + 2] - bb);
    if (d < 4) mask[i] = 0;
  }
  mask = open(mask, w, h);
  return largestComponent(mask, w, h);
}

export async function analyzeProductImage(
  src: HTMLImageElement | ImageBitmap | HTMLCanvasElement,
  opts: AnalyzeOptions,
): Promise<{ asset: GarmentAsset; report: AnalyzeReport }> {
  const t0 = performance.now();
  const { w: ow, h: oh } = sizeOf(src);
  if (!ow || !oh) throw new Error('이미지 크기를 알 수 없습니다.');
  const work = fit(ow, oh, WORK_SIZE);
  const workCanvas = toCanvas(src, work.w, work.h);
  const data = workCanvas.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, work.w, work.h).data;
  const img: RGBAImage = { data, width: work.w, height: work.h };

  opts.onStatus?.('옷 영역 분리 중…');
  const rule = removeBackground(img);
  let best: { mask: Uint8Array; a: TopAnalysis; method: AnalyzeReport['method'] } | null = null;
  const ruleA = analyzeTop(rule.mask, work.w, work.h);
  if (ruleA) best = { mask: rule.mask, a: ruleA, method: 'rule' };

  if (!best || best.a.confidence < 0.7) {
    opts.onStatus?.('정밀 분리 모델로 다시 분석 중…');
    try {
      const seg = await getMagic(opts.wasmBase);
      const mm = magicMask(seg, workCanvas, rule, img);
      const magicA = analyzeTop(mm, work.w, work.h);
      if (magicA && (!best || magicA.confidence > best.a.confidence + 0.05)) {
        best = { mask: mm, a: magicA, method: 'magic' };
      }
    } catch (err) {
      console.warn('대화형 분할 실패', err);
    }
  }
  // 비율이 실제 옷으로 불가능할 만큼 틀리면 망가진 착용 화면 대신 안내한다.
  if (!best || best.a.confidence < 0.35) {
    throw new Error('옷 모양을 인식하지 못했습니다. 배경이 단색인 상의 단독 사진(모델 착용 사진 제외)을 골라 주세요.');
  }

  // 텍스처: 원본을 최대 1024px로, 마스크를 부드럽게 키워 알파로 쓴다.
  opts.onStatus?.('착용 준비 중…');
  const tex = fit(ow, oh, TEX_SIZE);
  const texCanvas = toCanvas(src, tex.w, tex.h);
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = work.w;
  maskCanvas.height = work.h;
  const mctx = maskCanvas.getContext('2d')!;
  const mimg = mctx.createImageData(work.w, work.h);
  // 가장자리 1px을 깎아 배경색이 섞인 테두리(흰 테)를 없앤다.
  const edgeCut = erodeOnce(best.mask, work.w, work.h);
  for (let i = 0; i < edgeCut.length; i++) {
    mimg.data[i * 4 + 3] = edgeCut[i] ? 255 : 0;
  }
  mctx.putImageData(mimg, 0, 0);
  const tctx = texCanvas.getContext('2d')!;
  tctx.globalCompositeOperation = 'destination-in';
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(maskCanvas, 0, 0, tex.w, tex.h);
  tctx.globalCompositeOperation = 'source-over';

  // 라벨은 가장 가까운 분석 픽셀 값으로 키운다.
  const labels = new Uint8Array(tex.w * tex.h);
  for (let y = 0; y < tex.h; y++) {
    const sy = Math.min(work.h - 1, Math.floor((y * work.h) / tex.h));
    for (let x = 0; x < tex.w; x++) {
      const sx = Math.min(work.w - 1, Math.floor((x * work.w) / tex.w));
      labels[y * tex.w + x] = best.a.labels[sy * work.w + sx];
    }
  }
  // 알파가 거의 없는 곳은 라벨도 비운다(메쉬가 빈 칸을 덮지 않도록).
  const alpha = tctx.getImageData(0, 0, tex.w, tex.h).data;
  for (let i = 0; i < labels.length; i++) if (alpha[i * 4 + 3] < 8) labels[i] = 0;

  const k = tex.w / work.w;
  const keypoints: Record<string, Vec2> = {};
  const kpJson: Record<string, [number, number]> = {};
  for (const [name, p] of Object.entries(best.a.keypoints)) {
    keypoints[name] = { x: p.x * k, y: p.y * k };
    kpJson[name] = [p.x * k, p.y * k];
  }
  const info: ProductInfo = {
    id: opts.id ?? `photo-${Date.now()}`,
    name: opts.name ?? '내 상품 사진',
    category: 'top',
    image: '',
    thumb: '',
    parts: '',
    size: [tex.w, tex.h],
    keypoints: kpJson,
    sleeve: best.a.sleeve === 'none' ? null : best.a.sleeve,
  };
  const asset: GarmentAsset = {
    info,
    image: texCanvas,
    labels,
    width: tex.w,
    height: tex.h,
    kp: keypoints,
    meshes: buildMeshes(labels, tex.w, tex.h),
  };
  return {
    asset,
    report: {
      method: best.method,
      confidence: best.a.confidence,
      sleeve: best.a.sleeve,
      warnings: best.a.warnings,
      ms: performance.now() - t0,
    },
  };
}

/** 주소(URL)나 파일에서 이미지를 불러온다. 다른 사이트 이미지는 CORS 허용 시에만 픽셀을 읽을 수 있다. */
export async function loadImageSource(src: string | Blob): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = 'async';
  if (typeof src === 'string') {
    img.crossOrigin = 'anonymous';
    img.src = src;
  } else {
    img.src = URL.createObjectURL(src);
  }
  await img.decode();
  return img;
}

function erodeOnce(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = mask[i] & mask[i - 1] & mask[i + 1] & mask[i - w] & mask[i + w];
    }
  }
  return out;
}
