// 임의의 상품 사진(쇼핑몰·업로드) → 착용 엔진용 자산(투명 배경 텍스처, 부위 라벨, 기준점, 메쉬).
//
// 1) 빠른 규칙 방식(배경색·경계 기반 flood fill)으로 옷을 분리하고 기준점을 찾는다.
// 2) 신뢰도가 낮으면 MediaPipe 대화형 분할(magic touch: 사진 가운데를 찍으면 그 물체를 분리)로
//    다시 분리해 보고, 더 신뢰도가 높은 결과를 쓴다.
// 모든 처리는 브라우저 안에서 이뤄지며 사진은 어디에도 전송되지 않는다.

import { FilesetResolver, ImageSegmenter, InteractiveSegmenterLegacy, PoseLandmarker } from '@mediapipe/tasks-vision';
import { buildMeshes, type GarmentAsset, type ProductInfo } from '../garment.ts';
import type { Vec2 } from '../math.ts';
import { largestComponent, open, removeBackground, type RGBAImage } from './background.ts';
import { analyzeTop, type TopAnalysis } from './top.ts';
import { analyzeWorn, type Landmark } from './worn.ts';

const WORK_SIZE = 512;
const TEX_SIZE = 1024;
const MP_MODELS = 'https://storage.googleapis.com/mediapipe-models';
const POSE_MODEL = `${MP_MODELS}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`;
const SEG_MODEL = `${MP_MODELS}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite`;
const MAGIC_MODEL =
  'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite';

export interface AnalyzeReport {
  method: 'rule' | 'magic' | 'worn';
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

let photoModels: Promise<{ pose: PoseLandmarker; seg: ImageSegmenter }> | null = null;

/** 모델 착용 사진용: 정지 사진 모드의 자세 추정·다중 클래스 분할(CPU, 처음 한 번만 불러옴) */
function getPhotoModels(wasmBase: string): Promise<{ pose: PoseLandmarker; seg: ImageSegmenter }> {
  photoModels ??= (async () => {
    const fs = await FilesetResolver.forVisionTasks(wasmBase);
    const [pose, seg] = await Promise.all([
      PoseLandmarker.createFromOptions(fs, {
        baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        numPoses: 2,
      }),
      ImageSegmenter.createFromOptions(fs, {
        baseOptions: { modelAssetPath: SEG_MODEL, delegate: 'CPU' },
        runningMode: 'IMAGE',
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      }),
    ]);
    return { pose, seg };
  })();
  photoModels.catch(() => (photoModels = null));
  return photoModels;
}

/** 마지막 착용 사진 분석에서 사람(어깨가 보이는 자세)을 찾았는지 */
let personSeen = false;

interface WornResult {
  /** 옷 부분을 잘라 낸 원본 해상도 사진 */
  crop: HTMLCanvasElement;
  work: { w: number; h: number };
  mask: Uint8Array;
  a: TopAnalysis;
}

/** 사진에 사람이 있으면 모델 착용 사진으로 보고 상의를 잘라 분석한다. 사람이 없으면 null. */
async function analyzeWornPhoto(src: HTMLImageElement | ImageBitmap | HTMLCanvasElement, workCanvas: HTMLCanvasElement, wasmBase: string): Promise<WornResult | null> {
  const { pose, seg } = await getPhotoModels(wasmBase);
  const res = pose.detect(workCanvas);
  const lms = res.landmarks?.[0];
  personSeen = !!lms && (lms[11].visibility ?? 0) > 0.5 && (lms[12].visibility ?? 0) > 0.5;
  if (!lms) return null;
  // 두 사람 이상(커플 연출 사진 등)은 어느 옷인지 알 수 없어 쓰지 않는다.
  if ((res.landmarks?.length ?? 0) > 1) return null;
  const W0 = workCanvas.width;
  const H0 = workCanvas.height;
  const P = (i: number): Landmark => ({ x: lms[i].x * W0, y: lms[i].y * H0, visibility: lms[i].visibility });
  const need = [11, 12, 23, 24];
  if (need.some((i) => (lms[i].visibility ?? 1) < 0.5)) return null;
  // 상체와 팔을 넉넉히 감싸는 영역만 잘라 분할 해상도를 높인다.
  const sw = Math.abs(P(11).x - P(12).x);
  const pts = [11, 12, 13, 14, 15, 16, 23, 24].map(P).filter((p) => (p.visibility ?? 1) > 0.3);
  const hipY = (P(23).y + P(24).y) / 2;
  const torso = hipY - (P(11).y + P(12).y) / 2;
  let x0 = Math.min(...pts.map((p) => p.x)) - sw * 0.35;
  let x1 = Math.max(...pts.map((p) => p.x)) + sw * 0.35;
  let y0 = Math.min(P(11).y, P(12).y) - sw * 0.5;
  let y1 = Math.max(hipY + torso * 0.55, ...pts.map((p) => p.y + torso * 0.12));
  x0 = Math.max(0, x0);
  y0 = Math.max(0, y0);
  x1 = Math.min(W0, x1);
  y1 = Math.min(H0, y1);
  if (x1 - x0 < 32 || y1 - y0 < 32) return null;
  const { w: ow } = sizeOf(src);
  const k = ow / W0; // 분석 → 원본 배율
  const cw = Math.round((x1 - x0) * k);
  const ch = Math.round((y1 - y0) * k);
  const tex = fit(cw, ch, TEX_SIZE);
  const crop = document.createElement('canvas');
  crop.width = tex.w;
  crop.height = tex.h;
  const cctx = crop.getContext('2d', { willReadFrequently: true })!;
  cctx.imageSmoothingQuality = 'high';
  cctx.drawImage(src, x0 * k, y0 * k, cw, ch, 0, 0, tex.w, tex.h);
  const work = fit(tex.w, tex.h, WORK_SIZE);
  const wc = toCanvas(crop, work.w, work.h);
  const rgba = wc.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, work.w, work.h).data;
  const clothes = new Float32Array(work.w * work.h);
  let skin = 0;
  seg.segment(wc, (r) => {
    // 사람 피부(2 몸, 3 얼굴)가 보여야 모델 착용 사진으로 인정한다(옷만 찍힌 사진의 자세 오검출 방지).
    for (const c of [2, 3]) {
      const sm = r.confidenceMasks?.[c]?.getAsFloat32Array();
      if (sm) for (let i = 0; i < sm.length; i++) if (sm[i] > 0.5) skin++;
    }
    const m = r.confidenceMasks?.[4]; // 4 = 옷
    if (!m) return;
    skin /= m.width * m.height;
    const f = m.getAsFloat32Array();
    for (let y = 0; y < work.h; y++) {
      const my = Math.min(m.height - 1, Math.floor((y * m.height) / work.h));
      for (let x = 0; x < work.w; x++) {
        const mx = Math.min(m.width - 1, Math.floor((x * m.width) / work.w));
        clothes[y * work.w + x] = f[my * m.width + mx];
      }
    }
  });
  if (skin < 0.01) return null;
  const s = work.w / (x1 - x0);
  const local: Landmark[] = lms.map((l) => ({ x: (l.x * W0 - x0) * s, y: (l.y * H0 - y0) * s, visibility: l.visibility }));
  const a = analyzeWorn(clothes, rgba, work.w, work.h, local);
  if (!a) return null;
  const mask = new Uint8Array(work.w * work.h);
  for (let i = 0; i < mask.length; i++) mask[i] = a.labels[i] ? 1 : 0;
  return { crop, work, mask, a };
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
  const ruleA = analyzeTop(rule.mask, work.w, work.h, data);
  if (ruleA) best = { mask: rule.mask, a: ruleA, method: 'rule' };

  let texSrc: CanvasImageSource = src;
  let texSize = { w: ow, h: oh };
  let workSize = work;
  {
    // 사람이 찍혀 있으면 모델 착용 사진으로 분석한다(단색 배경의 모델 사진은 규칙 방식이 사람 전체를 옷으로 볼 수 있다).
    opts.onStatus?.('모델 착용 사진인지 확인 중…');
    personSeen = false;
    try {
      const worn = await analyzeWornPhoto(src, workCanvas, opts.wasmBase);
      if (worn && worn.a.confidence >= 0.4) {
        best = { mask: worn.mask, a: worn.a, method: 'worn' };
        texSrc = worn.crop;
        texSize = { w: worn.crop.width, h: worn.crop.height };
        workSize = { ...worn.work, s: 1 };
      }
    } catch (err) {
      console.warn('착용 사진 분석 실패', err);
    }
  }
  // 사람이 찍혔는데 착용 사진 분석이 안 되면(뒷모습·옆모습·여러 명·확대) 옷만 분리하는 모델도 사람째 잘라 오므로 쓰지 않는다.
  if (personSeen && best?.method !== 'worn' && (!best || best.a.confidence < 0.7)) {
    throw new Error('정면 상반신이 보이는 사진이 아닙니다(뒷모습·옆모습·확대·여러 명). 다른 사진을 골라 주세요.');
  }
  if (!best || best.a.confidence < 0.7) {
    opts.onStatus?.('정밀 분리 모델로 다시 분석 중…');
    try {
      const seg = await getMagic(opts.wasmBase);
      const mm = magicMask(seg, workCanvas, rule, img);
      const magicA = analyzeTop(mm, work.w, work.h, data);
      if (magicA && (!best || magicA.confidence > best.a.confidence + 0.05)) {
        best = { mask: mm, a: magicA, method: 'magic' };
        texSrc = src;
        texSize = { w: ow, h: oh };
        workSize = work;
      }
    } catch (err) {
      console.warn('대화형 분할 실패', err);
    }
  }
  // 비율이 실제 옷으로 불가능할 만큼 틀리면 망가진 착용 화면 대신 안내한다.
  if (!best || best.a.confidence < 0.35) {
    throw new Error('옷 모양을 인식하지 못했습니다. 배경이 단색인 상의 단독 사진이나 정면 모델 착용 사진을 골라 주세요.');
  }

  // 텍스처: 원본을 최대 1024px로, 마스크를 부드럽게 키워 알파로 쓴다.
  opts.onStatus?.('착용 준비 중…');
  const tex = fit(texSize.w, texSize.h, TEX_SIZE);
  const texCanvas = toCanvas(texSrc, tex.w, tex.h);
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = workSize.w;
  maskCanvas.height = workSize.h;
  const mctx = maskCanvas.getContext('2d')!;
  const mimg = mctx.createImageData(workSize.w, workSize.h);
  // 가장자리 1px을 깎아 배경색이 섞인 테두리(흰 테)를 없앤다.
  const edgeCut = erodeOnce(best.mask, workSize.w, workSize.h);
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
    const sy = Math.min(workSize.h - 1, Math.floor((y * workSize.h) / tex.h));
    for (let x = 0; x < tex.w; x++) {
      const sx = Math.min(workSize.w - 1, Math.floor((x * workSize.w) / tex.w));
      labels[y * tex.w + x] = best.a.labels[sy * workSize.w + sx];
    }
  }
  // 알파가 거의 없는 곳은 라벨도 비운다(메쉬가 빈 칸을 덮지 않도록).
  const alpha = tctx.getImageData(0, 0, tex.w, tex.h).data;
  for (let i = 0; i < labels.length; i++) if (alpha[i * 4 + 3] < 8) labels[i] = 0;

  const k = tex.w / workSize.w;
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
    widthScale: best.a.widthScale,
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

/**
 * 빠른 사전 평가(규칙 방식만, 모델 불필요): 쇼핑몰 페이지의 여러 사진 중
 * "배경이 단색이고 옷 형태가 분명한 사진"을 고르는 데 쓴다. 0~1(높을수록 옷만 찍힌 사진).
 */
export function quickAssess(src: HTMLImageElement | ImageBitmap | HTMLCanvasElement): { confidence: number; sleeve: TopAnalysis['sleeve'] | null } {
  const { w: ow, h: oh } = sizeOf(src);
  if (!ow || !oh) return { confidence: 0, sleeve: null };
  const work = fit(ow, oh, WORK_SIZE);
  const c = toCanvas(src, work.w, work.h);
  const data = c.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, work.w, work.h).data;
  const rule = removeBackground({ data, width: work.w, height: work.h });
  // 배경이 단색이 아니면(모델 착용·연출 사진) 규칙 방식 결과를 믿지 않는다.
  const bgPenalty = rule.spread > 40 ? 0.3 : rule.spread > 20 ? 0.7 : 1;
  const a = analyzeTop(rule.mask, work.w, work.h, data);
  return { confidence: (a?.confidence ?? 0) * bgPenalty, sleeve: a?.sleeve ?? null };
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
