// 참고 사진 → 앱 설정: 사용자가 고른 사진을 브라우저 안에서 분석해 색·스타일을 뽑는다(사진은 밖으로 나가지 않는다).
//  - 메이크업: 얼굴 점으로 입술·볼·눈꺼풀·눈썹·눈매 색을 재서 셰이더 색 계산의 역산(style-math.ts)으로 목표색을 구한다
//  - 헤어: 분할(머리카락)로 평균색·뿌리/끝 색을 잰다
//  - 네일: 손 점으로 손톱 색을, 손이 안 잡히면 피부·배경을 뺀 주요 색을 쓴다
//  - 타투: 피부 위의 잉크(어둡거나 피부와 색이 다른 곳)만 오려 도안으로 만든다
// 모델은 IMAGE 모드로 따로 만들어(실시간 추적기와 분리) 처음 쓸 때만 불러온다.

import { FaceLandmarker, FilesetResolver, HandLandmarker, ImageSegmenter, type NormalizedLandmark } from '@mediapipe/tasks-vision';
import type { Vec2 } from '../engine/math.ts';
import { chroma, measureFace, type FaceMeasure } from './face-measure.ts';
import type { RGB } from './makeup.ts';
import type { NailStyle } from './nail.ts';
import type { TattooDesign } from './tattoo-designs.ts';
import type { StyleAIResult, StyleHints } from './style-attributes.ts';
import { dominantColors, gam, glossFrom, hairTarget, hex, isSkin, lin, lipTarget, luma, tintTarget } from './style-math.ts';

const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';
const FACE_MODEL = `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`;
const SEG_MODEL = `${MODEL_BASE}/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite`;
const HAND_MODEL = `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`;
const CLS = { background: 0, hair: 1, bodySkin: 2, faceSkin: 3 } as const;
/** 분석용 그림 크기 상한(긴 변) */
const MAX_SIDE = 768;

export type PhotoMode = 'auto' | 'makeup' | 'hair' | 'nail' | 'tattoo';

export interface MakeupStyle {
  lip?: { color: RGB; amount: number; gloss: number; style: 'full' | 'gradient' | 'blur' };
  /** pos: 0 눈 밑 사과존 ~ 1 광대 */
  /** redness: 사진에서 잰 붉은 기(0.08 이상이면 블러셔로 봄). 스타일 AI의 '블러셔 없음' 힌트가 측정을 덮어쓸지 정할 때 쓴다 */
  blush?: { color: RGB; amount: number; pos: number; size: number; redness?: number };
  shadow?: { color: RGB; amount: number };
  brow?: { color: RGB; amount: number };
  liner?: { color: RGB; amount: number };
  /** 쉐딩·하이라이터 */
  contour?: { color: RGB; amount: number };
  /** 참고: 측정값 요약(디버그) */
  debug: Record<string, string | number>;
}

export interface StyleResult {
  makeup?: MakeupStyle;
  /** 스타일 AI(분류기) 결과 힌트와 한 줄 설명(켜져 있을 때) */
  ai?: { hints: StyleHints; headline?: string; description?: string; raw?: StyleAIResult };
  /** 사진 얼굴 측정값(되먹임 비교용) */
  measure?: FaceMeasure;
  hair?: { color: RGB; tip: RGB | null; amount: number };
  nail?: { color: RGB; style: NailStyle };
  tattoo?: TattooDesign;
  /** 사용자에게 보여 줄 한 줄 설명 */
  summary: string;
  /** 분석에 쓴 축소 그림 */
  thumb: HTMLCanvasElement;
}

// ---- 화장 안 한 얼굴의 자연 비율(맨얼굴 시험 사진 6장으로 잰 값, 밝기 1로 맞춘 색조). 사진 비율이 이보다 얼마나 다른지로 화장 정도를 정한다 ----
/** 눈 밑 사과존·볼 가운데·광대 / 그 자리의 예상 피부색 */
// (화장 없는 사진 여러 장에서 사과존·볼은 예상 피부색과 거의 같은 색조였다. 스튜디오 조명의 푸른 반사광은 제외)
export const NATURAL_APPLE: RGB = [0.97, 1.0, 1.02];
export const NATURAL_MID: RGB = [0.98, 1.0, 1.03];
export const NATURAL_BONE: RGB = [0.96, 1.0, 1.04];
/** 윗눈꺼풀(더 어두운 쪽) 색조와 밝기 */
export const NATURAL_SHADOW: RGB = [1.32, 0.95, 0.76];
export const NATURAL_SHADOW_L = 0.38;
/** 속눈썹 선 바로 위 띠 밝기 / 윗눈꺼풀 밝기 */
export const NATURAL_LINER_L = 0.39;
/** 눈썹 밝기 / 예상 피부 밝기 */
export const NATURAL_BROW_L = 0.44;
/** 광대 아래·콧대 밝기 / 예상 피부 밝기 */
export const NATURAL_HOLLOW_L = 0.95;
export const NATURAL_BRIDGE_L = 1.1;
/** 진하기를 정해 두고 그때 가운데에서 실제로 섞이는 비율(합성 결과를 재서 맞춘 값) */
export const BLUSH_AMOUNT = 0.9;
export const BLUSH_KEFF = 0.28;
export const SHADOW_AMOUNT = 0.6;
export const SHADOW_KEFF = 0.35;

type Pixels = { w: number; h: number; d: Uint8ClampedArray; img: ImageData };
type SegMasks = { hair: Float32Array; body: Float32Array; face: Float32Array; bg: Float32Array; w: number; h: number };

export class PhotoAnalyzer {
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  private face: FaceLandmarker | null = null;
  private seg: ImageSegmenter | null = null;
  private hands: HandLandmarker | null = null;
  private readonly wasmBase: string;

  constructor(wasmBase: string) {
    this.wasmBase = wasmBase;
  }

  private async fs(): Promise<NonNullable<typeof this.fileset>> {
    if (!this.fileset) this.fileset = await FilesetResolver.forVisionTasks(this.wasmBase);
    return this.fileset;
  }

  /** GPU 위임을 먼저 시도하고 실패하면 CPU */
  private async make<T>(f: (d: 'GPU' | 'CPU') => Promise<T>): Promise<T> {
    try {
      return await f('GPU');
    } catch {
      return f('CPU');
    }
  }

  private async faceModel(): Promise<FaceLandmarker> {
    if (!this.face) {
      const fs = await this.fs();
      this.face = await this.make((delegate) =>
        FaceLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: FACE_MODEL, delegate }, runningMode: 'IMAGE', numFaces: 1 }),
      );
    }
    return this.face;
  }

  private async segModel(): Promise<ImageSegmenter> {
    if (!this.seg) {
      const fs = await this.fs();
      this.seg = await this.make((delegate) =>
        ImageSegmenter.createFromOptions(fs, {
          baseOptions: { modelAssetPath: SEG_MODEL, delegate },
          runningMode: 'IMAGE',
          outputCategoryMask: false,
          outputConfidenceMasks: true,
        }),
      );
    }
    return this.seg;
  }

  private async handModel(): Promise<HandLandmarker> {
    if (!this.hands) {
      const fs = await this.fs();
      this.hands = await this.make((delegate) =>
        HandLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: HAND_MODEL, delegate }, runningMode: 'IMAGE', numHands: 2 }),
      );
    }
    return this.hands;
  }

  /**
   * 사진을 분석한다. mode가 auto면 얼굴·손·머리카락이 보이는 정도로 무엇을 가져올지 정한다.
   */
  async analyze(source: ImageBitmap | HTMLImageElement, mode: PhotoMode, onStatus?: (s: string) => void): Promise<StyleResult> {
    const canvas = document.createElement('canvas');
    const scale = Math.min(1, MAX_SIDE / Math.max(source.width, source.height));
    canvas.width = Math.max(2, Math.round(source.width * scale));
    canvas.height = Math.max(2, Math.round(source.height * scale));
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const px: Pixels = { w: canvas.width, h: canvas.height, d: imgData.data, img: imgData };
    const out: StyleResult = { summary: '', thumb: canvas };
    const parts: string[] = [];

    const wantFace = mode === 'auto' || mode === 'makeup';
    const wantHair = mode === 'auto' || mode === 'hair';
    const wantNail = mode === 'auto' || mode === 'nail';
    const wantTattoo = mode === 'auto' || mode === 'tattoo';

    let facePts: Vec2[] | null = null;
    if (wantFace || wantHair) {
      onStatus?.('얼굴 찾는 중…');
      const fm = await this.faceModel();
      const r = fm.detect(canvas);
      if (r.faceLandmarks.length > 0) facePts = toPx(r.faceLandmarks[0], px.w, px.h);
    }
    let segMasks: SegMasks | null = null;
    if (wantHair || wantTattoo || wantNail || wantFace) {
      onStatus?.('머리카락·피부 나누는 중…');
      const sm = await this.segModel();
      sm.segment(canvas, (res) => {
        const m = res.confidenceMasks;
        if (!m || m.length < 4) return;
        segMasks = {
          hair: m[CLS.hair].getAsFloat32Array().slice(),
          body: m[CLS.bodySkin].getAsFloat32Array().slice(),
          face: m[CLS.faceSkin].getAsFloat32Array().slice(),
          bg: m[CLS.background].getAsFloat32Array().slice(),
          w: m[0].width,
          h: m[0].height,
        };
      });
    }

    // 얼굴이 사진 폭의 12% 이상이면 메이크업을 잰다
    const faceW = facePts ? Math.hypot(facePts[234].x - facePts[454].x, facePts[234].y - facePts[454].y) : 0;
    // 머리색은 먼저 잰다: 얼굴 위에 드리운 머리카락(앞머리·볼 옆)을 피부 표본에서 빼는 데 쓴다
    const hairM = segMasks ? measureHair(px, segMasks) : null;
    if (wantFace && facePts && (mode === 'makeup' || faceW > px.w * 0.12)) {
      const sm = segMasks as SegMasks | null;
      const hairG = hairM ? hairM.color : null; // 감마 공간
      const skinOk = sm
        ? (q: Vec2): boolean => {
            const si = Math.floor((q.y * sm.h) / px.h) * sm.w + Math.floor((q.x * sm.w) / px.w);
            // 분할이 머리카락이라 하는 곳은 피부가 아니다
            if (sm.hair[si] >= 0.35 || !(sm.face[si] > 0.5 || sm.body[si] > 0.5)) return false;
            // 분할이 놓친 가는 앞머리: 픽셀 색이 이 사진의 머리색과 비슷하면 뺀다(머리색이 블러셔·아이섀도로 읽히는 것 방지)
            if (hairG) {
              const k = (Math.round(q.y) * px.w + Math.round(q.x)) * 4;
              const r = px.d[k] / 255;
              const g = px.d[k + 1] / 255;
              const b = px.d[k + 2] / 255;
              // 색감(밝기로 나눈 비율)과 밝기가 모두 머리색과 가까우면 머리카락으로 본다
              const lp = (r + g + b) / 3;
              const lh = (hairG[0] + hairG[1] + hairG[2]) / 3;
              const d = Math.hypot(r / lp - hairG[0] / lh, g / lp - hairG[1] / lh, b / lp - hairG[2] / lh);
              if (d < 0.25 && Math.abs(lp - lh) < 0.25) return false;
            }
            return true;
          }
        : undefined;
      out.measure = measureFace(px.img, facePts, skinOk);
      out.makeup = makeupFromMeasure(out.measure);
      const got = ['lip', 'blush', 'shadow', 'liner', 'brow'].filter((k) => (out.makeup as unknown as Record<string, unknown>)[k]);
      if (got.length) parts.push(`메이크업(${got.map((k) => ({ lip: '립', blush: '블러셔', shadow: '아이섀도', liner: '아이라인', brow: '눈썹' })[k]).join('·')})`);
    }
    if (wantHair && segMasks) {
      const h = hairM;
      if (h && (mode === 'hair' || h.frac > 0.03)) {
        out.hair = { color: h.color, tip: h.tip, amount: 0.85 };
        parts.push(h.tip ? '헤어(옴브레)' : '헤어');
      }
    }
    if (wantNail && !out.makeup && (mode === 'nail' || mode === 'auto')) {
      onStatus?.('손 찾는 중…');
      const hm = await this.handModel();
      const r = hm.detect(canvas);
      const n = measureNail(px, r.landmarks.map((lm) => toPx(lm, px.w, px.h)), mode === 'nail', segMasks);
      if (n) {
        out.nail = n;
        parts.push('네일');
      }
    }
    if (wantTattoo && !out.makeup && !out.nail && segMasks && (mode === 'tattoo' || !out.hair)) {
      const t = extractTattoo(px, segMasks);
      if (t) {
        out.tattoo = t;
        parts.push('타투 도안');
      }
    }
    out.summary = parts.length ? `사진에서 ${parts.join(', ')}을(를) 가져왔어요` : '사진에서 가져올 수 있는 스타일을 찾지 못했어요';
    return out;
  }
}

function toPx(lm: NormalizedLandmark[], w: number, h: number): Vec2[] {
  return lm.map((l) => ({ x: l.x * w, y: l.y * h }));
}

// ---- 픽셀 표본 도구 ----

const lin1 = (v: number): number => Math.pow(v / 255, 2.2);

/** 다각형(구멍 제외) 안 픽셀을 선형 RGB로 모은다 */
function samplePolys(px: Pixels, polys: Vec2[][], holes: Vec2[][] = []): RGB[] {
  const m = document.createElement('canvas');
  m.width = px.w;
  m.height = px.h;
  const mc = m.getContext('2d', { willReadFrequently: true })!;
  const fill = (p: Vec2[], color: string): void => {
    mc.fillStyle = color;
    mc.beginPath();
    p.forEach((q, k) => (k ? mc.lineTo(q.x, q.y) : mc.moveTo(q.x, q.y)));
    mc.closePath();
    mc.fill();
  };
  for (const p of polys) fill(p, '#fff');
  for (const p of holes) fill(p, '#000');
  const md = mc.getImageData(0, 0, px.w, px.h).data;
  const out: RGB[] = [];
  for (let k = 0; k < px.w * px.h; k++) if (md[k * 4] > 128) out.push([lin1(px.d[k * 4]), lin1(px.d[k * 4 + 1]), lin1(px.d[k * 4 + 2])]);
  return out;
}

const circle = (c: Vec2, r: number): Vec2[] => Array.from({ length: 16 }, (_, k) => ({ x: c.x + Math.cos((k / 16) * Math.PI * 2) * r, y: c.y + Math.sin((k / 16) * Math.PI * 2) * r }));

/** 밝기 위아래 10%를 뺀 평균(반사광·그늘 제외) */
function trimmedMean(px: RGB[]): RGB | null {
  if (px.length < 12) return null;
  const s = [...px].sort((a, b) => luma(a) - luma(b));
  const a = Math.floor(s.length * 0.1);
  const sel = s.slice(a, s.length - a);
  const out: RGB = [0, 0, 0];
  for (const q of sel) for (let c = 0; c < 3; c++) out[c] += q[c];
  return out.map((v) => v / sel.length) as RGB;
}

function percentileL(px: RGB[], p: number): number {
  const s = px.map(luma).sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(s.length * p))] ?? 0;
}


// ---- 메이크업: 측정값 → 설정 ----

const lerp = (a: Vec2, b: Vec2, t: number): Vec2 => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

export function makeupFromMeasure(m: FaceMeasure): MakeupStyle {
  const debug: MakeupStyle['debug'] = { skin: hex(m.skin) };
  const res: MakeupStyle = { debug };
  const fmt = (r: RGB | null): string => (r ? r.map((v) => v.toFixed(2)).join(' ') : '-');

  // 립
  if (m.lip) {
    const color = lipTarget(m.lip, m.skin);
    const gloss = glossFrom(m.lipP50, m.lipP90);
    let style: 'full' | 'gradient' | 'blur' = 'full';
    if (m.lipEdgeSat !== null) {
      if (m.lipEdgeSat < 0.7) style = 'blur';
      else if (m.lipEdgeSat < 0.85) style = 'gradient';
    }
    res.lip = { color, amount: m.lipDiff < 0.08 ? 0.45 : 0.85, gloss, style };
    debug.lip = hex(m.lip);
    debug.lipDiff = +m.lipDiff.toFixed(2);
    debug.lipEdgeSat = m.lipEdgeSat === null ? '-' : +m.lipEdgeSat.toFixed(2);
  }

  // 블러셔: 세 위치 중 붉은 기(초록 채널이 줄어든 정도)가 가장 강한 곳
  const spots: { r: RGB | null; nat: RGB; pos: number }[] = [
    { r: m.blushApple, nat: NATURAL_APPLE, pos: 0.15 },
    { r: m.blushMid, nat: NATURAL_MID, pos: 0.5 },
    // 광대 위치는 머리카락·귀 그늘 때문에 자동 판정에서 뺀다(스타일 AI가 셰이딩이라 하면 윤곽으로 처리)
  ];
  void NATURAL_BONE;
  void m.blushBone;
  debug.blush = `${fmt(m.blushApple)} | ${fmt(m.blushMid)} | ${fmt(m.blushBone)}`;
  let best: { t: { color: RGB; amount: number }; pos: number; dev: number } | null = null;
  const redOf = (sp: (typeof spots)[number]): number => {
    if (!sp.r) return 0;
    const c = chroma(sp.r);
    // 붉은 기: 자연 색조보다 빨강이 초록보다 더 커야 한다(밝기만 다른 그늘·하이라이트는 제외)
    return c[0] / sp.nat[0] - c[1] / sp.nat[1];
  };
  // 광대(bone)는 머리카락·귀 그늘이 섞이기 쉬워, 사과존·볼에도 붉은 기가 있고 광대가 그보다 뚜렷할 때만 광대 위치로 본다
  const cheekRed = Math.max(redOf(spots[0]), redOf(spots[1]));
  for (const sp of spots) {
    if (!sp.r) continue;
    const redness = redOf(sp);
    const t = tintTarget(chroma(sp.r), sp.nat, BLUSH_AMOUNT, BLUSH_KEFF, 0.05);
    if (!t || redness < 0.08 || cheekRed < 0.08) continue;
    if (!best || redness > best.dev) best = { t, pos: sp.pos, dev: redness };
  }
  if (best) {
    // 퍼짐: 붉은 기가 두세 곳에 함께 있으면 넓게 바른 것
    const reds = spots.filter((sp) => sp.r && chroma(sp.r)[0] / sp.nat[0] - chroma(sp.r)[1] / sp.nat[1] > 0.03).length;
    res.blush = { ...best.t, pos: best.pos, size: reds >= 3 ? 1.5 : reds === 2 ? 1.25 : 1, redness: best.dev };
  }

  // 아이섀도: 안쪽·바깥쪽 중 더 진한 쪽 기준(그라데이션은 합성 쪽에서 만든다)
  debug.shadow = `${fmt(m.shadowIn)} | ${fmt(m.shadowOut)} | 밑 ${fmt(m.underEye)}`;
  const sh = [m.shadowIn, m.shadowOut].filter((x): x is RGB => !!x).sort((a, b) => luma(a) - luma(b))[0];
  if (sh) {
    // 자연 눈꺼풀보다 어둡거나(음영) 색조가 다르면(컬러 섀도) 섀도로 본다
    const dark = Math.max(0, 1 - luma(sh) / NATURAL_SHADOW_L);
    const t = tintTarget(chroma(sh), NATURAL_SHADOW, SHADOW_AMOUNT, SHADOW_KEFF, 0.1);
    if (dark > 0.15 || t) {
      const base = t?.color ?? gam([0.637 * 0.6, 0.366 * 0.55, 0.254 * 0.5] as RGB);
      // 어두울수록 색을 더 진하게
      const col = gam(lin(base).map((v) => v * (1 - dark * 0.5)) as RGB);
      res.shadow = { color: col, amount: Math.min(1, SHADOW_AMOUNT + dark * 0.6) };
    }
  }

  // 아이라인: 속눈썹 선 위 띠가 눈꺼풀보다 훨씬 어두우면
  if (m.linerL !== null) {
    debug.linerL = +m.linerL.toFixed(2);
    const r = m.linerL / NATURAL_LINER_L;
    if (r < 0.7) {
      const dark = r < 0.4;
      res.liner = { color: dark ? [0.1, 0.08, 0.08] : [0.23, 0.16, 0.13], amount: dark ? 0.85 : 0.5 };
    }
  }

  // 눈썹: 눈썹 위 뼈 피부 대비 밝기
  if (m.brow) {
    const br = luma(m.brow);
    debug.browL = +br.toFixed(2);
    if (br < NATURAL_BROW_L * 0.9) {
      const tint = m.brow.map((v) => Math.min(1, v / Math.max(br, 1e-3))) as RGB;
      const REFc: RGB = [0.637, 0.366, 0.254];
      const col = gam(REFc.map((v, i) => v * tint[i] * Math.max(0.15, br)) as RGB);
      res.brow = { color: col, amount: br < NATURAL_BROW_L * 0.6 ? 0.45 : 0.3 };
    }
  }

  // 윤곽: 광대 아래가 볼보다 뚜렷이 어둡거나 콧대가 코 옆보다 뚜렷이 밝으면
  if (m.hollowL !== null || m.noseBridgeL !== null) {
    debug.contour = `${m.hollowL?.toFixed(2) ?? '-'} / ${m.noseBridgeL?.toFixed(2) ?? '-'} (피부표본 ${m.skinSamples})`;
    const shade = m.hollowL !== null ? Math.max(0, 1 - m.hollowL / NATURAL_HOLLOW_L) : 0;
    const light = m.noseBridgeL !== null ? Math.max(0, m.noseBridgeL / NATURAL_BRIDGE_L - 1) : 0;
    const k = Math.max(shade * 4, light * 3);
    if (k > 0.3) res.contour = { color: [0.55, 0.47, 0.39], amount: Math.min(0.7, 0.3 + k * 0.4) };
  }
  return res;
}

// ---- 헤어 ----

function measureHair(px: Pixels, seg: { hair: Float32Array; w: number; h: number }): { color: RGB; tip: RGB | null; frac: number } | null {
  const sx = seg.w / px.w;
  const sy = seg.h / px.h;
  const pts: { c: RGB; y: number }[] = [];
  const step = Math.max(1, Math.floor(px.w / 200));
  for (let y = 0; y < px.h; y += step)
    for (let x = 0; x < px.w; x += step) {
      const si = Math.floor(y * sy) * seg.w + Math.floor(x * sx);
      if (seg.hair[si] < 0.8) continue;
      const k = (y * px.w + x) * 4;
      pts.push({ c: [lin1(px.d[k]), lin1(px.d[k + 1]), lin1(px.d[k + 2])], y });
    }
  const frac = pts.length / ((px.w / step) * (px.h / step));
  if (pts.length < 60) return null;
  const mean = (arr: { c: RGB }[]): RGB => {
    const s: RGB = [0, 0, 0];
    for (const q of arr) for (let i = 0; i < 3; i++) s[i] += q.c[i];
    return s.map((v) => v / arr.length) as RGB;
  };
  const ys = pts.map((q) => q.y).sort((a, b) => a - b);
  const y1 = ys[Math.floor(ys.length / 3)];
  const y2 = ys[Math.floor((ys.length * 2) / 3)];
  const t = hairTarget(mean(pts), mean(pts.filter((q) => q.y <= y1)), mean(pts.filter((q) => q.y >= y2)));
  return { ...t, frac };
}

// ---- 네일 ----

function measureNail(
  px: Pixels,
  hands: Vec2[][],
  forced: boolean,
  seg: { body: Float32Array; w: number; h: number } | null,
): { color: RGB; style: NailStyle } | null {
  let samples: RGB[] = [];
  const perNail: RGB[] = [];
  for (const h of hands) {
    for (const [t, d] of [[8, 7], [12, 11], [16, 15], [20, 19], [4, 3]]) {
      const seg = Math.hypot(h[t].x - h[d].x, h[t].y - h[d].y);
      const c = lerp(h[d], h[t], 0.62);
      const s = samplePolys(px, [circle(c, seg * 0.22)]);
      const m = trimmedMean(s);
      if (m) {
        perNail.push(m);
        samples.push(...s);
      }
    }
  }
  let color: RGB | null = null;
  if (perNail.length >= 2) {
    // 손톱 색 중 중간 밝기의 것(손톱이 아닌 곳을 찍은 표본을 피함)
    perNail.sort((a, b) => luma(a) - luma(b));
    color = perNail[Math.floor(perNail.length / 2)];
  } else if (forced || hands.length === 0) {
    // 손 점이 없으면: 피부·흰 배경을 뺀 주요 색(가까이 찍은 손톱 사진)
    const all: RGB[] = [];
    const step = Math.max(1, Math.floor(px.w / 120));
    for (let y = 0; y < px.h; y += step)
      for (let x = 0; x < px.w; x += step) {
        const k = (y * px.w + x) * 4;
        const c: RGB = [lin1(px.d[k]), lin1(px.d[k + 1]), lin1(px.d[k + 2])];
        if (isSkin(c)) continue;
        const g = gam(c);
        const mx = Math.max(...g);
        const mn = Math.min(...g);
        if (mx > 0.85 && mx - mn < 0.08) continue; // 흰 배경
        if (seg) {
          const si = Math.floor((y * seg.h) / px.h) * seg.w + Math.floor((x * seg.w) / px.w);
          if (seg.body[si] > 0.7) continue; // 분할이 피부라고 하는 곳
        }
        all.push(c);
      }
    if (all.length < 50) return null;
    const dom = dominantColors(all, 4);
    // 너무 어두운(그늘·검은 배경) 무리는 뒤로
    const pick = dom.find((d) => luma(d.color) > 0.03) ?? dom[0];
    if (!pick || pick.share < 0.15) return null;
    color = pick.color;
    samples = all;
  }
  if (!color) return null;
  // 마감: 반짝이 점이 많으면 글리터, 채도 낮고 명암 차가 크면 크롬
  let style: NailStyle = 'solid';
  if (samples.length > 30) {
    const med = percentileL(samples, 0.5);
    const p95 = percentileL(samples, 0.95);
    const specks = samples.filter((c) => luma(c) > med * 2.2).length / samples.length;
    const g = gam(color);
    const sat = (Math.max(...g) - Math.min(...g)) / Math.max(Math.max(...g), 1e-3);
    if (specks > 0.06) style = 'glitter';
    else if (sat < 0.15 && p95 / Math.max(med, 1e-3) > 2.5) style = 'chrome';
  }
  return { color: gam(color), style };
}

// ---- 타투 ----

function extractTattoo(px: Pixels, seg: { body: Float32Array; face: Float32Array; bg: Float32Array; w: number; h: number }): TattooDesign | null {
  const { w, h, d } = px;
  const n = w * h;
  const skinP = new Float32Array(n);
  const person = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const sy = Math.floor((y * seg.h) / h);
    for (let x = 0; x < w; x++) {
      const si = sy * seg.w + Math.floor((x * seg.w) / w);
      const k = y * w + x;
      skinP[k] = Math.max(seg.body[si], seg.face[si]);
      person[k] = 1 - seg.bg[si];
    }
  }
  // 피부 기준색: 분할이 피부라 하고 색도 피부인 곳
  const skinPx: RGB[] = [];
  for (let k = 0; k < n; k += 7) {
    if (skinP[k] < 0.7) continue;
    const c: RGB = [lin1(d[k * 4]), lin1(d[k * 4 + 1]), lin1(d[k * 4 + 2])];
    if (isSkin(c)) skinPx.push(c);
  }
  if (skinPx.length < 40) return null;
  const skin = trimmedMean(skinPx)!;
  const Ls = luma(skin);
  const sSum = skin[0] + skin[1] + skin[2];
  const sr = skin[0] / sSum;
  const sg = skin[1] / sSum;
  // 피부 영역을 넓혀(잉크 부분은 분할이 피부로 안 볼 수 있음) 그 안에서 잉크를 찾는다
  const region = boxBlur(skinP, w, h, Math.max(2, Math.round(Math.min(w, h) * 0.04)));
  const alpha = new Float32Array(n);
  let count = 0;
  let x0 = w;
  let y0 = h;
  let x1 = 0;
  let y1 = 0;
  for (let k = 0; k < n; k++) {
    if (region[k] < 0.35 || person[k] < 0.5) continue;
    const c: RGB = [lin1(d[k * 4]), lin1(d[k * 4 + 1]), lin1(d[k * 4 + 2])];
    const L = luma(c);
    const dark = smooth(0.18, 0.55, 1 - L / Math.max(Ls, 1e-3));
    const sum = c[0] + c[1] + c[2] + 1e-6;
    const chroma = Math.hypot(c[0] / sum - sr, c[1] / sum - sg);
    const colored = smooth(0.05, 0.14, chroma) * smooth(0.02, 0.08, L);
    const a = Math.max(dark, colored);
    if (a > 0.25) {
      count++;
      const x = k % w;
      const y = (k / w) | 0;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    alpha[k] = a;
  }
  if (count < n * 0.004) return null;
  const pad = Math.round(Math.max(x1 - x0, y1 - y0) * 0.06) + 2;
  x0 = Math.max(0, x0 - pad);
  y0 = Math.max(0, y0 - pad);
  x1 = Math.min(w - 1, x1 + pad);
  y1 = Math.min(h - 1, y1 + pad);
  const cw = x1 - x0 + 1;
  const ch = y1 - y0 + 1;
  const canvas = document.createElement('canvas');
  canvas.width = cw;
  canvas.height = ch;
  const ctx = canvas.getContext('2d')!;
  const img = ctx.createImageData(cw, ch);
  for (let y = 0; y < ch; y++)
    for (let x = 0; x < cw; x++) {
      const k = (y0 + y) * w + (x0 + x);
      const o = (y * cw + x) * 4;
      // 색은 사진의 잉크색을 조금 진하게(피부에 섞인 만큼 되돌림)
      const a = alpha[k];
      img.data[o] = Math.max(0, Math.min(255, d[k * 4] * (1 - 0.2 * a)));
      img.data[o + 1] = Math.max(0, Math.min(255, d[k * 4 + 1] * (1 - 0.2 * a)));
      img.data[o + 2] = Math.max(0, Math.min(255, d[k * 4 + 2] * (1 - 0.2 * a)));
      img.data[o + 3] = Math.round(a * 255);
    }
  ctx.putImageData(img, 0, 0);
  // 도안 크기 상한
  const SIZE = 512;
  const s = Math.min(1, SIZE / Math.max(cw, ch));
  if (s < 1) {
    const small = document.createElement('canvas');
    small.width = Math.max(1, Math.round(cw * s));
    small.height = Math.max(1, Math.round(ch * s));
    small.getContext('2d')!.drawImage(canvas, 0, 0, small.width, small.height);
    return { id: 'photo', name: '사진 도안', aspect: small.width / small.height, canvas: small };
  }
  return { id: 'photo', name: '사진 도안', aspect: cw / ch, canvas };
}

function smooth(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}

/** 상자 흐림(가로·세로 누적합) */
function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / (2 * r + 1);
      s += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / (2 * r + 1);
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}
