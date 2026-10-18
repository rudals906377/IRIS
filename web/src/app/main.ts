// 화면 구성과 사용자 조작. 엔진(engine/)을 불러 카메라·사진 분석·설정을 연결한다.
//
// 흐름은 하나다: 카메라 시작 → 마음에 드는 사진 고르기 → 사진의 메이크업·헤어·네일·타투가 내 모습에 입혀짐.
// 입혀진 부위는 칩으로 끄고 켜고, "진하기" 하나로 전체를 옅게·진하게 한다.
//
// 개발·시험용 주소 인자(화면에는 없고 자동 시험·평가 도구가 쓴다):
//   ?src=sample            예시 영상으로 바로 시작
//   ?src=<경로>             같은 사이트의 동영상으로 바로 시작(시험용)
//   ?ref=<사진 주소>&refmode=auto|makeup|hair|nail|tattoo   사진을 분석해 적용
//   ?look=<룩 이름>         시험용 메이크업 묶음(daily, idol, coral, red, smoky, rose, glam, clear)
//   ?over=-1~1 ?pearl=0~1 ?lstyle=full|gradient|blur   입술 라인·펄·립 모양
//   ?hair=<번호|이름> ?nail=<번호|이름>&nstyle=…&nlen=0~1   시험용 색상표
//   ?tattoo=<도안 id>&place=<위치>&tsize=0~1   시험용 내장 도안(예: tattoo=moon&place=forearmL)
//   ?hud=1                  처리 시간 표시 켜기
//   ?debug=lm,seg           개발자 표시(얼굴 점, 분할)
//   ?delegate=CPU|GPU       처리 장치

import { LOOKS, PALETTES, PART_LABELS, type LookName, type PartName } from '../beauty/palettes.ts';
import type { MakeupLook, RGB } from '../beauty/makeup.ts';
import { BROW_L_LOWER, BROW_L_UPPER, BROW_R_LOWER, BROW_R_UPPER, EYE_L_LOWER, EYE_L_UPPER, EYE_R_LOWER, EYE_R_UPPER, LIPS_OUTER, type LipStyle } from '../beauty/face-regions.ts';
import { builtinDesigns, designFromFile, type TattooDesign } from '../beauty/tattoo-designs.ts';
import { PLACE_LABELS, type TattooPlace } from '../beauty/tattoo-place.ts';
import { NATURAL_APPLE, NATURAL_BROW_L, NATURAL_LINER_L, NATURAL_MID, NATURAL_SHADOW, PhotoAnalyzer, SKIN_REF, josa, paperInk, summaryOf, tattooFromSource, upscaleInkSource, type PhotoMode, type StyleResult } from '../beauty/photo-style.ts';
import { FACE_OVAL, measureFace, type FaceMeasure } from '../beauty/face-measure.ts';
import { matchDarkness, matchLip, matchTint } from '../beauty/style-match.ts';
import { analyzeStyle } from '../beauty/style-ai.ts';
import { hintsFromStyleAI, promptFromStyleAI, type StyleHints } from '../beauty/style-attributes.ts';
import { checkGenServer, generate as genGenerate, genServerUrl, setGenServerUrl, swapFaces, toDataUrl, type GenCategory } from '../beauty/gen-mode.ts';
import { gam, lin, lipTarget, luma } from '../beauty/style-math.ts';
import type { NailStyle } from '../beauty/nail.ts';
import { BeautyEngine } from '../engine/engine.ts';
import { fmt, summarize } from '../engine/stats.ts';
import { DEFAULT_TRACKER_CONFIG, type Delegate, type TrackerConfig } from '../engine/tracker.ts';
import { StickerGL } from '../engine/sticker-gl.ts';
import { parseFace } from '../beauty/face-parse.ts';
import { boxBlur, closeHoles, colorGate, fillInterior, flyaways, guidedFilter, hairBands, hairCore, pureHair, rgb2lab, skinGate, hairDist, hairSpan, recolorHair, type HairBands, type HairDist } from '../beauty/still-hair.ts';
import { growNailMask, hullFill, maskContrast, nailFrame, rimContrast, splitPolished, syntheticNail, transferNail, type NailFrame } from '../beauty/still-nail.ts';
import { samImage, type SamImage } from '../beauty/sam.ts';
import { fillPoly, swapFace } from '../beauty/face-swap.ts';
import { transferMakeup, type MakeupAmounts } from '../beauty/still-makeup.ts';
import { extractDesign, findInk, isDesignSheet, type InkSource } from '../beauty/tattoo-extract.ts';
import { affine3, armStrip, chestInkScale, inkOnArm, limbRegion, maskOutLimbs, measureHalfWidth, stripPoint, transferArmInk, transferTorsoInk, type Arm, type InkMap, type Strip } from '../beauty/still-tattoo.ts';
import { nailQuads } from '../beauty/nail-place.ts';
import { FACE_CUT, applyPose, buildWarpMesh, cutMesh, headProxy, projectMesh, rotationDeg, similarity3, type Pose3, type Vec3, type WarpMesh } from '../beauty/head-warp.ts';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
const base = new URL('./', location.href);
const WASM_BASE = new URL('mediapipe/wasm', base).href;
const canMp4 = document.createElement('video').canPlayType('video/mp4; codecs="avc1.4d401e"') !== '';
const SAMPLE_VIDEO = new URL(canMp4 ? 'samples/sample-person.mp4' : 'samples/sample-person.webm', base).href;

const statusEl = $<HTMLDivElement>('status');
const hudEl = $<HTMLPreElement>('hud');
const guideEl = $<HTMLDivElement>('guide');
const startEl = $<HTMLDivElement>('start');
/** 사진 + 사진 합성(웹캠 없이): 주소에 ?mode=photo. 내 사진을 '멈춘 카메라'로 열어 나머지는 웹캠과 똑같이 처리한다 */
const PHOTO_MODE = params.get('mode') === 'photo';
const dropEl = $<HTMLDivElement>('drop');
const viewEl = $<HTMLDivElement>('view');
const panelEl = $<HTMLElement>('panel');
const btnPhotoEl = $<HTMLButtonElement>('btn-photo');
const pickHintEl = $<HTMLElement>('pick-hint');
const btnGenEl = $<HTMLButtonElement>('btn-gen');
const btnSnapEl = $<HTMLButtonElement>('btn-snap');
const chipsEl = $<HTMLElement>('chips');
const controlsEl = $<HTMLElement>('controls');
const amountEl = $<HTMLInputElement>('amount');
const loopbackEl = $<HTMLPreElement>('loopback-result');
const sizeRowEl = $<HTMLElement>('size-row');
const sizeEl = $<HTMLInputElement>('size');
const styleFileEl = $<HTMLInputElement>('style-file');
const refEl = $<HTMLElement>('ref');
const refImgEl = $<HTMLImageElement>('ref-img');
const refTextEl = $<HTMLElement>('ref-text');

const engine = new BeautyEngine({
  video: $<HTMLVideoElement>('video'),
  canvas: $<HTMLCanvasElement>('gl'),
  overlay: $<HTMLCanvasElement>('overlay'),
  wasmBase: WASM_BASE,
});
// 개발 도구·자동 시험에서 상태를 확인할 수 있게 노출
(window as unknown as { iris: BeautyEngine }).iris = engine;

const coarse = matchMedia('(pointer: coarse)').matches;
const trackerConfig: TrackerConfig = {
  ...DEFAULT_TRACKER_CONFIG,
  delegate: (params.get('delegate') as Delegate) ?? 'GPU',
  segEvery: coarse ? 2 : 1,
  headSeg: params.get('headseg') !== '0',
  // 자체 머리카락 모델(실험): ?hairmodel=models/hair-matte-256.onnx 또는 설정 체크박스로 그 자리에서만 켠다.
  // 예전에는 체크를 저장해 두어, 한 번 켜 본 뒤로 계속 얼룩진 분할로 염색되는 일이 있었다 → 저장하지 않는다
  hairModel: params.get('hairmodel') ?? '',
};
// 자체 손톱 모델(실험): ?nailmodel=models/nail-seg.onnx
const NAIL_MODEL_PARAM = params.get('nailmodel') ?? '';
/** 자체 학습 머리카락 모델(설정 체크박스가 켜면 이 파일) */
const HAIR_MODEL_URL = 'models/hair-matte-256.onnx';

let running = false;
let lastFaceSeen = performance.now();
let loopbackSummary: ReturnType<typeof summarize> | null = null;
let loopbackFailures = 0;

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}
/** 모델 불러오기 안내: 다 불러오면 그 안내일 때만 지운다(그사이 뜬 다른 안내 — 타투가 옷에 가림 등 — 를 지우지 않게) */
let modelMsg = '';
function modelStatus(msg: string): void {
  if (msg) {
    modelMsg = msg;
    setStatus(msg);
  } else if (statusEl.textContent === modelMsg) setStatus('');
}

// ---- 입혀진 스타일 상태 ----

/** 부위별 색(없음이면 null)과 진하기. 사진 분석이 채운다 */
const chosen: Record<PartName, { color: RGB | null; amount: number }> = {
  base: { color: null, amount: 0.5 },
  contour: { color: null, amount: 0.5 },
  lip: { color: null, amount: 0.75 },
  shadow: { color: null, amount: 0.45 },
  blush: { color: null, amount: 0.35 },
  liner: { color: null, amount: 0.85 },
  brow: { color: null, amount: 0.35 },
  hair: { color: null, amount: 0.7 },
  nail: { color: null, amount: 0.95 },
};
let nailStyle: NailStyle = 'solid';
/** 무늬별 둘째 색: 프렌치 끝·도트는 흰색, 그라데이션은 흰색 쪽으로, 글리터는 금색 */
const NAIL_COLOR2: Record<NailStyle, RGB> = { solid: [1, 1, 1], french: [0.97, 0.96, 0.94], gradient: [0.96, 0.9, 0.88], glitter: [0.95, 0.8, 0.45], dots: [0.97, 0.96, 0.94], chrome: [1, 1, 1], jelly: [1, 1, 1], cateye: [0.92, 0.93, 0.97], aurora: [1, 1, 1] };
/** 네일 연장 길이(0 자연 ~ 1 긴 아몬드) */
let nailLength = 0;
/** 헤어 그라데이션 끝 색(옴브레) */
let hairTip: RGB | null = null;
/** 블러셔 위치(0 눈 밑 ~ 1 광대)와 퍼짐 배율 */
let blushPos = 0.5;
let blushSize = 1;
let gloss = 0.3;
/** 아이섀도 눈꼬리 쪽 색(두 색 그라데이션; 없으면 한 색) */
let shadowOut: RGB | null = null;
/** 입술 라인(-1~1)과 아이섀도 펄(0~1) */
let overlip = 0;
let lipStyle: LipStyle = 'full';
let pearl = 0;
/** 사진보다 옅게·진하게: 모든 부위의 진하기에 곱한다(새 사진을 고르면 1) */
let strength = 1;

type Chip = PartName | 'tattoo' | 'hairgen';
/** 사용자가 칩으로 끈 부위 */
const off = new Set<Chip>();
const CHIP_ORDER: Chip[] = ['lip', 'blush', 'shadow', 'liner', 'brow', 'contour', 'base', 'hair', 'hairgen', 'nail', 'tattoo'];
const CHIP_LABELS: Record<Chip, string> = { ...PART_LABELS, tattoo: '타투', hairgen: '헤어(생성)' };

// ---- 타투 ----
const tattoo = {
  design: null as TattooDesign | null,
  place: 'forearmL' as TattooPlace,
  size: 0.5,
  amount: 0.85,
  /** 화면을 눌러 고른 부위 안 위치(없으면 가운데) */
  along: undefined as number | undefined,
  across: undefined as number | undefined,
};
const tattooPickEl = $<HTMLButtonElement>('tattoo-pick');
const tattooCropEl = $<HTMLButtonElement>('tattoo-crop');
let tattooPicking = false;
const INK: [number, number, number] = [0.16, 0.17, 0.2];

const amt = (a: number): number => Math.min(1, a * strength);

function applyTattooToEngine(): void {
  const d = tattoo.design;
  engine.tattoo = d && !off.has('tattoo') && !tatBaked() ? { design: d, place: tattoo.place, size: tattoo.size, amount: amt(tattoo.amount), ink: INK, along: tattoo.along, across: tattoo.across } : null;
  // 사진 모드: 자리·크기·도안이 바뀌면 다시 굽는다
  if (PHOTO_MODE && engine.source.kind === 'image' && !swapMode && placeBaked !== placeKey()) scheduleStillHair();
  // 타투는 몸 관절점이 필요하다: 처음 켤 때 자세 추적을 켠다(모델을 한 번 내려받음)
  if (d && !trackerConfig.pose) {
    trackerConfig.pose = true;
    if (running) void engine.tracker.configure(trackerConfig, modelStatus).then(() => modelStatus(''));
  }
}

function applyLookToEngine(): void {
  const look: MakeupLook = {};
  for (const part of Object.keys(chosen) as PartName[]) {
    const c = chosen[part];
    if (!c.color || off.has(part) || part === 'hair' || part === 'nail') continue;
    // 사진 모드: 고른 사진 화장을 그대로 옮겨 구웠으면(얼굴 바꾸기 포함) 실시간 화장은 눈썹만
    if ((mkBaked() || (PHOTO_MODE && swapMode)) && part !== 'brow') continue;
    if (part === 'lip') look.lip = { color: c.color, amount: amt(c.amount), gloss, over: overlip, style: lipStyle };
    else if (part === 'blush') look.blush = { color: c.color, amount: amt(c.amount), pos: blushPos, size: blushSize };
    else if (part === 'shadow') look.shadow = { color: c.color, amount: amt(c.amount), pearl, colorOut: shadowOut ?? undefined };
    else look[part] = { color: c.color, amount: amt(c.amount) };
  }
  engine.look = look;
  const h = chosen.hair;
  engine.hair = h.color && !off.has('hair') ? { color: h.color, amount: amt(h.amount), tip: hairTip ?? undefined } : null;
  // 사진 모드: 염색은 셰이더 대신 고른 사진의 실제 머리색 분포를 올마다 옮겨 사진에 굽는다(still-hair.ts)
  if (PHOTO_MODE && engine.source.kind === 'image' && !swapMode) {
    engine.hair = null;
    // 네일도 사진에 굽는다(찾은 손톱이 있을 때). 아직 없으면 실시간 네일을 미리보기로 둔다
    if (myNails.length && refNailArt()) engine.nail = null;
    scheduleStillHair();
    if (chosen.nail.color && !off.has('nail')) void ensureNails();
  }
  const n = chosen.nail;
  engine.nail = n.color && !off.has('nail') ? { color: n.color, color2: NAIL_COLOR2[nailStyle], style: nailStyle, amount: amt(n.amount), length: nailLength } : null;
  // 사진 모드: 손 점으로 그리는 실시간 네일은 정지 사진에서 손톱 옆 피부에 칼로 오린 듯 얹히기도 한다
  // → 사진에 구운 네일만 쓴다(손톱을 SAM 으로 오려 고른 사진·가상 손톱 그림을 옮김, 빠진 손톱은 눌러서 더한다)
  if (PHOTO_MODE && engine.source.kind === 'image' && !swapMode) engine.nail = null;
  // 네일은 손 점이 필요하다: 처음 켤 때 손 추적을 켠다
  if (n.color && !trackerConfig.hands) {
    trackerConfig.hands = true;
    if (running) void engine.tracker.configure(trackerConfig, modelStatus).then(() => modelStatus(''));
  }
}

function applyAll(): void {
  applyLookToEngine();
  applyTattooToEngine();
}

/** 시험용 메이크업 묶음(?look=). 화면에는 메뉴가 없다 */
function selectLook(name: LookName): void {
  const l = LOOKS[name];
  for (const part of Object.keys(chosen) as PartName[]) {
    if (part === 'hair' || part === 'nail') continue;
    const v = l.parts[part];
    chosen[part].color = v ? v.color : null;
    if (v) chosen[part].amount = v.amount;
  }
  gloss = l.gloss ?? gloss;
  pearl = l.pearl ?? 0;
  shadowOut = null;
  lipStyle = l.lipStyle ?? 'full';
  blushPos = 0.5;
  blushSize = 1;
  applyLookToEngine();
  renderChips();
}

const fromHex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as RGB;
const c = (h: string): RGB => fromHex(h);

/** 지금 입혀진 부위를 칩으로 보인다. 누르면 그 부위만 끄고 켠다 */
function renderChips(): void {
  const items = CHIP_ORDER.filter((k) => (k === 'tattoo' ? !!tattoo.design : k === 'hairgen' ? !!hairSticker : !!chosen[k].color));
  chipsEl.replaceChildren(
    ...items.map((k) => {
      const b = document.createElement('button');
      b.className = 'chip' + (off.has(k) ? '' : ' on');
      b.textContent = CHIP_LABELS[k];
      b.title = off.has(k) ? '다시 켜기' : '잠시 끄기';
      b.addEventListener('click', () => {
        const turnOn = off.has(k);
        if (turnOn) off.delete(k);
        else off.add(k);
        applyAll();
        renderChips();
        // 꺼 두었던 메이크업을 처음 켜면 그때 사진과 맞추는 되먹임을 돌린다
        const measure = lastStyle?.measure;
        if (turnOn && k !== 'hair' && k !== 'nail' && k !== 'tattoo' && measure && lastMatch.length === 0 && running) void feedback(measure);
      });
      return b;
    }),
  );
  const any = items.length > 0;
  chipsEl.hidden = !any;
  controlsEl.hidden = !any;
  // 사진을 고른 뒤에는 큰 단추 대신 사진 줄의 "다른 사진"을 쓴다
  btnPhotoEl.hidden = !!lastStyle;
  pickHintEl.hidden = any || !!lastStyle;
  const hasTattoo = !!tattoo.design;
  sizeRowEl.hidden = !hasTattoo;
  sizeEl.value = String(tattoo.size);
  tattooPickEl.hidden = !hasTattoo;
  tattooCropEl.hidden = !lastStyle?.tattooSrc;
}

/** 입혀진 것을 모두 지운다 */
function clearAll(): void {
  for (const part of Object.keys(chosen) as PartName[]) chosen[part].color = null;
  hairTip = null;
  shadowOut = null;
  tattoo.design = null;
  tattoo.along = tattoo.across = undefined;
  hairSticker = null;
  engine.overlayDraw = null;
  off.clear();
  strength = 1;
  amountEl.value = '1';
  lastStyle = null;
  lastSource = null;
  refEl.hidden = true;
  applyAll();
  renderChips();
}

// ---- 사진 고르기 ----
let photoAnalyzer: PhotoAnalyzer | null = null;
/** 마지막 분석 결과(개발 도구·자동 시험용) */
let lastStyle: StyleResult | null = null;
/** 마지막으로 고른 사진 원본(영역 고르기용) */
let lastSource: ImageBitmap | null = null;

/** 사진 속에 사람이 없고 바탕이 희거나 투명하면(도안 그림) 그대로 타투 도안으로 쓴다 */
function looksLikeDesign(thumb: HTMLCanvasElement): boolean {
  const ctx = thumb.getContext('2d', { willReadFrequently: true })!;
  const d = ctx.getImageData(0, 0, thumb.width, thumb.height).data;
  let blank = 0;
  const n = d.length / 4;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] < 30 || (d[i] > 235 && d[i + 1] > 235 && d[i + 2] > 235)) blank++;
  }
  return blank > n * 0.5;
}

const GROUPS = ['makeup', 'hair', 'nail', 'tattoo'] as const;
type Group = (typeof GROUPS)[number];
const MAKEUP_PARTS = ['lip', 'blush', 'shadow', 'liner', 'brow', 'contour', 'base'] as const;
const isGroup = (x: unknown): x is Group => (GROUPS as readonly string[]).includes(x as string);

async function applyPhoto(source: ImageBitmap | HTMLImageElement, mode: PhotoMode, file?: File, original?: ImageBitmap): Promise<StyleResult> {
  nailLabelColor = null;
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  if (source instanceof ImageBitmap) lastSource = original ?? source;
  setStatus('사진 보는 중…');
  let r = await photoAnalyzer.analyze(source, mode, setStatus);
  // 스타일 AI: 사진이 무엇에 관한 것인지(메이크업·헤어·네일·타투)와 무늬·모양·기법. 색은 위에서 잰 값을 쓴다.
  // 실패해도 색 적용은 계속한다
  if ($<HTMLInputElement>('opt-style-ai').checked && source instanceof ImageBitmap) {
    try {
      const cat = mode === 'auto' ? 'auto' : mode;
      const ai = await analyzeStyle(source, cat, (p) => {
        if (p.status === 'progress' && p.file && p.progress !== undefined) setStatus(`스타일 AI 모델 내려받는 중 ${Math.round(p.progress)}%`);
        else if (p.status === 'ready') setStatus('스타일 읽는 중…');
      });
      // 측정이 AI 가 말한 분야를 못 찾았으면(예: 등 타투 사진에서 손만 찾아 네일로 읽음) 그 분야로 다시 본다
      const want = ai.category;
      if (mode === 'auto' && isGroup(want) && !r[want]) {
        setStatus('사진 다시 보는 중…');
        const r2 = await photoAnalyzer.analyze(source, want, setStatus);
        if (r2[want]) r = r2;
      }
      r.ai = { hints: hintsFromStyleAI(ai), headline: ai.headline, description: ai.description_ko, raw: ai };
      applyHints(r, r.ai.hints);
      if (r.summary.startsWith('사진에서 가져왔어요')) r.summary = summaryOf(r);
    } catch (err) {
      console.warn('스타일 AI를 쓰지 못했습니다(색만 적용)', err);
    }
  }
  // 사람이 없는 도안 그림(흰 바탕·투명 PNG)은 그대로 타투 도안으로. 다만 스타일 AI 가 다른 분야(헤어·네일·메이크업)라고 본 사진
  // (흰 바탕에 펼친 티셔츠 사진이 '도안'으로 읽혀 타투 도안이 됐음)은 도안이 아니다 → 찾지 못했다고 알린다
  const aiOther = ['hair', 'nail', 'makeup'].includes(String(r.ai?.raw?.category ?? ''));
  if (!r.makeup && !r.hair && !r.nail && !r.tattoo && file && !aiOther && looksLikeDesign(r.thumb)) {
    try {
      r.tattoo = await designFromFile(file);
      r.summary = '그림을 타투 도안으로 가져왔어요';
    } catch (err) {
      console.warn('도안으로 열지 못했습니다', err);
    }
  }
  // 도안 시트(종이에 그린 도안 여러 개, 사람 없음): 종이를 바탕으로 잉크를 뽑는다. 하나만 쓰려면 ✂️ 도안 고르기
  const aiTattoo = r.ai?.raw?.category === 'tattoo' || r.ai?.raw?.secondary?.category === 'tattoo';
  if (!r.makeup && !r.hair && !r.nail && !r.tattoo && !r.faces && (aiTattoo || mode === 'tattoo') && !aiOther && photoAnalyzer.lastInk) {
    const ps = paperInk(photoAnalyzer.lastInk);
    // 도안 수십 개가 빽빽한 시트는 자동으로 하나를 고르지 않는다(시트 전체는 바코드처럼, 한 덩어리는 조각난 도안이 됐음)
    const sheet = isDesignSheet(ps);
    const t = sheet ? null : tattooFromSource(ps, null, PHOTO_MODE ? 1600 : 512);
    if (sheet) r.summary = '도안이 여러 개인 시트라 하나를 고르지 못했어요 · ✂️ 도안 고르기로 원하는 도안 하나를 감싸 주세요';
    if (t) {
      r.tattoo = t.design;
      r.tattooSrc = { src: ps, rect: t.rect, inkMask: t.inkMask, paper: true };
      r.summary = '도안 그림에서 가져왔어요 · 여러 개면 ✂️ 도안 고르기로 하나를 골라요';
    }
  }
  // 사진의 주제(AI 분야)가 아닌 것은 찾았어도 꺼 둔다: 립 사진의 머리색까지 바꾸지 않게. 칩을 누르면 켜진다
  const cats = new Set([r.ai?.raw?.category, r.ai?.raw?.secondary?.category].filter(isGroup));
  const on = (g: Group): boolean => cats.size === 0 || cats.has(g);
  const setOff = (keys: readonly Chip[], v: boolean): void => {
    for (const k of keys) if (v) off.add(k);
    else off.delete(k);
  };
  lastStyle = r;
  strength = 1;
  amountEl.value = '1';
  const m = r.makeup;
  if (m) {
    // 사진의 메이크업으로 바꾼다(사진에 없는 부위는 지움). 피부 보정은 사진이 보정된 경우가 많아 약하게 켠다
    for (const part of ['lip', 'blush', 'shadow', 'liner', 'brow', 'contour'] as const) {
      const v = m[part];
      chosen[part].color = v ? v.color : null;
      if (v) chosen[part].amount = v.amount;
    }
    shadowOut = m.shadow?.colorOut ?? null;
    blushPos = m.blush?.pos ?? 0.5;
    blushSize = m.blush?.size ?? 1;
    chosen.base.color = c('#d1a38a');
    chosen.base.amount = 0.35;
    setOff(MAKEUP_PARTS, !on('makeup'));
    if (m.lip) {
      gloss = m.lip.gloss;
      lipStyle = m.lip.style;
    }
    pearl = 0;
  }
  if (r.hair) {
    chosen.hair.color = r.hair.color;
    chosen.hair.amount = r.hair.amount;
    hairTip = r.hair.tip;
    setOff(['hair'], !on('hair'));
  }
  if (r.nail) {
    chosen.nail.color = r.nail.color;
    nailStyle = r.nail.style;
    setOff(['nail'], !on('nail'));
  }
  applyLookToEngine();
  if (r.tattoo) {
    tattoo.design = r.tattoo;
    // 사진 모드: 원본 해상도로 도안을 다시 뽑아 가는 선까지 살린다
    // (분석한 그림 기준 좌표이므로 ✂️ 영역으로 자른 경우에도 그 그림에서 다시 뽑는다)
    if (PHOTO_MODE && r.tattooSrc) {
      const hi = upscaleInkSource(r.tattooSrc.src, source, 1600);
      if (hi.scale > 1) {
        const rc = r.tattooSrc.rect;
        const t = tattooFromSource(hi.src, { x: rc.x * hi.scale, y: rc.y * hi.scale, w: rc.w * hi.scale, h: rc.h * hi.scale }, 1600);
        if (t) {
          tattoo.design = t.design;
          r.tattoo = t.design;
          r.tattooSrc = { src: hi.src, rect: t.rect, inkMask: t.inkMask, paper: r.tattooSrc.paper };
        }
      }
    }
    tattoo.along = tattoo.across = undefined;
    // 사진 모드: 팔 타투면 팔 대 팔로 옮겨 사진에 굽는다(still-tattoo.ts)
    refTat = PHOTO_MODE && r.tattooSrc && !r.tattooSrc.paper && tattoo.design.canvas
      ? { img: source, w: r.tattooSrc.src.w, h: r.tattooSrc.src.h, design: tattoo.design, rect: r.tattooSrc.rect, src: r.tattooSrc.src }
      : null;
    // 가슴 도안은 고른 사진에서 잰 크기(도안 세로 / 가슴 피부 폭)로: 가슴 평면 굽기(chestPlacePlan)의 세로 기준이 어깨 폭 45% 이므로
    // 가슴 폭 ≈ 어깨 폭 0.9 로 보고 size = (ratio·0.9/0.45 − 0.25)/0.6. AI 크기 말('스몰' 0.3)은 실제의 절반도 안 됐다
    if (refTat && tattoo.place === 'chest') {
      const cs = chestInkScale(refTat.src.skin, refTat.w, refTat.h, refTat.rect);
      dbgTat({ chestScale: cs });
      if (cs) tattoo.size = Math.min(1, Math.max(0.15, (cs.ratio * 2 - 0.25) / 0.6));
    }
    setOff(['tattoo'], !on('tattoo'));
    applyTattooToEngine();
  }
  lastMatch = [];
  renderChips();
  refImgEl.src = r.thumb.toDataURL('image/jpeg', 0.8);
  // 못 찾은 분야 안내는 최종 결과 기준(도안 그림·종이 도안으로 나중에 타투가 잡힌 경우 '타투는 찾지 못했어요'가 남았음)
  const got: Record<string, unknown> = { 메이크업: r.makeup, 헤어: r.hair, 네일: r.nail, 타투: r.tattoo };
  const missed = (r.missed ?? []).filter((k) => !got[k]);
  const missedNote = missed.length && !r.summary.includes('못했어요') ? ` · ${missed.join('·')}${josa(missed[missed.length - 1], '은', '는')} 이 사진에서 찾지 못했어요` : '';
  // AI 가 읽은 스타일 설명은 네 분야 중 하나라도 찾았을 때만 붙인다(티셔츠 사진에 '긴머리 슬릭백' 같은 엉뚱한 설명이 붙었음)
  const anyFound = !!(r.makeup || r.hair || r.nail || r.tattoo);
  refTextEl.textContent = (r.ai?.headline && anyFound ? `${r.summary} · ${r.ai.headline}` : r.summary) + (r.faces && r.faces > 1 ? ` · 얼굴 ${r.faces}명 중 큰 얼굴(다른 사람은 ✂️ 영역)` : '') + missedNote;
  refEl.hidden = false;
  setStatus(running ? '' : PHOTO_MODE ? '내 사진을 올리면 바로 합쳐져요' : '카메라를 시작하면 바로 입혀져요');
  if (r.makeup && r.measure && !off.has('lip')) {
    // 영상이 아직 시작 전이면(사진 분석이 먼저 끝난 경우) 잠시 기다린다
    for (let i = 0; i < 100 && !running; i++) await new Promise((res) => setTimeout(res, 100));
    if (running) await feedback(r.measure);
  }
  return r;
}

/** 사진 되먹임을 한 번 돌린다(실패해도 앱은 계속) */
async function feedback(measure: FaceMeasure): Promise<void> {
  try {
    await matchToPhoto(measure);
  } catch (err) {
    console.error('되먹임 실패', err);
    lastMatch.push({ error: String(err) });
  }
}

/**
 * 스타일 AI 힌트를 분석 결과에 얹는다: 종류(모양·무늬·기법)는 AI가, 색은 측정값이 정한다.
 * 측정에서 못 찾은 부위는 그 종류의 기본색으로 채우고, AI가 '없음'이라 하면 뺀다.
 */
/** 측정이 손톱을 못 찾아 스타일 AI 색 이름으로 정한 네일 색(sRGB 0~1). 고른 사진 손톱 찾기 결과를 검증하는 데 쓴다 */
let nailLabelColor: RGB | null = null;
/** 스타일 AI의 네일 색 이름 → 색(이름에 든 낱말로) */
function nailColorFromLabel(label: string | undefined): RGB | null {
  if (!label) return null;
  const table: [string, string][] = [
    ['연핑크', '#f2c4c8'], ['누드', '#e6c2ad'], ['베이지', '#e6cfb8'], ['코랄', '#f08070'], ['버건디', '#6a1a2a'], ['와인', '#6a1a2a'],
    ['레드', '#c4202a'], ['빨강', '#c4202a'], ['핑크', '#f090a8'], ['화이트', '#f4f1ee'], ['흰', '#f4f1ee'], ['블랙', '#1c1a1c'], ['검정', '#1c1a1c'],
    ['실버', '#c8c8cc'], ['그레이', '#9a9aa0'], ['회색', '#9a9aa0'], ['네이비', '#283c70'], ['블루', '#5a7ac8'], ['파랑', '#5a7ac8'], ['민트', '#a0e0c8'],
    ['그린', '#6aa070'], ['초록', '#6aa070'], ['퍼플', '#9a70c0'], ['보라', '#9a70c0'], ['라벤더', '#c0a8e0'], ['골드', '#d2b060'],
    ['브라운', '#7a5040'], ['갈색', '#7a5040'], ['오렌지', '#f08a3a'], ['옐로', '#f2d26a'], ['노랑', '#f2d26a'], ['투명', '#f0d6d2'], ['클리어', '#f0d6d2'],
  ];
  for (const [k, hex] of table) if (label.includes(k)) return c(hex);
  return null;
}
function applyHints(r: StyleResult, h: StyleHints): void {
  const m = r.makeup;
  if (m && h.makeup) {
    const k = h.makeup;
    if (m.lip) {
      if (k.lipStyle) m.lip.style = k.lipStyle;
      if (k.gloss === 'glossy') m.lip.gloss = Math.max(m.lip.gloss, 0.55);
      else if (k.gloss === 'matte') m.lip.gloss = Math.min(m.lip.gloss, 0.12);
    }
    if (k.shadow) {
      const defaults: Record<string, RGB> = { shade: c('#8a5a44'), smoky: c('#4a4550'), pearl: c('#c9a27a'), color: c('#7a6ab8'), pink: c('#d98fa0') };
      if (!m.shadow) m.shadow = { color: defaults[k.shadow], amount: k.shadow === 'smoky' ? 0.8 : 0.5 };
      else if (k.shadow === 'smoky') m.shadow = { color: gam(lin(m.shadow.color).map((v) => v * 0.6) as RGB), amount: Math.max(m.shadow.amount, 0.75) };
    }
    if (k.liner === 'cat') m.liner = { color: [0.1, 0.08, 0.08], amount: 0.9 };
    else if (k.liner === 'soft' && !m.liner) m.liner = { color: [0.23, 0.16, 0.13], amount: 0.45 };
    // 스타일 AI의 '블러셔 없음'은 측정이 약할 때만 따른다(사진에 붉은 기가 뚜렷하면 측정을 믿는다)
    // (0.15 는 옅은 분홍 블러셔(10·20번, 붉은 기 0.10~0.13)까지 지워 버렸다 → 0.09)
    if (k.blush === 'none' && (m.blush?.redness ?? 0) < 0.09) m.blush = undefined;
    else if (k.blush && !m.blush) m.blush = { color: k.blush === 'coral' ? c('#cb795e') : c('#cb7979'), amount: 0.9, pos: 0.15, size: 1.25 };
    if (k.contour === 'shading') m.contour = { color: c('#8c7a70'), amount: Math.max(m.contour?.amount ?? 0, 0.5) };
    else if (k.contour === 'highlight') m.contour = { color: c('#9a7862'), amount: Math.max(m.contour?.amount ?? 0, 0.35) };
    if (k.base === 'dewy') {
      chosen.base.amount = 0.45;
      if (m.lip) m.lip.gloss = Math.max(m.lip.gloss, 0.45);
    } else if (k.base === 'matte') chosen.base.amount = 0.5;
  }
  if (r.hair && h.hair?.tech) {
    if ((h.hair.tech === 'ombre' || h.hair.tech === 'balayage') && !r.hair.tip) {
      // 끝이 밝은 그라데이션: 잰 색을 뿌리로, 끝은 그보다 밝게
      r.hair.tip = gam(lin(r.hair.color).map((v) => Math.min(1, v * 1.8 + 0.03)) as RGB);
    } else if (h.hair.tech === 'solid') r.hair.tip = null;
  }
  // 스타일 AI가 네일 사진이라고 봤는데 색 측정이 손톱을 못 찾았으면(피부색에 가까운 연핑크·누드·시럽 네일) AI가 읽은 색 이름으로.
  // 그때 측정이 잡은 '타투'(손가락 주름·그늘)는 AI가 타투라고 하지 않았으면 버린다
  // AI 가 읽은 네일 색 이름은 측정이 손톱을 찾았어도 기억한다: 고른 사진에서 오린 '손톱'의 색이 이와 크게 다르면(ΔE 35 넘게)
  // 손톱이 아니라 배경을 오린 것(13번: 흰 털옷 + 팔찌 조각(회색)이 손톱으로 잡혀 내 손톱이 회색 털로 칠해졌음)
  if (h.category === 'nail') nailLabelColor = nailColorFromLabel(h.nail?.colorLabel);
  if (h.category === 'nail' && !r.nail) {
    const col = nailLabelColor;
    if (col) {
      // 앱의 색(RGB)은 선형값(hex() 가 감마로 바꿈): 표의 sRGB 를 선형으로
      r.nail = { color: lin(col), style: h.nail?.style ?? 'solid' };
      if (r.tattoo && !h.tattoo) {
        r.tattoo = undefined;
        r.tattooSrc = undefined;
      }
    }
  }
  if (r.nail && h.nail) {
    if (h.nail.style) r.nail.style = h.nail.style;
    if (h.nail.length !== undefined) nailLength = h.nail.length;
  }
  if (r.tattoo && h.tattoo) {
    if (h.tattoo.place) tattoo.place = h.tattoo.place;
    if (h.tattoo.size !== undefined) tattoo.size = h.tattoo.size;
  }
}

/** 고른 사진 파일 하나를 분석해 입힌다 */
async function pickFile(f: File): Promise<void> {
  if (!f.type.startsWith('image/')) {
    setStatus('사진 파일(JPG·PNG)을 골라 주세요');
    return;
  }
  try {
    await applyPhoto(await createImageBitmap(f), (params.get('refmode') as PhotoMode | null) ?? 'auto', f);
  } catch (err) {
    console.error(err);
    setStatus(`사진을 읽지 못했어요: ${(err as Error).message}`);
  }
}

styleFileEl.addEventListener('change', () => {
  const f = styleFileEl.files?.[0];
  styleFileEl.value = '';
  if (f) void pickFile(f);
});
btnPhotoEl.addEventListener('click', () => styleFileEl.click());
$('ref-change').addEventListener('click', () => styleFileEl.click());
$('ref-clear').addEventListener('click', clearAll);

// 사진을 끌어다 놓거나(드래그) 붙여 넣어도(Ctrl+V) 된다
let dragDepth = 0;
document.addEventListener('dragenter', (e) => {
  if (!e.dataTransfer?.types.includes('Files')) return;
  dragDepth++;
  dropEl.hidden = false;
});
document.addEventListener('dragleave', () => {
  if (--dragDepth <= 0) {
    dragDepth = 0;
    dropEl.hidden = true;
  }
});
document.addEventListener('dragover', (e) => {
  if (e.dataTransfer?.types.includes('Files')) e.preventDefault();
});
document.addEventListener('drop', (e) => {
  dragDepth = 0;
  dropEl.hidden = true;
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  e.preventDefault();
  // 사진 합성 모드에서 아직 내 사진이 없으면 놓은 사진이 내 사진
  if (PHOTO_MODE && !running) void openMyPhoto(f);
  else void pickFile(f);
});
document.addEventListener('paste', (e) => {
  const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith('image/'));
  const f = item?.getAsFile();
  if (!f) return;
  if (PHOTO_MODE && !running) void openMyPhoto(f);
  else void pickFile(f);
});

// ---- 사진 + 사진 합성: 내 사진 ----
const myPhotoFileEl = $<HTMLInputElement>('myphoto-file');
const btnMyPhotoEl = $<HTMLButtonElement>('btn-myphoto');
/**
 * 사진 모드 염색: ① 내 사진 머리 주변을 정밀 얼굴 분할(한 번) → 원본 해상도 알파(가이디드 필터)
 * ② 고른 사진 머리카락의 밝기 분위별 색(한 번) ③ 내 머리카락 밝기 순위를 그 색으로 → 사진에 구워 엔진에 넣는다.
 * 화장·네일·타투는 엔진이 그 위에 그대로 그린다.
 */
let photoOriginal: ImageBitmap | null = null;
let mineHair: { alpha: Float32Array; rgba: Uint8ClampedArray; w: number; h: number; dist: HairDist } | null = null;
/** 내 머리카락 매트를 못 만든 까닭(사용자 안내). 머리카락 픽셀이 300 도 안 되는 아주 짧은·탈색한 머리 */
let mineHairNote = '';
/** 이번 굽기에서 염색을 건너뛴 까닭(새까만 머리 + 밝은 염색). 굽기 끝에 안내가 지워지지 않게 남긴다 */
let hairSkipNote = '';
/** 화장 계획을 못 세운 까닭(얼굴 점 없음). 굽기 끝에 알린다 */
let mkPlanNote = '';
let refHair: { src: ImageBitmap; dist: HairDist | null; bands: HairBands | null } | null = null;
let stillHairTimer = 0;
let stillHairBusy: Promise<void> | null = null;
let stillHairKey = '';
function scheduleStillHair(): void {
  clearTimeout(stillHairTimer);
  stillHairTimer = window.setTimeout(() => void updateStillHair(), 200);
}
async function mineHairMatte(): Promise<typeof mineHair> {
  const orig = photoOriginal;
  if (!orig) return null;
  if (mineHair) return mineHair;
  // 얼굴 점이 잡힐 때까지 잠깐 기다린다
  let r: { x: number; y: number; size: number } | null = null;
  // 실제 시간으로 2.5초까지만(큰 사진은 화면 처리에 밀려 100ms 기다림이 0.5초씩 걸려 80번이면 30초 넘게 기다렸다).
  // 못 잡으면 아래 사진용 얼굴 점으로
  const t0 = performance.now();
  while (!(r = engine.tracker.headRect(engine.source.width, engine.source.height)) && performance.now() - t0 < 2500) await new Promise((res) => setTimeout(res, 100));
  if (photoOriginal !== orig) return null;
  // 실시간 얼굴 추적이 못 잡는 작은 얼굴(전신 사진)은 사진용 얼굴 점(몸 관절점으로 머리를 잘라 다시 찾음)으로 머리 상자를 만든다
  if (!r) {
    photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
    const pts = await photoAnalyzer.faceLandmarks(stillCanvas()).catch(() => null);
    if (!pts || photoOriginal !== orig) return null;
    const W0 = engine.source.width, H0 = engine.source.height;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const q of pts) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
    // 실시간 추적의 headRect 와 같은 규칙: 얼굴 폭 2.4배 정사각형, 중심은 얼굴 중심보다 위
    const fw = x1 - x0, fh = y1 - y0, size = Math.min(Math.max(fw * 2.4, fh * 2), W0, H0);
    if (size < 64) return null;
    r = { x: Math.max(0, Math.min(W0 - size, (x0 + x1) / 2 - size / 2)), y: Math.max(0, Math.min(H0 - size, (y0 + y1) / 2 - fh * 0.35 - size / 2)), size };
  }
  const W = engine.source.width, H = engine.source.height;
  // 긴 머리까지 들어오게 머리 상자를 키우고 조금 아래로
  const size = Math.min(Math.max(W, H), r.size * 1.45);
  const rect = { x: r.x + r.size / 2 - size / 2, y: r.y + r.size * 0.62 - size / 2, size };
  const s = orig.width / W;
  setStatus('머리카락을 정밀하게 나누는 중…');
  await new Promise((res) => setTimeout(res, 30)); // 안내 문구가 먼저 보이게
  const p = await parseFace(orig, { x: rect.x * s, y: rect.y * s, size: rect.size * s }, setStatus);
  if (photoOriginal !== orig) return null;
  engine.tracker.setStill({ ...p, rect });
  // 처리 해상도의 내 사진
  const cv = document.createElement('canvas');
  cv.width = W;
  cv.height = H;
  const cx = cv.getContext('2d', { willReadFrequently: true })!;
  cx.drawImage(orig, 0, 0, W, H);
  const rgba = cx.getImageData(0, 0, W, H).data;
  // 512 확률 → 처리 해상도(상자 밖은 0) → 내 사진 밝기로 가이디드 필터
  const coarse = new Float32Array(W * H);
  const faceP = new Float32Array(W * H);
  const N = p.size;
  for (let y = Math.max(0, Math.floor(rect.y)); y < Math.min(H, Math.ceil(rect.y + rect.size)); y++) {
    const v = ((y - rect.y) / rect.size) * N - 0.5;
    const y0 = Math.max(0, Math.min(N - 2, Math.floor(v)));
    const ty = Math.max(0, Math.min(1, v - y0));
    for (let x = Math.max(0, Math.floor(rect.x)); x < Math.min(W, Math.ceil(rect.x + rect.size)); x++) {
      const u = ((x - rect.x) / rect.size) * N - 0.5;
      const x0 = Math.max(0, Math.min(N - 2, Math.floor(u)));
      const tx = Math.max(0, Math.min(1, u - x0));
      const h = p.hair;
      coarse[y * W + x] = ((h[y0 * N + x0] * (1 - tx) + h[y0 * N + x0 + 1] * tx) * (1 - ty) + (h[(y0 + 1) * N + x0] * (1 - tx) + h[(y0 + 1) * N + x0 + 1] * tx) * ty) / 255;
      const f = p.face, b = p.body;
      const j = y0 * N + x0;
      faceP[y * W + x] = Math.max(f[j], b[j], f[j + 1], b[j + 1], f[j + N], b[j + N], f[j + N + 1], b[j + N + 1]) / 255;
    }
  }
  // 긴 머리는 머리 상자 밖(어깨 아래)까지 내려온다: 상자 밖은 사진 전체 머리카락 분할로, 상자 아래쪽 가장자리는 둘을 섞는다
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  const segHair = await photoAnalyzer.hairMask(cv).catch(() => null);
  if (photoOriginal !== orig) return null;
  if (segHair) {
    const bx0 = rect.x, by0 = rect.y, bx1 = rect.x + rect.size, by1 = rect.y + rect.size;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x;
      const inBox = x >= bx0 && x < bx1 && y >= by0 && y < by1;
      // 상자 안: 가장자리(아래·옆 15%)로 갈수록 분할을 더 믿는다
      const edge = inBox ? Math.max(0, 1 - Math.min(by1 - y, x - bx0, bx1 - x) / (rect.size * 0.15)) : 1;
      coarse[i] = Math.max(coarse[i], segHair[i] * edge);
    }
  }
  const guide = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) guide[i] = (0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]) / 255;
  const px = Math.max(2, Math.round(rect.size / 300));
  const alpha = guidedFilter(guide, coarse, W, H, px, 2e-3);
  // 확실하지 않은 낮은 값은 지운다(배경·피부로 번지는 옅은 띠 방지), 높은 값은 1 로.
  // 가이디드 필터는 어두운 머리 + 어두운 배경에서 밖으로 번진다 → 분할이 머리카락이라고 본 곳(조금 넓혀서) 밖은 쓰지 않는다
  // 배경(어두운 나무·벽)이 머리카락으로 읽힌 곳은 실시간 분할의 '사람' 밖이면 지운다
  const person = engine.tracker.personMask(W, H);
  for (let i = 0; i < alpha.length; i++) {
    let gate = Math.max(0, Math.min(1, (coarse[i] - 0.25) / 0.35));
    if (person) gate = Math.min(gate, Math.max(0, Math.min(1, (person[i] - 0.3) / 0.4)));
    alpha[i] = Math.min(Math.max(0, Math.min(1, (alpha[i] - 0.15) / 0.7)), gate);
  }
  // 분할이 덜 확신하는 머리(아주 짧은 머리 등)는 머리 속 알파가 0.7 언저리에 머문다 → 그대로면 원래 머리색이 30% 비쳐
  // 밝게 염색해도 회색으로 탁해졌다. 머리 알파 95% 자리가 0.9 아래면 그 값이 1 이 되게 늘린다
  {
    const av: number[] = [];
    for (let i = 0; i < alpha.length; i += 3) if (alpha[i] > 0.2) av.push(alpha[i]);
    if (av.length > 200) {
      av.sort((p, q) => p - q);
      const top = av[Math.floor((av.length - 1) * 0.95)];
      if (top > 0.3 && top < 0.9) for (let i = 0; i < alpha.length; i++) alpha[i] = Math.min(1, alpha[i] / top);
    }
  }
  // 아주 어두운 머리카락 덩어리는 분할·필터가 놓쳐 작은 구멍(원래 색 얼룩)이 된다 → 머리 크기에 맞춰 닫기
  const closed = closeHoles(alpha, W, H, Math.max(3, Math.round(rect.size / 90)));
  for (let i = 0; i < alpha.length; i++) alpha[i] = closed[i];
  // 가장자리에서 내 머리색과 다른 픽셀(어두운 배경)은 뺀다
  colorGate(rgba, alpha, coarse);
  // 머리 속(둘레가 다 머리카락)은 알파를 1 쪽으로: 분할이 덜 확신한(뒤로 넘긴 머리) 머리 속이 0.85, 정수리 윤기는
  // 색 거르기에서 0.5 로 빠져 원래 색이 비쳤다(밝게 염색하면 탁하고 가르마 옆이 회색 얼룩). 색 거르기 뒤에 한다
  fillInterior(alpha, W, H, Math.max(4, Math.round(rect.size / 25)));
  // 밖으로 날린 가는 잔머리도 같이 물들인다(안 하면 밝은 염색에서 머리 둘레가 검은 테두리처럼 남는다)
  flyaways(rgba, W, H, alpha, rect.size / 20, faceP);
  // 가르마 두피·시스루 앞머리 사이 이마처럼 피부색인 곳은 덜 물들인다(밝게 염색하면 거친 밝은 얼룩이 됐다)
  skinGate(rgba, alpha, faceP);
  const dist = hairDist(rgba, alpha);
  if (!dist) {
    // 확실한 머리카락이 300px 도 안 됨(밝게 탈색한 버즈컷 00121): 실시간 염색으로 떨어뜨려도 아무것도 안 바뀌고 안내도 없어
    // '아무 일도 없는' 화면이 됐다 → 못 한다고 알린다
    mineHairNote = '내 사진에서 머리카락을 거의 찾지 못해 염색할 수 없어요(아주 짧거나 밝게 탈색한 머리). 머리가 잘 보이는 사진을 써 주세요';
    return null;
  }
  mineHairNote = '';
  mineHair = { alpha, rgba, w: W, h: H, dist };
  return mineHair;
}
async function refHairDist(): Promise<HairDist | null> {
  const src = lastSource;
  if (!src) return null;
  if (refHair?.src === src) return refHair.dist;
  // 고른 사진 전체를 정사각형으로(남는 곳은 검게) 나눠 머리카락 픽셀의 색 분포만 쓴다
  const size = Math.max(src.width, src.height);
  const p = await parseFace(src, { x: (src.width - size) / 2, y: (src.height - size) / 2, size });
  const a = new Float32Array(p.size * p.size);
  for (let i = 0; i < a.length; i++) a[i] = p.hair[i] / 255;
  // 머리 속만(둘레 4칸, 512 기준): 시스루 앞머리 사이 이마·가닥 사이 배경색이 머리색으로 섞이지 않게
  // + 색이 피부·배경에 더 가까운 픽셀(시스루 앞머리 사이 이마, 가닥 사이 배경)은 뺀다
  const face = new Float32Array(a.length), body = new Float32Array(a.length), bg = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) {
    face[i] = p.face[i] / 255;
    body[i] = p.body[i] / 255;
    bg[i] = 1 - Math.max(a[i], face[i], body[i]);
  }
  const core = pureHair(p.rgba, hairCore(a, p.size, p.size, 0.7, 4), [face, body, bg]);
  let dist = hairDist(p.rgba, core, 0.7, 200), bands = hairBands(p.rgba, p.size, core, 0.7, 200);
  if (!dist) {
    // 얼굴이 없는 사진(미용실 뒷모습 염색 사진)은 정밀 얼굴 분할이 머리카락을 거의 못 잡아 분포가 비고, 실시간 염색으로
    // 떨어져 토막 난 줄무늬가 됐다 → 사진 분석이 쓰는 인물 분할(MediaPipe)의 머리카락 확률로 분포를 만든다
    photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
    const cv = document.createElement('canvas');
    cv.width = cv.height = p.size;
    const ctx = cv.getContext('2d')!;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, p.size, p.size);
    ctx.drawImage(src, (src.width - size) / 2, (src.height - size) / 2, size, size, 0, 0, p.size, p.size);
    const hm = await photoAnalyzer.hairMask(cv).catch(() => null);
    if (hm) {
      const core2 = pureHair(p.rgba, hairCore(hm, p.size, p.size, 0.5, 4), [face, body, bg]);
      dist = hairDist(p.rgba, core2, 0.7, 200);
      bands = hairBands(p.rgba, p.size, core2, 0.7, 200);
      console.log('[hair] 고른 사진 머리카락: 정밀 분할 대신 인물 분할로', dist ? '성공' : '실패');
    }
  }
  refHair = { src, dist, bands };
  return refHair.dist;
}
// ---- 사진 모드 타투: 고른 사진의 팔 타투를 내 팔 같은 자리로(still-tattoo.ts) ----
/** user: ✂️ 로 직접 고른 영역(그 부분만 옮기고 팔 전체로 넓히지 않는다) */
type RefTat = { img: ImageBitmap | HTMLImageElement; w: number; h: number; design: TattooDesign; rect: { x: number; y: number; w: number; h: number }; src: InkSource; user?: boolean };
let refTat: RefTat | null = null as RefTat | null;
type TatPlan =
  | { dst: Arm; skin: Float32Array; ink: InkMap; src: Arm | ((lambda: number, s: number) => { x: number; y: number } | null); torso?: undefined }
  | { torso: { toRef: number[]; fromRef: number[] }; skin: Float32Array; ink: InkMap };
/** 팔 대 팔로 못 옮긴 까닭(사용자 안내) */
let tatNote = '';
let tatPlan: { ref: RefTat; mine: ImageBitmap; p: Promise<TatPlan | null> } | null = null;
/** 사진에 구운 팔 타투가 있으면 실시간 타투는 끈다 */
function tatBaked(): boolean {
  if (!PHOTO_MODE || swapMode) return false;
  if (placeBaked && placeBaked === placeKey()) return true;
  return !!(refTat && tattoo.design === refTat.design && tatPlan?.ref === refTat && tatPlan.mine === photoOriginal && tatPlanOk);
}
// ---- 사진 모드: 고른 자리에 도안 굽기(✂️ 로 고른 도안·도안 시트·팔 대 팔이 안 되는 사진) ----
// 실시간 타투는 가늘거나 굽은 팔에서 피부 판정이 거칠어 도안이 잘리거나 안 보이기도 해서, 사진은 정밀 몸 피부로 굽는다
let placeBaked = '';
const designIds = new WeakMap<object, number>();
let designSeq = 0;
function placeKey(): string {
  const d = tattoo.design;
  if (!d) return '';
  if (!designIds.has(d)) designIds.set(d, ++designSeq);
  return `${designIds.get(d)}|${tattoo.place}|${tattoo.size.toFixed(3)}|${tattoo.along ?? '-'}|${tattoo.across ?? '-'}|${strength.toFixed(2)}`;
}
let myBody: { mine: ImageBitmap; p: Promise<{ pose: { p: { x: number; y: number }[]; vis: number[] }; skin: Float32Array } | null> } | null = null;
function myBodyOf(): Promise<{ pose: { p: { x: number; y: number }[]; vis: number[] }; skin: Float32Array } | null> {
  const mine = photoOriginal!;
  if (myBody?.mine !== mine) {
    myBody = {
      mine,
      p: (async () => {
        photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
        const mc = stillCanvas();
        const pose = await photoAnalyzer.pose(mc);
        const skin = await photoAnalyzer.bodySkin(mc);
        return pose && skin ? { pose, skin } : null;
      })().catch(() => null),
    };
  }
  return myBody.p;
}
/** 도안을 내 팔·다리·가슴의 고른 자리(📍·크기)에 놓는 대응. 목이나 팔이 안 보이면 null */
async function placeTattooPlan(): Promise<TatPlan | null> {
  const d = tattoo.design;
  if (d?.canvas && tattoo.place === 'chest') return chestPlacePlan(d.canvas, d.aspect);
  const m = /^(forearm|upperArm|thigh|calf)(L|R)$/.exec(tattoo.place);
  if (!d?.canvas || !m) return null;
  const body = await myBodyOf();
  if (!body) return null;
  const legs = m[1] === 'thigh' || m[1] === 'calf';
  const a = armsOf(body.pose, body.skin, engine.source.width, engine.source.height, legs)[m[2] === 'L' ? 0 : 1];
  if (!a.seen || a.vis < 0.4) {
    // 관절점은 있는데 피부가 안 보이면(긴소매·바지) 굽어도 안 보인다: 아무 변화 없이 끝나 무엇이 잘못됐는지 몰랐다
    if (a.vis >= 0.4 && !a.seen) tatNote = `고른 타투 자리(${legs ? '다리' : '팔'})가 내 사진에선 옷에 가려 보이지 않아요. 📍 자리 고르기로 보이는 피부에 놓아 보세요`;
    return null;
  }
  const arm = a.arm;
  const upLen = Math.hypot(arm.e.x - arm.s.x, arm.e.y - arm.s.y), loLen = Math.hypot(arm.w.x - arm.e.x, arm.w.y - arm.e.y);
  const first = m[1] === 'upperArm' || m[1] === 'thigh';
  // 위팔·허벅지는 어깨·엉덩이 관절 조금 아래부터(실시간 타투와 같은 자리)
  const segStart = first ? upLen * 0.12 : upLen, segLen = first ? upLen * 0.88 : loLen;
  const r = first ? arm.rUp : arm.rLo;
  // 실시간 타투(tattooMesh)와 같은 크기 규칙: 세로 = 부위 길이의 25~85%, 가로는 둘레 절반을 넘지 않게
  const rotate = d.aspect > 1.3;
  const aspect = rotate ? 1 / d.aspect : d.aspect;
  let h = segLen * (0.25 + 0.6 * tattoo.size), w = h * aspect;
  const maxW = Math.PI * r * 0.9;
  if (w > maxW) { h *= maxW / w; w = maxW; }
  const lc = segStart + segLen * Math.min(0.92, Math.max(0.08, tattoo.along ?? 0.5));
  const th0 = Math.asin(Math.min(0.95, Math.max(-0.95, tattoo.across ?? 0)));
  const c = d.canvas;
  const rgba = c.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, c.width, c.height).data;
  const ink: InkMap = { rgba, w: c.width, h: c.height, rect: { x: 0, y: 0, w: c.width, h: c.height } };
  const src = (l: number, sg: number) => {
    // 원기둥 둘레 길이(앞에서 본 옆 거리 s → 각도), 도안 오른쪽은 화면 -n 쪽(실시간 타투와 같은 방향)
    const arc = r * (Math.asin(Math.max(-1, Math.min(1, sg))) - th0);
    const u = 0.5 - arc / w, v = (l - lc) / h + 0.5;
    const X = rotate ? v : u, Y = rotate ? 1 - u : v;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    return { x: X * c.width, y: Y * c.height };
  };
  return { dst: arm, skin: legs ? legSkin(body.pose, body.skin, engine.source.width, engine.source.height) : body.skin, ink, src };
}

/** 가슴 자리: 어깨선 아래 평면에 도안을 놓아 정밀 몸 피부로 굽는다(실시간 가슴 타투는 어깨 폭의 14% 크기 격자라 아주 작고 가늘게만 보였다:
 * 고른 사진에서 관절점을 못 찾는 가슴 클로즈업 도안은 이 길밖에 없다). 실시간과 같은 자리·방향, 세로 기준 길이만 어깨 폭 45% */
async function chestPlacePlan(c: HTMLCanvasElement, aspect0: number): Promise<TatPlan | null> {
  const body = await myBodyOf();
  if (!body) return null;
  const P = body.pose.p, V = body.pose.vis;
  if (Math.min(V[11], V[12]) < 0.5) return null;
  const l = P[11], r = P[12];
  const sw = Math.hypot(l.x - r.x, l.y - r.y);
  if (sw < 8) return null;
  const W = engine.source.width, H = engine.source.height;
  // 어깨선에 수직인 아래 방향, 어깨선 방향(왼→오른)
  let down = { x: -(l.y - r.y) / sw, y: (l.x - r.x) / sw };
  if (down.y < 0) down = { x: -down.x, y: -down.y };
  const right = { x: -down.y, y: down.x };
  const mid = { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
  const segStart = sw * 0.05, segLen = sw * 0.45, limbW = sw * 0.72;
  let h = segLen * (0.25 + 0.6 * tattoo.size), w = h * aspect0;
  if (w > limbW) { h *= limbW / w; w = limbW; }
  const oc = Math.min(0.95, Math.max(-0.95, tattoo.across ?? 0)) * (limbW / 2);
  const sx = c.width / w, sy = c.height / h;
  // 세로 자리 along 에 놓았을 때의 변환과 도안 자리의 내 피부 비율(10×10 격자)
  const planAt = (along: number) => {
    const lc = segStart + segLen * Math.min(0.92, Math.max(0.08, along));
    const center = { x: mid.x + down.x * lc + right.x * oc, y: mid.y + down.y * lc + right.y * oc };
    // 내 사진 px → 도안 px: u = 0.5 + ((p − 가운데)·right)/w, v = 0.5 + ((p − 가운데)·down)/h
    const toRef = [right.x * sx, right.y * sx, c.width * 0.5 - (center.x * right.x + center.y * right.y) * sx, down.x * sy, down.y * sy, c.height * 0.5 - (center.x * down.x + center.y * down.y) * sy];
    const det = toRef[0] * toRef[4] - toRef[1] * toRef[3];
    if (Math.abs(det) < 1e-9) return null;
    const fromRef = [toRef[4] / det, -toRef[1] / det, (toRef[1] * toRef[5] - toRef[2] * toRef[4]) / det, -toRef[3] / det, toRef[0] / det, (toRef[2] * toRef[3] - toRef[0] * toRef[5]) / det];
    let on = 0, all = 0;
    for (let v = 0.05; v < 1; v += 0.1) for (let u = 0.05; u < 1; u += 0.1) {
      const X = Math.round(fromRef[0] * u * c.width + fromRef[1] * v * c.height + fromRef[2]), Y = Math.round(fromRef[3] * u * c.width + fromRef[4] * v * c.height + fromRef[5]);
      all++;
      if (X >= 0 && Y >= 0 && X < W && Y < H && body.skin[Y * W + X] > 0.5) on++;
    }
    return { center, toRef, fromRef, skinFrac: on / all };
  };
  // 세로 자리를 고르지 않았으면 피부가 가장 많이 보이는 자리로(기본 0.5 에 가까운 쪽 우선):
  // 브이넥 사진(01992)은 가슴 피부가 아래로 갈수록 좁아져 기본 자리에선 도안 아랫부분이 옷 밑에 숨었다
  let best = planAt(tattoo.along ?? 0.5);
  if (!best) return null;
  if (tattoo.along == null) {
    for (let a = 0.08; a <= 0.92; a += 0.06) {
      const q = planAt(a);
      if (q && q.skinFrac > best.skinFrac + 0.02) best = q;
    }
  }
  const { center, toRef, fromRef, skinFrac } = best;
  dbgTat({ chestPlace: { sw, w, h, center, skinFrac } });
  if (skinFrac < 0.15) {
    tatNote = '고른 타투 자리(가슴)가 내 사진에선 옷에 가려 보이지 않아요. 📍 자리 고르기로 보이는 피부에 놓아 보세요';
    return null;
  }
  const rgba = c.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, c.width, c.height).data;
  const ink: InkMap = { rgba, w: c.width, h: c.height, rect: { x: 0, y: 0, w: c.width, h: c.height } };
  // 몸통 도안은 손에는 굽지 않는다(허리에 얹은 손등)
  return { torso: { toRef, fromRef }, skin: legSkin(body.pose, body.skin, W, H, true), ink };
}

/** 내 사진에서 가슴(어깨 사이 아래)이 옷에 가려 있으면 미리 보기할 팔 자리, 보이면 null */
async function coveredTorsoArm(): Promise<'forearmL' | 'forearmR' | null> {
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  const mc = stillCanvas();
  const mp = await photoAnalyzer.pose(mc);
  const ms = await photoAnalyzer.bodySkin(mc);
  if (!mp || !ms || Math.min(mp.vis[11], mp.vis[12]) < 0.5) return null;
  const A = mp.p[11], B = mp.p[12];
  const sw = Math.hypot(A.x - B.x, A.y - B.y);
  const c = { x: (A.x + B.x) / 2, y: (A.y + B.y) / 2 + sw * 0.25 };
  let on = 0, all = 0;
  for (let dy = -0.15; dy <= 0.15; dy += 0.05) for (let dx = -0.25; dx <= 0.25; dx += 0.05) {
    const X = Math.round(c.x + dx * sw), Y = Math.round(c.y + dy * sw);
    if (X < 0 || Y < 0 || X >= mc.width || Y >= mc.height) continue;
    all++;
    if (ms[Y * mc.width + X] > 0.5) on++;
  }
  if (!all || on > all * 0.3) return null;
  const arms = armsOf(mp, ms, mc.width, mc.height);
  const k = arms[0].vis >= arms[1].vis ? 0 : 1;
  return arms[k].seen ? (k === 0 ? 'forearmL' : 'forearmR') : null;
}
let tatPlanOk = false;
const TAT_LOOKING = '타투가 팔 어디에 있는지 보는 중…';
/** 다리 타투용 피부: 두 팔·손 자리를 뺀다(다리 앞에 놓인 손·손목에 다리 타투가 칠해졌음) */
function legSkin(pose: { p: { x: number; y: number }[]; vis: number[] }, skin: Float32Array, W: number, H: number, handsOnly = false): Float32Array {
  const caps: { a: { x: number; y: number }; b: { x: number; y: number }; r: number }[] = [];
  armsOf(pose, skin, W, H).forEach(({ arm }, k) => {
    const P = pose.p;
    if (!handsOnly) caps.push({ a: arm.s, b: arm.e, r: arm.rUp * 1.15 }, { a: arm.e, b: arm.w, r: arm.rLo * 1.15 });
    // 손: 새끼·검지 뿌리 가운데 쪽으로 손가락 끝까지(손 점이 안 보이면 아래팔 방향으로)
    const [pk, ix] = k === 0 ? [17, 19] : [18, 20];
    const handOk = Math.min(pose.vis[pk] ?? 0, pose.vis[ix] ?? 0) > 0.3;
    const dir = handOk
      ? { x: (P[pk].x + P[ix].x) / 2 - arm.w.x, y: (P[pk].y + P[ix].y) / 2 - arm.w.y }
      : { x: (arm.w.x - arm.e.x) * 0.35, y: (arm.w.y - arm.e.y) * 0.35 };
    caps.push({ a: arm.w, b: { x: arm.w.x + dir.x * 2, y: arm.w.y + dir.y * 2 }, r: arm.rLo * 1.6 });
  });
  return maskOutLimbs(skin, W, H, caps);
}

/** 사진 관절점 → 팔(왼·오른, 사람 기준) */
function armsOf(pose: { p: { x: number; y: number }[]; vis: number[] }, skin: Float32Array, W: number, H: number, legs = false): { arm: Arm; vis: number; seen: boolean }[] {
  // 다리도 같은 3점 사슬(엉덩이·무릎·발목)로 본다
  return (legs ? [[23, 25, 27], [24, 26, 28]] : [[11, 13, 15], [12, 14, 16]]).map(([s, e, w]) => {
    const P = pose.p;
    const up = Math.hypot(P[e].x - P[s].x, P[e].y - P[s].y), lo = Math.hypot(P[w].x - P[e].x, P[w].y - P[e].y);
    const ru = measureHalfWidth(skin, W, H, P[s], P[e]), rl = measureHalfWidth(skin, W, H, P[e], P[w]);
    return {
      arm: { s: P[s], e: P[e], w: P[w], rUp: ru ?? up * 0.2, rLo: rl ?? lo * 0.16 },
      vis: Math.min(pose.vis[s], pose.vis[e]),
      seen: ru !== null || rl !== null,
    };
  });
}
/** 타투 옮기기 단계 표시(큰 사진에서 어디서 오래 걸리는지 콘솔로 보려고) */
function tatStep(name: string): void {
  dbgTat({ step: name });
  console.debug('[tat]', name, Math.round(performance.now()));
}
function dbgTat(o: object): void {
  const w = window as unknown as { irisTat?: object };
  w.irisTat = { ...w.irisTat, ...o };
}
async function armTattooPlan(): Promise<TatPlan | null> {
  const ref = refTat, mine = photoOriginal;
  if (!ref || !mine) return null;
  if (tatPlan?.ref !== ref || tatPlan.mine !== mine) {
    tatPlanOk = false;
    tatNote = '';
    const p = (async (): Promise<TatPlan | null> => {
      photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
      setStatus(TAT_LOOKING);
      tatStep('refPose');
      // 고른 사진: 도안 뽑기와 같은 크기로
      const rc = document.createElement('canvas');
      rc.width = ref.w;
      rc.height = ref.h;
      rc.getContext('2d')!.drawImage(ref.img, 0, 0, ref.w, ref.h);
      const rp = await photoAnalyzer.pose(rc);
      tatStep('refSkin');
      const rs = await photoAnalyzer.bodySkin(rc);
      dbgTat({ refPose: !!rp, refSkin: !!rs });
      tatStep('refFace');
      if (!rs) return null;
      // 고른 사진 얼굴(윤곽 15% 넓힘)은 도안이 아니다: 팔을 얼굴까지 든 셀카에서 눈·눈썹·콧구멍·입술이 잉크로 뽑혀
      // 내 아래팔에 검은 조각으로 찍혔다 → 잉크에서만 뺀다(팔 영역(limbRegion)에서도 빼면 얼굴에 붙은 팔 띠가 끊겨 팔로 못 읽었다)
      const refFacePts = await photoAnalyzer.faceLandmarks(rc).catch(() => null);
      tatStep('ink');
      let faceM: Float32Array | null = null;
      if (refFacePts) {
        const ov = FACE_OVAL.map((i) => refFacePts[i]);
        const cx = ov.reduce((t, q) => t + q.x, 0) / ov.length, cy = ov.reduce((t, q) => t + q.y, 0) / ov.length;
        faceM = fillPoly(ov.map((q) => ({ x: cx + (q.x - cx) * 1.15, y: cy + (q.y - cy) * 1.15 })), ref.w, ref.h);
      } else if (rp && rp.vis[0] > 0.5) {
        // 얼굴 점을 못 찾으면(이마가 사진 위로 잘린 셀카) 관절점으로: 코를 가운데, 귀 사이 거리(없으면 눈 사이 2.2배)를 가로 반지름으로 한 타원
        const P = rp.p, nose = P[0];
        const ears = rp.vis[7] > 0.5 && rp.vis[8] > 0.5 ? Math.hypot(P[7].x - P[8].x, P[7].y - P[8].y) / 2 : 0;
        const eyes = rp.vis[2] > 0.5 && rp.vis[5] > 0.5 ? Math.hypot(P[2].x - P[5].x, P[2].y - P[5].y) * 1.1 : 0;
        const rx = Math.max(ears, eyes) * 1.15;
        if (rx > 4) {
          const pts: { x: number; y: number }[] = [];
          for (let k = 0; k < 36; k++) pts.push({ x: nose.x + rx * Math.cos((k / 36) * Math.PI * 2), y: nose.y + rx * 1.4 * Math.sin((k / 36) * Math.PI * 2) });
          faceM = fillPoly(pts, ref.w, ref.h);
        }
      }
      // 얼굴 점도 관절점도 못 찾을 때가 있다(같은 셀카도 실행마다 다름 → 얼굴이 그대로 잉크로 남아 내 손목에 입술·콧구멍 조각):
      // 정밀 분할(BiSeNet)의 얼굴 채널(피부·눈·눈썹·코·입)로도 찾아 합친다. 사진 전체를 정사각형(검은 여백)으로 넣는다
      let faceParse = 0;
      try {
        const size = Math.max(ref.w, ref.h), x0 = (ref.w - size) / 2, y0 = (ref.h - size) / 2;
        const fp = await parseFace(rc, { x: x0, y: y0, size });
        const m = new Float32Array(ref.w * ref.h);
        for (let y = 0; y < ref.h; y++) for (let x = 0; x < ref.w; x++) {
          const u = Math.floor(((x - x0) / size) * fp.size), v = Math.floor(((y - y0) / size) * fp.size);
          if (fp.face[v * fp.size + u] > 128) { m[y * ref.w + x] = 1; faceParse++; }
        }
        // 작은 조각(사진의 0.2% 미만: 뒤에 선 다른 사람·오검출)은 무시, 둘레 2% 넓혀 눈썹·입술 가장자리까지
        if (faceParse >= ref.w * ref.h * 0.002) {
          const bl = boxBlur(m, ref.w, ref.h, Math.max(2, Math.round(size * 0.02)));
          const fm = faceM ?? new Float32Array(ref.w * ref.h);
          for (let i = 0; i < fm.length; i++) if (bl[i] > 0.01) fm[i] = 1;
          faceM = fm;
        }
      } catch {
        // 분할 모델을 못 불러오면 얼굴 점·관절점 타원만
      }
      dbgTat({ refFacePts: !!refFacePts, faceM: !!faceM, faceParse });
      const eraseFace = (im: InkMap): void => {
        if (!faceM) return;
        for (let y = 0; y < im.h; y++) for (let x = 0; x < im.w; x++) {
          const X = Math.round(im.rect.x + ((x + 0.5) / im.w) * im.rect.w), Y = Math.round(im.rect.y + ((y + 0.5) / im.h) * im.rect.h);
          if (X >= 0 && Y >= 0 && X < ref.w && Y < ref.h && faceM[Y * ref.w + X] > 0) im.rgba[(y * im.w + x) * 4 + 3] = 0;
        }
      };
      const dc = ref.design.canvas!;
      let ink: InkMap = { rgba: new Uint8ClampedArray(dc.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, dc.width, dc.height).data), w: dc.width, h: dc.height, rect: ref.rect };
      eraseFace(ink);
      let total = 0;
      for (let i = 3; i < ink.rgba.length; i += 4) if (ink.rgba[i] >= 32) total += ink.rgba[i] / 255;
      // 몸통 타투(등·가슴): 잉크가 고른 사진 어깨~엉덩이 안에 절반 넘게 있으면 어깨·엉덩이 대응으로 옮긴다
      const torsoPts = (pp: { p: { x: number; y: number }[]; vis: number[] }) => {
        const P = pp.p, sw = Math.hypot(P[11].x - P[12].x, P[11].y - P[12].y);
        const mid = { x: (P[11].x + P[12].x) / 2, y: (P[11].y + P[12].y) / 2 };
        // 엉덩이가 안 보이면 어깨 가운데에서 어깨선에 수직으로 어깨 폭만큼 아래
        const hipsOk = Math.min(pp.vis[23], pp.vis[24]) > 0.5;
        let down = { x: -(P[12].y - P[11].y) / sw, y: (P[12].x - P[11].x) / sw };
        if (down.y < 0) down = { x: -down.x, y: -down.y };
        const third = hipsOk ? { x: (P[23].x + P[24].x) / 2, y: (P[23].y + P[24].y) / 2 } : { x: mid.x + down.x * sw * 1.3, y: mid.y + down.y * sw * 1.3 };
        return { pts: [P[11], P[12], third], hipsOk, sw };
      };
      if (rp && Math.min(rp.vis[11], rp.vis[12]) > 0.5) {
        const tp = torsoPts(rp);
        const poly = [rp.p[11], rp.p[12], tp.pts[2]];
        // 어깨·엉덩이 사다리꼴(어깨 폭 15% 넓힘) 안 잉크
        const c = { x: (poly[0].x + poly[1].x + poly[2].x) / 3, y: (poly[0].y + poly[1].y + poly[2].y) / 3 };
        const P4 = [rp.p[12], rp.p[11], { x: tp.pts[2].x + (rp.p[11].x - rp.p[12].x) / 2, y: tp.pts[2].y + (rp.p[11].y - rp.p[12].y) / 2 }, { x: tp.pts[2].x - (rp.p[11].x - rp.p[12].x) / 2, y: tp.pts[2].y - (rp.p[11].y - rp.p[12].y) / 2 }]
          .map((q) => ({ x: c.x + (q.x - c.x) * 1.15, y: c.y + (q.y - c.y) * 1.15 }));
        const inside = (x: number, y: number) => {
          let ins = false;
          for (let i = 0, j = 3; i < 4; j = i++) {
            const a = P4[i], b = P4[j];
            if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) ins = !ins;
          }
          return ins;
        };
        let inT = 0;
        for (let y = 0; y < ink.h; y++) for (let x = 0; x < ink.w; x++) {
          const a = ink.rgba[(y * ink.w + x) * 4 + 3];
          if (a < 32) continue;
          if (inside(ref.rect.x + ((x + 0.5) / ink.w) * ref.rect.w, ref.rect.y + ((y + 0.5) / ink.h) * ref.rect.h)) inT += a / 255;
        }
        dbgTat({ torsoFrac: inT / Math.max(1, total), refNose: rp.vis[0], refSh: [rp.p[11], rp.p[12], rp.vis[11], rp.vis[12], tp.hipsOk], inkRect: ref.rect, P4 });
        // 팔꿈치를 둘 다 못 잡았으면(팔을 얼굴까지 든 셀카 등) 관절점이 팔을 놓쳐 어깨 사다리꼴이 팔을 덮는다:
        // 잉크가 있는 피부 덩어리가 팔처럼 길쭉하면(길이 ≥ 굵기 3배) 몸통이 아니라 팔로 본다(배·손등에 소매 조각이 찍혔다)
        let armLike = false;
        if (Math.max(rp.vis[13], rp.vis[14]) < 0.5 && !ref.user) {
          const found = findInk(ref.src);
          const m = new Float32Array(ref.w * ref.h);
          const av = ref.src.avoid;
          for (let i = 0; i < m.length; i++) m[i] = Math.max(rs[i], ref.src.skin[i], found.ink[i], found.rimLine[i], av ? 1 - av[i] : 0);
          const limb = limbRegion(m, ref.w, ref.h, ref.rect);
          const st = limb ? armStrip(limb.mask, ref.w, ref.h, limb.box) : null;
          armLike = !!st && st.len / (2 * st.mean) >= 3;
          dbgTat({ armLike, armLen: st ? +(st.len / (2 * st.mean)).toFixed(2) : null });
        }
        if (inT > total * 0.5 && !armLike) {
          const mc = stillCanvas();
          tatStep('myPose');
          const mp = await photoAnalyzer.pose(mc);
          tatStep('mySkin');
          const ms = await photoAnalyzer.bodySkin(mc);
          tatStep('torso');
          if (!mp || !ms || Math.min(mp.vis[11], mp.vis[12]) < 0.5) return null;
          // 뒷모습(얼굴이 안 보임)은 관절점이 왼쪽·오른쪽·크기를 자주 틀려 옮기지 않는다: 앞모습끼리(가슴·쇄골)만
          const refFace = refFacePts;
          const myFace = await photoAnalyzer.faceLandmarks(mc);
          // 얼굴이 사진 위로 잘린 앞모습(가슴·팔 타투 사진에 흔함)도 있다: 얼굴이 없어도 사람 왼어깨(11)가 그림 오른쪽이면 앞모습
          const sw0 = Math.abs(rp.p[11].x - rp.p[12].x);
          const refFront = !!refFace || rp.p[11].x - rp.p[12].x > sw0 * 0.3;
          dbgTat({ refFace: !!refFace, refFront, myFace: !!myFace });
          if (!refFront || !myFace) {
            tatNote = !refFront ? '등 타투예요. 등 자리로 옮기기는 아직 안 돼서 팔에 미리 보여 드려요(📍 자리 고르기로 옮길 수 있어요)' : '내 사진에서 얼굴·어깨가 보여야 가슴 타투를 그 자리에 옮길 수 있어요(지금은 팔에 미리 보기)';
            return null;
          }
          // 어깨를 그림 왼쪽→오른쪽으로 정렬해 크기·회전·위치만 맞춘다(도안이 늘어나지 않게)
          const frame = (pp: { p: { x: number; y: number }[] }) => {
            const [A, B] = [pp.p[11], pp.p[12]].sort((u, v) => u.x - v.x);
            return [A, B, { x: A.x - (B.y - A.y), y: A.y + (B.x - A.x) }];
          };
          const a = frame(mp), b = frame(rp);
          const toRef = affine3(a, b), fromRef = affine3(b, a);
          if (!toRef || !fromRef) return null;
          // 옮길 자리가 내 사진에선 옷에 가려 있으면(티셔츠 입은 가슴) 굽어도 안 보인다
          let torsoVis = 1;
          {
            let on = 0, all = 0;
            for (let y = 0; y < ink.h; y += 2) for (let x = 0; x < ink.w; x += 2) {
              const al = ink.rgba[(y * ink.w + x) * 4 + 3] / 255;
              if (al < 0.15) continue;
              const rx = ref.rect.x + ((x + 0.5) / ink.w) * ref.rect.w, ry = ref.rect.y + ((y + 0.5) / ink.h) * ref.rect.h;
              const X = Math.round(fromRef[0] * rx + fromRef[1] * ry + fromRef[2]), Y = Math.round(fromRef[3] * rx + fromRef[4] * ry + fromRef[5]);
              all += al;
              if (X >= 0 && Y >= 0 && X < mc.width && Y < mc.height && ms[Y * mc.width + X] > 0.5) on += al;
            }
            torsoVis = all ? on / all : 1;
            dbgTat({ torsoVisible: torsoVis });
          }
          // 가려 있으면 몸통은 건너뛰고 아래 팔 대 팔로(가슴·팔 같이 있는 사진의 소매는 옮길 수 있다)
          // 몸통 도안은 손에는 굽지 않는다(허리에 얹은 손등에 글씨가 찍혔다)
          if (torsoVis >= 0.25) return { torso: { toRef, fromRef }, skin: legSkin(mp, ms, mc.width, mc.height, true), ink };
        }
      }
      // 팔 점을 못 찾으면(몸이 안 나온 팔 사진) 아래 '띠' 방식으로
      const legs = /^(thigh|calf)/.test(tattoo.place);
      const rArms = rp ? armsOf(rp, rs, ref.w, ref.h, legs) : null;
      const on = rArms ? rArms.map((a) => inkOnArm(ink, a.arm)) : [{ up: 0, lo: 0 }, { up: 0, lo: 0 }];
      let side = on[0].up + on[0].lo >= on[1].up + on[1].lo ? 0 : 1;
      // 고른 사진 팔 점이 믿을 만하고 잉크가 그 팔에 절반 넘게 있으면 마디끼리 옮긴다
      const inImg = (q: { x: number; y: number }) => q.x >= 0 && q.y >= 0 && q.x < ref.w && q.y < ref.h;
      const ra = rArms?.[side];
      const poseOk = !!ra && ra.seen && ra.vis > 0.6 && inImg(ra.arm.s) && inImg(ra.arm.e) && on[side].up + on[side].lo >= total * 0.5;
      let strip: Strip | null = null;
      if (!poseOk) {
        // 가까운 팔 사진(몸이 안 나옴): 피부·잉크 영역 모양에서 팔 축·굵기를 읽는다. 팔 타투라고 읽힌 경우만
        if (!/^(forearm|upperArm|thigh|calf)/.test(tattoo.place)) return null;
        // ✂️ 로 일부만 고른 도안은 팔 어디인지 알 수 없다: 실시간 타투(📍 자리 고르기)로
        if (ref.user) return null;
        // 자동으로 고른 도안 영역은 소매 일부만일 때가 많다: 그 영역이 있는 팔(피부·잉크 덩어리) 전체로 넓혀 다시 뽑는다
        const found = findInk(ref.src);
        const m = new Float32Array(ref.w * ref.h);
        // 타투가 빽빽한 팔은 분할이 옷으로 봐 피부에서 빠진다: 도안 뽑기에서 '피부 아님'을 덜어 낸 곳(1 − avoid)도 팔로
        const av = ref.src.avoid;
        for (let i = 0; i < m.length; i++) m[i] = Math.max(rs[i], ref.src.skin[i], found.ink[i], found.rimLine[i], av ? 1 - av[i] : 0);
        const limb = limbRegion(m, ref.w, ref.h, ref.rect);
        if (!limb) return null;
        if (!ref.user) {
          const ex = extractDesign(ref.src, limb.box, 1600);
          if (ex) {
            ink = { rgba: ex.rgba, w: ex.w, h: ex.h, rect: ex.rect };
            eraseFace(ink);
          }
        }
        strip = armStrip(limb.mask, ref.w, ref.h, limb.box);
        dbgTat({ strip: strip && { len: strip.len, mean: strip.mean, ratio: strip.ratio, p: strip.p, h: [...strip.h], refW: ref.w, rect: ink.rect } });
        // 팔 모양(길이가 굵기의 2.5배 넘음)이 아니면 실시간 타투로
        if (!strip || strip.len / (2 * strip.mean) < 2.5) return null;
      }
      const mc = stillCanvas();
      tatStep('myPose2');
      const mp = await photoAnalyzer.pose(mc);
      tatStep('mySkin2');
      const ms = await photoAnalyzer.bodySkin(mc);
      tatStep('arm');
      if (!mp || !ms) return null;
      const mArms = armsOf(mp, ms, mc.width, mc.height, legs);
      const ok = (a: (typeof mArms)[number]) => a.seen && a.vis > 0.4;
      if (strip) side = mArms[0].vis >= mArms[1].vis ? 0 : 1;
      // 같은 쪽 팔이 보이면 그 팔, 아니면 반대쪽 팔(글씨가 뒤집히지 않게 거울로 뒤집지는 않는다)
      const mine2 = ok(mArms[side]) ? side : ok(mArms[1 - side]) ? 1 - side : -1;
      dbgTat({ mArms: mArms.map((a) => ({ ...a.arm, vis: a.vis, seen: a.seen })), mine2, poseOk });
      if (mine2 < 0) return null;
      const dst = mArms[mine2].arm;
      const dSkin = legs ? legSkin(mp, ms, mc.width, mc.height) : ms;
      if (!strip) return { dst, skin: dSkin, ink, src: ra!.arm };
      // 띠: 어깨에서 시작(전체 소매) 또는 아래팔 가운데(작은 도안, 굵기 비로 크기)
      const st = strip;
      const upLen = Math.hypot(dst.e.x - dst.s.x, dst.e.y - dst.s.y), loLen = Math.hypot(dst.w.x - dst.e.x, dst.w.y - dst.e.y);
      // 어깨 쪽이 뚜렷이 굵으면 어깨~손목 전체 소매로 보고 길이로 맞춘다(굵기는 근육·각도에 따라 많이 달라서)
      const full = st.ratio > 1.25;
      // 전체 소매: 손목(끝 쪽 35% 안에서 가장 가는 곳)까지를 내 어깨~손목에 맞춘다(주먹·손이 함께 찍혀도)
      let wristTau = st.len;
      if (full) {
        const K = st.h.length;
        let kmin = K - 1;
        for (let i = Math.floor(K * 0.65); i < K - 1; i++) if (st.h[i] < st.h[kmin]) kmin = i;
        wristTau = (kmin / (K - 1)) * st.len;
      }
      const k = full ? wristTau / (upLen + loLen) : st.mean / ((dst.rUp + dst.rLo) / 2);
      const L = st.len / k;
      const l0 = full ? 0 : upLen + loLen / 2 - L / 2;
      return { dst, skin: dSkin, ink, src: (l: number, sg: number) => stripPoint(st, (l - l0) * k, sg) };
    })().catch((err) => {
      console.warn('팔 타투 옮기기 실패(실시간 타투로)', err);
      return null;
    });
    tatPlan = { ref, mine, p };
    const r = await p;
    if (tatPlan?.p === p) {
      tatPlanOk = !!r;
      applyTattooToEngine();
    }
    return r;
  }
  return tatPlan.p;
}

// ---- 사진 모드 메이크업: 고른 사진 얼굴의 화장(맨 피부와의 차이)을 내 얼굴에 그대로 옮긴다(still-makeup.ts) ----
type MkPlan = { myPts: { x: number; y: number }[]; mySkin: Float32Array | null; ref: Uint8ClampedArray; rw: number; rh: number; refPts: { x: number; y: number }[]; refSkin: Float32Array | null; refHands: Float32Array | null };
let mkPlan: { mine: ImageBitmap; ref: ImageBitmap; p: Promise<MkPlan | null>; ok: boolean } | null = null;
function mkBaked(): boolean {
  return !!(PHOTO_MODE && !swapMode && mkPlan?.ok && mkPlan.mine === photoOriginal && mkPlan.ref === lastSource);
}
function mkAmounts(): MakeupAmounts {
  const has = (k: PartName) => !!(chosen[k].color && !off.has(k));
  const v = Math.min(1, strength);
  return { lip: has('lip') ? v : 0, eye: has('shadow') || has('liner') ? v : 0, skin: has('blush') || has('contour') || has('base') ? v : 0 };
}
async function makeupPlan(): Promise<MkPlan | null> {
  const mine = photoOriginal, ref = lastSource;
  if (!mine || !ref) return null;
  if (mkPlan?.mine === mine && mkPlan.ref === ref) return mkPlan.p;
  const p = (async (): Promise<MkPlan | null> => {
    photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
    const mc = stillCanvas(), rc = toCanvas(ref, 1600);
    const myPts = await photoAnalyzer.faceLandmarks(mc);
    const refPts = await photoAnalyzer.faceLandmarks(rc);
    // 얼굴 점을 못 찾으면 화장을 못 옮긴다(실시간 화장으로 떨어지지만 사진에선 대개 아무것도 안 바뀜) → 까닭을 남겨 굽기 끝에 알린다
    mkPlanNote = !myPts ? '내 사진에서 얼굴을 찾지 못해 화장을 옮길 수 없어요. 얼굴이 정면으로 크게 보이는 사진을 써 주세요'
      : !refPts ? '고른 사진의 얼굴이 작거나 옆을 보고 있어 화장을 옮길 수 없어요(헤어 등 다른 것은 그대로 입혀요)' : '';
    console.debug('[makeup] 얼굴 점', { mine: !!myPts, ref: !!refPts });
    if (!myPts || !refPts) return null;
    setStatus('화장을 얼굴 모양에 맞추는 중…');
    const mySkin = await skinOf(mc, myPts, 5);
    const refSkin = await skinOf(rc, refPts, 5);
    // 얼굴 가린 손: 몸 피부 분할 + 손 점으로 그린 손가락(얼굴 위 손가락은 분할이 얼굴로 보기도 한다)
    const body = await photoAnalyzer.bodySkin(rc).catch(() => null);
    const fingers = await photoAnalyzer.handMask(rc).catch(() => null);
    let refHands = body;
    if (body && fingers) for (let i = 0; i < body.length; i++) body[i] = Math.max(body[i], fingers[i]);
    else if (fingers) refHands = fingers;
    const rd = rc.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, rc.width, rc.height).data;
    return { myPts, mySkin, ref: rd, rw: rc.width, rh: rc.height, refPts, refSkin, refHands };
  })().catch((err) => {
    console.warn('사진 화장 옮기기 실패(실시간 화장으로)', err);
    return null;
  });
  const entry = { mine, ref, p, ok: false };
  mkPlan = entry;
  const r = await p;
  if (mkPlan === entry) {
    entry.ok = !!r;
    applyLookToEngine();
  }
  return r;
}

async function updateStillHair(): Promise<void> {
  if (!PHOTO_MODE || !photoOriginal || engine.source.kind !== 'image' || swapMode) return;
  if (stillHairBusy) {
    await stillHairBusy;
    return updateStillHair();
  }
  const orig = photoOriginal;
  const h = chosen.hair;
  // 분포를 그대로 옮기므로 진하기 1 이 '고른 사진 그대로'(분석의 amount 는 셰이더용이라 쓰지 않는다)
  const amount = h.color && !off.has('hair') && lastSource ? strength : 0;
  const nailOn = !!(chosen.nail.color && !off.has('nail') && myNails.length && refNailArt());
  const tatOn = !!(refTat && tattoo.design === refTat.design && !off.has('tattoo'));
  let plan = tatOn ? await armTattooPlan() : null;
  // 가슴·목 타투인데 내 사진에선 그 자리가 옷에 가려 있으면 굽어도 안 보인다 → 보이는 팔에 미리 보기 + 안내
  if (tatOn && !plan && /^(chest|neck)/.test(tattoo.place)) {
    const arm = await coveredTorsoArm();
    if (arm) {
      tattoo.place = arm;
      tattoo.along = tattoo.across = undefined;
      tatNote = '고른 타투 자리(몸통·목)가 내 사진에선 옷에 가려 있어요. 보이는 팔에 미리 보여 드려요(📍 자리 고르기로 옮길 수 있어요)';
      applyTattooToEngine();
    }
  }
  // 팔 대 팔이 안 되면 고른 자리에 굽는다
  const pk = placeKey();
  const placeOn = !plan && !!tattoo.design && !off.has('tattoo');
  if (placeOn) plan = await placeTattooPlan();
  const placed = placeOn && !!plan;
  const mk = mkAmounts();
  // 고른 사진에서 화장을 읽었을 때만(얼굴이 보이는 사진)
  const mkOn = !!(lastSource && lastStyle?.makeup && (mk.lip || mk.eye || mk.skin));
  const mkp = mkOn ? await makeupPlan() : null;
  const key = `${mkp ? `${mk.lip}:${mk.eye}:${mk.skin}` : 0}|${amount.toFixed(3)}|${lastSource ? 1 : 0}|${nailOn ? myNails.length + ':' + strength.toFixed(2) + ':' + nailGen : 0}|${plan ? strength.toFixed(2) : 0}|${placed ? pk : ''}`;
  // 네일을 골랐는데 내 손톱을 하나도 못 찾았으면(손이 작게 나온 전신 사진 등) 굽기 끝에 안내가 지워져 아무 일도 없는 듯 보였다
  // → 못 찾았다고 남긴다(손톱을 눌러 더하는 길도 함께). 같은 결과라 다시 굽지 않을 때도 다시 띄운다
  const nailNote = chosen.nail.color && !off.has('nail') && nailsAuto === photoOriginal && !myNails.length ? '내 사진에서 손톱을 찾지 못했어요. 손톱을 하나씩 눌러 주세요(누른 손톱에 고른 사진의 네일이 입혀져요)' : '';
  // 굽기 끝에 남길 안내: 손톱 못 찾음 > 머리카락 못 찾음(염색을 골랐을 때)
  const bakeNote = (): string => nailNote || (amount > 0 ? mineHairNote || hairSkipNote : '') || (mkOn && !mkp ? mkPlanNote : '');
  if (key === stillHairKey && refHair?.src === lastSource) {
    // 같은 결과면 다시 굽지 않지만, 타투를 못 옮긴 안내(옷에 가림 등)는 다른 안내에 지워졌을 수 있어 다시 띄운다
    if (tatOn && !plan && tatNote) setStatus(tatNote);
    else if (bakeNote()) setStatus(bakeNote());
    return;
  }
  // 작업이 곧바로 끝나는 경우(할 일 없음)에도 표시가 남지 않게, 끝난 뒤 '그 작업이면' 지운다
  // 굽는 사이(화장·타투 준비는 수십 초) 얼굴 바꾸기를 켰거나 내 사진을 바꿨으면 결과를 버린다
  const stale = () => swapMode || photoOriginal !== orig || engine.source.kind !== 'image';
  const job = (async () => {
    try {
      if (stale()) return;
      if (amount <= 0 && !nailOn && !plan && !mkp) {
        engine.source.replaceStill(photoOriginal!);
        if (placeBaked) {
          placeBaked = '';
          applyTattooToEngine();
        }
        stillHairKey = key;
        if (bakeNote()) setStatus(bakeNote());
        return;
      }
      const base = stillRGBA();
      let out = base.rgba;
      if (amount > 0) {
        const mine = await mineHairMatte();
        const ref = await refHairDist();
        // 내 머리카락이 새까맣게 찍혀 결이 없으면(밝기 상위 10% 가 L 14 아래: sam1 어두운 사진 9.9, 아프로 00035 는 28) 밝은 염색(고른 사진
        // 가운데 L 35 넘음)은 결 없는 하얀 석고 덩어리가 된다(배경과 머리를 가를 근거도 없음) → 하지 않고 이유를 알린다
        const nQ = mine ? mine.dist.L.length - 1 : 0;
        hairSkipNote = '';
        if (mine && ref && mine.dist.L[Math.round(nQ * 0.9)] < 14 && ref.L[nQ >> 1] > 35) {
          hairSkipNote = '내 사진의 머리카락이 너무 어둡게 찍혀 결이 보이지 않아, 밝은 색 염색은 자연스럽게 할 수 없어요. 밝은 곳에서 찍은 사진을 써 주세요';
          setStatus(hairSkipNote);
          console.log('[hair] 새까만 머리 + 밝은 염색: 건너뜀', mine.dist.L[Math.round(nQ * 0.9)].toFixed(1), ref.L[nQ >> 1].toFixed(1));
        } else if (!mine && mineHairNote) {
          // 내 머리카락이 거의 없으면 실시간 염색도 할 게 없다: 하지 않고 이유를 알린다
          setStatus(mineHairNote);
        } else if (!mine || !ref) {
          // 정밀 방식이 안 되면(얼굴·머리카락을 못 찾음) 실시간 염색으로
          engine.hair = { color: h.color!, amount: amt(h.amount), tip: hairTip ?? undefined };
          setStatus(mine ? '고른 사진에서 머리카락을 찾지 못해 머리색만 옮겼어요' : '');
        } else {
          setStatus('머리카락 색을 올마다 옮기는 중…');
          // 투톤·옴브레 사진이면 뿌리→끝 띠마다 옮긴다
          const rb = refHair?.bands;
          const span = rb?.ombre ? hairSpan(mine.alpha, mine.w) : null;
          out = recolorHair(mine.rgba, mine.w, mine.h, mine.alpha, mine.dist, ref, Math.min(1, amount), 1, rb && span ? { ref: rb, ...span } : null, rb?.lowSpread ?? null);
        }
      }
      if (mkp) {
        setStatus('고른 사진 화장을 옮기는 중…');
        out = transferMakeup(out, base.w, base.h, mkp.myPts, mkp.ref, mkp.rw, mkp.rh, mkp.refPts, FACE_OVAL, mk, mkp.mySkin, mkp.refSkin, mkp.refHands, ((window as unknown as { irisMk: Record<string, unknown> }).irisMk = {}));
      }
      if (nailOn) {
        out = new Uint8ClampedArray(out);
        const art = refNailArt()!;
        myNails.forEach((n, i) => {
          const r = art.nails[i % art.nails.length];
          transferNail(out, base.w, base.h, n.mask, n.frame, art.rgba, art.w, art.h, r.frame, 0.35, Math.min(1, strength));
        });
      }
      if (plan) {
        setStatus('타투를 팔에 옮기는 중…');
        out = plan.torso
          ? transferTorsoInk(out, base.w, base.h, plan.skin, plan.ink, plan.torso.toRef, plan.torso.fromRef, Math.min(1, strength))
          : transferArmInk(out, base.w, base.h, plan.dst, plan.skin, plan.ink, plan.src, false, Math.min(1, strength));
      }
      const bmp = await createImageBitmap(new ImageData(out as unknown as Uint8ClampedArray<ArrayBuffer>, base.w, base.h));
      if (stale()) return;
      engine.source.replaceStill(bmp);
      // 고른 자리에 구웠으면 실시간 타투는 끈다(안 구웠으면 다시 켠다)
      const pb = placed ? pk : '';
      if (pb !== placeBaked) {
        placeBaked = pb;
        applyTattooToEngine();
      }
      stillHairKey = key;
      setStatus(bakeNote());
    } catch (err) {
      console.warn('사진 염색 실패', err);
      setStatus('');
    }
  })();
  stillHairBusy = job;
  try {
    await job;
  } finally {
    if (stillHairBusy === job) stillHairBusy = null;
  }
  // 몸통 타투를 반대쪽(앞/뒤) 사진에 넣으려던 경우 안내
  if (tatOn && (placed || !plan) && tatNote) setStatus(tatNote);
  // 굽기 계획 없이 끝났으면(고른 사진에서 몸·팔을 못 찾음 → 실시간 타투) '보는 중' 안내를 남기지 않는다
  else if (statusEl.textContent === TAT_LOOKING) setStatus('');
}

/** 처리 해상도의 원본 사진 RGBA(한 번만) */
let stillBase: { rgba: Uint8ClampedArray; w: number; h: number; src: ImageBitmap } | null = null;
function stillRGBA(): { rgba: Uint8ClampedArray; w: number; h: number } {
  const orig = photoOriginal!;
  if (stillBase?.src === orig) return stillBase;
  const W = engine.source.width, H = engine.source.height;
  const cv = document.createElement('canvas');
  cv.width = W;
  cv.height = H;
  const cx = cv.getContext('2d', { willReadFrequently: true })!;
  cx.drawImage(orig, 0, 0, W, H);
  stillBase = { rgba: cx.getImageData(0, 0, W, H).data, w: W, h: H, src: orig };
  return stillBase;
}

// ---- 사진 모드 네일: 손톱을 SAM 으로 오려 내고, 고른 사진의 손톱 그림을 옮겨 그린다(still-nail.ts) ----
let myNails: { mask: Uint8Array; frame: NailFrame }[] = [];
/** 손톱이 바뀔 때마다 늘려 다시 굽게 */
let nailGen = 0;
let mySam: { src: ImageBitmap; p: Promise<SamImage> } | null = null;
type NailArt = { src: ImageBitmap; rgba: Uint8ClampedArray; w: number; h: number; nails: { mask: Uint8Array; frame: NailFrame }[] };
let refNails: NailArt | null = null as NailArt | null;
let refNailsBusy: Promise<void> | null = null;
let nailsAuto: ImageBitmap | null = null;
/** 고른 사진 손톱들을 좋은 순으로: 또렷한(라플라시안 분산) 것 먼저, 초점이 안 맞아 흐리거나(가장 또렷한 것의 30% 아래)
 * 마스크 안에 어두운 배경이 많이 섞인(아주 어두운 픽셀 8% 넘게) 손톱은 다른 손톱이 둘 이상 있으면 뺀다.
 * (파란 옴브레 09번: 맨 왼쪽의 초점 나간 손톱이 첫 손톱으로 쓰여 내 엄지에 흐릿한 파란 덩어리 + 검은 조각이 찍혔다) */
function rankNails(rgba: Uint8ClampedArray, w: number, h: number, nails: { mask: Uint8Array; frame: NailFrame }[]): { mask: Uint8Array; frame: NailFrame }[] {
  if (nails.length < 2) return nails;
  const lum = (i: number): number => 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  const q = nails.map((n) => {
    let s = 0, s2 = 0, cnt = 0, dark = 0, all = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!n.mask[i]) continue;
      all++;
      if (lum(i) < 40) dark++;
      if (!n.mask[i - 1] || !n.mask[i + 1] || !n.mask[i - w] || !n.mask[i + w]) continue;
      const lp = 4 * lum(i) - lum(i - 1) - lum(i + 1) - lum(i - w) - lum(i + w);
      s += lp; s2 += lp * lp; cnt++;
    }
    const sharp = cnt ? s2 / cnt - (s / cnt) ** 2 : 0;
    return { n, sharp, dark: all ? dark / all : 1 };
  });
  const best = Math.max(...q.map((v) => v.sharp));
  const good = q.filter((v) => v.sharp >= best * 0.3 && v.dark <= 0.08);
  const use = good.length >= 2 ? good : q;
  use.sort((a, b) => b.sharp - a.sharp);
  if (use.length < q.length) console.log('[nail] 고른 사진 손톱 중 흐리거나 배경 섞인 것 뺌', q.length, '→', use.length, q.map((v) => ({ sharp: +v.sharp.toFixed(0), dark: +v.dark.toFixed(2) })));
  return use.map((v) => v.n);
}
function refNailArt(): NailArt | null {
  if (refNails && refNails.src === lastSource && refNails.nails.length) return refNails;
  // 손톱 모양을 못 찾았으면(색만 앎) 그 색의 가상 손톱 그림으로 굽는다: 실시간 네일은 정지 사진에서 어긋나 보인다
  const c = chosen.nail.color;
  if (!c || !lastSource || (refNailsBusy && !refNails)) return null;
  const key = c.join(',');
  if (synthNail?.key !== key) {
    const g = (v: number) => Math.round(255 * Math.pow(Math.max(0, Math.min(1, v)), 1 / 2.2));
    const sn = syntheticNail([g(c[0]), g(c[1]), g(c[2])]);
    synthNail = { key, art: { src: lastSource, rgba: sn.rgba, w: sn.w, h: sn.h, nails: [{ mask: sn.mask, frame: sn.frame }] } };
  }
  return synthNail.art;
}
let synthNail: { key: string; art: NailArt } | null = null as { key: string; art: NailArt } | null;
function stillCanvas(): HTMLCanvasElement {
  const b = stillRGBA();
  const cv = document.createElement('canvas');
  cv.width = b.w;
  cv.height = b.h;
  cv.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(b.rgba), b.w, b.h), 0, 0);
  return cv;
}
function mySamImage(): Promise<SamImage> {
  const orig = photoOriginal!;
  if (mySam?.src !== orig) mySam = { src: orig, p: samImage(stillCanvas()) };
  return mySam.p;
}
/**
 * 작은 손(전신 사진)용 SAM: 손 둘레만 잘라 키워서 오린다. 사진 전체로는 7px 남짓한 손톱이 SAM 특징 한 칸(16px)보다 작아
 * 손 전체가 잡혔다. 마스크는 원래 사진 크기로 되돌려 준다
 */
async function handSamImage(hp: { p: { x: number; y: number }[] }): Promise<SamImage> {
  const src = stillCanvas();
  const W = src.width, H = src.height;
  const xs = hp.p.map((q) => q.x), ys = hp.p.map((q) => q.y);
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
  const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * 1.6 + 8;
  const bx = Math.max(0, Math.floor(cx - size / 2)), by = Math.max(0, Math.floor(cy - size / 2));
  const bw = Math.min(W, Math.ceil(cx + size / 2)) - bx, bh = Math.min(H, Math.ceil(cy + size / 2)) - by;
  const k = 512 / Math.max(bw, bh);
  const c = document.createElement('canvas');
  c.width = Math.round(bw * k);
  c.height = Math.round(bh * k);
  c.getContext('2d')!.drawImage(src, bx, by, bw, bh, 0, 0, c.width, c.height);
  const sam = await samImage(c);
  return {
    w: W,
    h: H,
    async segment(x: number, y: number) {
      const r = await sam.segment((x - bx) * k, (y - by) * k);
      const back = (m: Uint8Array): Uint8Array => {
        const mask = new Uint8Array(W * H);
        for (let yy = by; yy < by + bh; yy++) for (let xx = bx; xx < bx + bw; xx++) {
          const u = Math.min(c.width - 1, Math.floor((xx - bx + 0.5) * k)), v = Math.min(c.height - 1, Math.floor((yy - by + 0.5) * k));
          mask[yy * W + xx] = m[v * c.width + u];
        }
        return mask;
      };
      return { mask: back(r.mask), score: r.score, alts: r.alts.map((a) => ({ mask: back(a.mask), score: a.score, area: a.area / (k * k) })) };
    },
  };
}
/**
 * 고른 사진에서 손톱 찾기. 손 점이 안 잡히는 가까운 네일 사진이 많아서 색으로 후보를 고른다:
 * 피부·흰/회색 배경·아주 어두운 곳을 뺀 픽셀을 색 상자로 묶고, 많은 색부터 그 색 덩어리 가운데를 SAM 으로 오려 낸 뒤
 * 손톱 모양(적당한 크기·길쭉함·꽉 찬 정도)인 것이 가장 많이 나온 색을 고른다.
 */
/**
 * 색으로 손톱 찾기: 피부·흰 바탕(바탕이 흰 사진만)·아주 어두운 곳을 뺀 픽셀을 색 상자로 묶고, 많은 색부터 그 색 덩어리 가운데를
 * SAM 으로 오려 손톱 모양(길쭉함·꽉 참)인 것이 가장 많은 색을 고른다. 손 점이 안 잡히는 가까운 손 사진용.
 */
/**
 * 찾은 '손톱' 가운데 윤기 반사 조각(사진의 0.3% 아래·거의 흰색)은 그 조각을 품은 손톱 전체(SAM 의 더 큰 후보: 4배 넘게,
 * 손톱 모양)로 바꾼다(이미 잡은 손톱과 겹치면 뺀다). 누드·젤리 손톱은 피부색에 가까워 반사만 잡혔다.
 */
async function glintsToNails(): Promise<void> {
  const base = stillRGBA();
  const { w, h, rgba } = base;
  let sam: SamImage | null = null;
  for (let k = myNails.length - 1; k >= 0; k--) {
    const m = myNails[k].mask;
    let a = 0, sx = 0, sy = 0;
    for (let i = 0; i < m.length; i++) if (m[i]) { a++; sx += i % w; sy += (i / w) | 0; }
    if (!a || a >= w * h * 0.003 || !nearWhite(rgba, m)) continue;
    sam ??= await mySamImage();
    const r = await sam.segment(sx / a, sy / a);
    let rep: { mask: Uint8Array; frame: NailFrame } | null = null;
    for (const c of r.alts) {
      if (c.area < a * 4 || c.area > w * h * 0.08 || c.score < 0.5 || overlapFrac(c.mask, m) < 0.9) continue;
      const cm = new Uint8Array(c.mask);
      hullFill(cm, w, h);
      const f = nailFrame(cm, w, h, null, rgba);
      if (!f) continue;
      const len = f.t1 - f.t0, wid = 2 * Math.max(...f.prof);
      if (len / Math.max(1, wid) < 1.15 || len / Math.max(1, wid) > 5 || c.area / Math.max(1, len * wid) < 0.55) continue;
      rep = { mask: cm, frame: f };
      break;
    }
    // 손톱 전체를 못 찾으면 그대로 둔다(작은 흰 매니큐어 손톱일 수 있다). 이미 잡은 손톱과 겹치면 뺀다
    if (rep && myNails.some((n, j) => j !== k && overlapFrac(n.mask, rep!.mask) > 0.5)) myNails.splice(k, 1);
    else if (rep) myNails[k] = rep;
  }
}
/** outer 에서 inner 를 뺀 픽셀 가운데 피부색(YCbCr) 비율 */
function skinFrac(rgba: Uint8ClampedArray, outer: Uint8Array, inner: Uint8Array): number {
  let n = 0, sk = 0;
  for (let i = 0; i < outer.length; i++) {
    if (!outer[i] || inner[i]) continue;
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const Y = 0.299 * r + 0.587 * g + 0.114 * b, Cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b, Cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b;
    n++;
    if (Y > 60 && Cr > 135 && Cr < 175 && Cb > 85 && Cb < 128 && r > g) sk++;
  }
  return n ? sk / n : 1;
}
/** 마스크에서 밝은 쪽 40% 픽셀이 거의 흰색(밝기 L 88 넘게, 채도 10 아래)인가(윤기 반사: 넓힌 가장자리는 둘레 손톱색이 섞인다) */
/** 마스크 픽셀의 밝기 가운데 값 픽셀의 rgba 위치(색 판정에 그 픽셀을 쓴다) */
function maskMedianOffset(rgba: Uint8ClampedArray, mask: Uint8Array): number {
  const idx: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) idx.push(i);
  if (!idx.length) return 0;
  idx.sort((a, b) => rgba[a * 4] + rgba[a * 4 + 1] + rgba[a * 4 + 2] - (rgba[b * 4] + rgba[b * 4 + 1] + rgba[b * 4 + 2]));
  return idx[idx.length >> 1] * 4;
}
/** 마스크가 사진 가장자리에 닿은 픽셀 수 */
function touchesBorder(mask: Uint8Array, w: number, h: number): number {
  let n = 0;
  for (let x = 0; x < w; x++) n += (mask[x] ? 1 : 0) + (mask[(h - 1) * w + x] ? 1 : 0);
  for (let y = 0; y < h; y++) n += (mask[y * w] ? 1 : 0) + (mask[y * w + w - 1] ? 1 : 0);
  return n;
}
function nearWhite(rgba: Uint8ClampedArray, mask: Uint8Array): boolean {
  const lab = new Float32Array(3);
  const px: [number, number][] = [];
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i]) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    px.push([lab[0], Math.hypot(lab[1], lab[2])]);
  }
  if (px.length < 10) return false;
  px.sort((p, q) => q[0] - p[0]);
  const top = px.slice(0, Math.max(4, Math.round(px.length * 0.4)));
  return top.reduce((t, p) => t + p[0], 0) / top.length > 88 && top.reduce((t, p) => t + p[1], 0) / top.length < 10;
}
/** 두 마스크가 겹친 넓이 / 작은 쪽 넓이 */
function overlapFrac(a: Uint8Array, b: Uint8Array): number {
  let na = 0, nb = 0, ov = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i]) na++;
    if (b[i]) { nb++; if (a[i]) ov++; }
  }
  return ov / Math.max(1, Math.min(na, nb));
}
async function nailsByColor(
  rgba: Uint8ClampedArray, w: number, h: number, getSam: () => Promise<SamImage>, status: string,
  region: { x0: number; y0: number; x1: number; y1: number }[] = [],
  /** 손톱 안 무늬 칸 → 손톱 전체로 넓히기(내 손톱만: 고른 사진은 넓힌 후보가 어두운 소매·배경까지 잡아 네일이 짙은 남색이 됐다) */
  expandRim = false,
): Promise<{ mask: Uint8Array; frame: NailFrame }[]> {
  const G = 4;
  const gw = Math.floor(w / G), gh = Math.floor(h / G);
  const bin = new Int16Array(gw * gh).fill(-1);
  const counts = new Map<number, number>();
  const isWhite = (o: number): boolean => {
    const mx = Math.max(rgba[o], rgba[o + 1], rgba[o + 2]), mn = Math.min(rgba[o], rgba[o + 1], rgba[o + 2]);
    return mx - mn < 22 && mx > 170;
  };
  // 흰·밝은 회색은 바탕이 흰 사진(가장자리에 흰 곳이 많음)에서만 뺀다: 어두운 바탕의 흰·연분홍 네일은 후보로 둔다
  let edge = 0, edgeWhite = 0;
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    if (x !== 0 && y !== 0 && x !== gw - 1 && y !== gh - 1) continue;
    edge++;
    if (isWhite(((y * G + 2) * w + x * G + 2) * 4)) edgeWhite++;
  }
  const whiteBg = edgeWhite > edge * 0.25;
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const o = ((y * G + 2) * w + x * G + 2) * 4;
    // 손이 잡혔으면 손 둘레에서만 색을 센다(손톱과 비슷한 색의 머리카락·옷이 많은 색을 차지하지 않게)
    if (region.length && !region.some((b) => x * G >= b.x0 && x * G <= b.x1 && y * G >= b.y0 && y * G <= b.y1)) continue;
    const r = rgba[o], g = rgba[o + 1], b = rgba[o + 2];
    const mx = Math.max(r, g, b);
    if (mx < 35) continue; // 아주 어두움
    if (whiteBg && isWhite(o)) continue; // 흰·밝은 회색 배경
    // 피부(YCbCr): 손가락은 빼고 손톱 후보만
    const Y = 0.299 * r + 0.587 * g + 0.114 * b, Cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b, Cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b;
    if (Y > 60 && Cr > 135 && Cr < 175 && Cb > 85 && Cb < 128 && r > g) continue;
    const k = (r >> 5) * 64 + (g >> 5) * 8 + (b >> 5);
    bin[y * gw + x] = k;
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  // 사진 가장자리에 많이 나오는 색(바탕 천·벽)은 손톱이 아니다: 가장자리 칸의 5% 넘게 차지하는 색 상자(이웃 포함)는 뺀다
  const edgeCnt = new Map<number, number>();
  let edgeAll = 0;
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    if (x !== 0 && y !== 0 && x !== gw - 1 && y !== gh - 1) continue;
    edgeAll++;
    const k = bin[y * gw + x];
    if (k >= 0) edgeCnt.set(k, (edgeCnt.get(k) ?? 0) + 1);
  }
  const nearB = (a: number, b: number): boolean => Math.abs((a >> 6) - (b >> 6)) <= 1 && Math.abs(((a >> 3) & 7) - ((b >> 3) & 7)) <= 1 && Math.abs((a & 7) - (b & 7)) <= 1;
  const isBg = (k: number): boolean => {
    let e = 0;
    for (const [kk, c] of edgeCnt) if (nearB(kk, k)) e += c;
    return e > edgeAll * 0.05;
  };
  // 많은 색 여섯 개까지(배경 천·벽 같은 큰 색 덩어리가 앞을 차지해도 손톱 색까지 닿게)
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k).filter((k) => !isBg(k)).slice(0, 6);
  const near = (a: number, b: number): boolean => Math.abs((a >> 6) - (b >> 6)) <= 1 && Math.abs(((a >> 3) & 7) - ((b >> 3) & 7)) <= 1 && Math.abs((a & 7) - (b & 7)) <= 1;
  let best: { mask: Uint8Array; frame: NailFrame }[] = [];
  let bestCtr = 0;
  let sam: SamImage | null = null;
  for (const k of top) {
    // 그 색(이웃 색 상자 포함) 덩어리
    const seen = new Uint8Array(gw * gh);
    const kWhite0 = (k >> 6) >= 6 && ((k >> 3) & 7) >= 6 && (k & 7) >= 6;
    const blobs: { n: number; x: number; y: number }[] = [];
    for (let i = 0; i < bin.length; i++) {
      if (seen[i] || bin[i] < 0 || !near(bin[i], k)) continue;
      const st = [i];
      seen[i] = 1;
      let n = 0, sx = 0, sy = 0;
      while (st.length) {
        const q = st.pop()!;
        const qx = q % gw, qy = (q / gw) | 0;
        n++; sx += qx; sy += qy;
        for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const nx = qx + dx, ny = qy + dy;
          if (nx < 0 || ny < 0 || nx >= gw || ny >= gh) continue;
          const j = ny * gw + nx;
          if (!seen[j] && bin[j] >= 0 && near(bin[j], k)) { seen[j] = 1; st.push(j); }
        }
      }
      // 흰 반사 조각은 작다(누드·젤리 손톱의 반사 한 줄이 4~7칸): 흰 색 상자는 2칸부터(둥근 점 반사는 2~3칸)
      if (n >= (kWhite0 ? 2 : 8)) blobs.push({ n, x: (sx / n) * G + 2, y: (sy / n) * G + 2 });
    }
    blobs.sort((a, b) => b.n - a.n);
    if (!blobs.length) continue;
    console.debug('[nail] bin', k, 'rgb', k >> 6, (k >> 3) & 7, k & 7, 'blobs', blobs.slice(0, 4).map((b) => `${b.n}@${Math.round(b.x)},${Math.round(b.y)}`).join(' '), 'whiteBg', whiteBg);
    setStatus(status);
    sam ??= await getSam();
    const got: { mask: Uint8Array; frame: NailFrame }[] = [];
    let gotCtr = 0;
    // 거의 흰 색(반사)이면 손톱마다 반사가 여럿이라 더 많이 본다(같은 손톱은 겹침으로 하나만 남는다)
    const kWhite = (k >> 6) >= 6 && ((k >> 3) & 7) >= 6 && (k & 7) >= 6;
    for (const b of blobs.slice(0, kWhite ? 12 : 4)) {
      const r = await sam.segment(b.x, b.y);
      let area = 0;
      for (let i = 0; i < r.mask.length; i++) area += r.mask[i];
      let glintGrown = false;
      // 윤기 나는 손톱의 하얀 반사 조각(작고 거의 흰색)을 잡았으면, 그 조각을 품은 더 큰 후보(손톱 전체)로 바꾼다:
      // 피부색에 가까운 누드·젤리 손톱은 색 찾기에서 피부로 빠지고 반사만 남아, 손톱마다 반사 조각만 칠해졌다
      if (area > 0 && area < w * h * 0.003 && nearWhite(rgba, r.mask)) {
        const big = r.alts.find((c) => c.area >= area * 4 && c.area <= w * h * 0.08 && c.score >= 0.5 && overlapFrac(c.mask, r.mask) > 0.9);
        if (big) {
          r.mask = big.mask;
          r.score = Math.max(r.score, big.score);
          area = big.area;
          glintGrown = true;
        }
      }
      // 손톱 안 무늬 한 칸(크롬 테두리 안 색 칸·프렌치 바탕)만 잡았으면 그 칸을 품은 손톱 전체로: 5배 안으로 크고,
      // 늘어난 테두리가 바깥 둘레와 색이 뚜렷이 다르고(ΔE 12 넘게) 피부색 픽셀이 30% 아래면 손톱의 일부다
      // (손가락까지 잡은 후보는 테두리가 피부색. 크롬 스틸레토는 안쪽 청록 칸만 잡혀 손톱의 1/4 만 칠해졌다)
      for (const c of expandRim ? r.alts : []) {
        if (c.area <= area * 1.15 || c.area > area * 5 || c.score < 0.5 || overlapFrac(c.mask, r.mask) < 0.9) continue;
        if (rimContrast(rgba, w, h, r.mask, c.mask) < 12 || skinFrac(rgba, c.mask, r.mask) > 0.3) continue;
        // 넓힌 것도 손톱 모양이어야(아래와 같은 기준): 이미 손톱 전체인 윤기 나는 손톱의 큰 후보는 옆 손톱·그늘까지 잡은
        // 울퉁불퉁한 덩어리라(채움 0.1~0.45) 넓히면 손톱이 통째로 빠졌다
        const cm = new Uint8Array(c.mask);
        hullFill(cm, w, h);
        const cf = nailFrame(cm, w, h, null, rgba);
        if (!cf) continue;
        const cl = cf.t1 - cf.t0, cwid = 2 * Math.max(...cf.prof);
        let ca = 0;
        for (let i = 0; i < cm.length; i++) ca += cm[i];
        // 껍질로 채운 뒤에도 5배 안(흩어진 조각 후보는 껍질이 손 전체만큼 커진다: 07번 손 6757 → 146230)
        if (ca > area * 5) continue;
        if (cl / Math.max(1, cwid) < 1.15 || cl / Math.max(1, cwid) > 5 || ca / Math.max(1, cl * cwid) < 0.55) continue;
        // 무늬 칸은 손톱 안에 사방으로 테두리를 두고 들어 있고 손톱과 같은 방향이다: 넓힌 것이 한쪽(옆 손톱 쪽)으로만 늘어나거나
        // 방향이 틀어지면(가로로 이어진 두 손톱) 손톱이 아니다
        const f0 = nailFrame(r.mask, w, h, null, rgba);
        const bb = (m: Uint8Array) => { let x0 = w, y0 = h, x1 = -1, y1 = -1; for (let i = 0; i < m.length; i++) if (m[i]) { const x = i % w, y = (i - x) / w; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; } return [x0, y0, x1, y1]; };
        const bi = bb(r.mask), bo = bb(cm);
        // 사방 여백은 넓힌 상자의 3%(옆 손톱으로 늘어난 것은 한쪽 여백이 0~1px, 무늬 칸은 12px 넘게), 방향은 코사인 0.6 넘게
        // (무늬 칸은 둥글어 손톱 축과 30° 남짓 어긋나기도 한다)
        const mg = Math.max(2, 0.03 * Math.max(bo[2] - bo[0], bo[3] - bo[1]));
        if (!f0 || Math.abs(f0.ax * cf.ax + f0.ay * cf.ay) < 0.6) continue;
        if (bi[0] - bo[0] < mg || bi[1] - bo[1] < mg || bo[2] - bi[2] < mg || bo[3] - bi[3] < mg) continue;
        r.mask = cm;
        area = ca;
        break;
      }
      // 작은 손톱(손이 사진의 일부)도: 최소 넓이는 사진의 0.06%(모양·대비로 거른다)
      if (r.score < 0.6 || area < w * h * 0.0006 || area > w * h * 0.08) continue;
      hullFill(r.mask, w, h);
      const f = nailFrame(r.mask, w, h, null, rgba);
      if (!f) continue;
      const len = f.t1 - f.t0, wid = 2 * Math.max(...f.prof);
      const elong = len / Math.max(1, wid);
      const solid = area / Math.max(1, len * wid);
      if (elong < 1.15 || elong > 5 || solid < 0.55) continue;
      // 둘레(손가락 피부)와 색이 뚜렷이 달라야 손톱(손가락 사이 피부·그늘이 잡히는 것 막기)
      // 반사 조각에서 넓힌 손톱(누드·젤리)은 둘레 피부와 색 차가 4~7 뿐이지만 반사를 품은 손톱 모양이라는 근거가 있다: 4 부터
      const ctr = maskContrast(rgba, w, h, r.mask);
      if (ctr < (glintGrown ? 4 : 8)) continue;
      // 바탕이 흰 사진에서 오린 것의 색이 흰·밝은 회색이면 바탕 조각이다(검은 바지 사이로 보인 흰 배경 틈이 손톱 모양이라
      // 바탕색으로 칠해졌다: 색 찾기는 흰색을 빼지만 틈 가장자리 회색 칸이 씨앗이 되어 SAM 이 틈 전체를 오렸다),
      // 사진 가장자리에 닿은 것(짧은 변 1% 넘게)도 손톱이 아니다
      if (whiteBg && isWhite(maskMedianOffset(rgba, r.mask))) continue;
      if (touchesBorder(r.mask, w, h) > Math.min(w, h) * 0.01) continue;
      // 한 손톱이 무늬(드립·프렌치)로 갈려 같은 색 덩어리 둘이면 SAM 이 같은 손톱을 두 번 오린다: 절반 넘게 겹치면 하나만
      if (got.some((g) => overlapFrac(g.mask, r.mask) > 0.5)) continue;
      got.push({ mask: r.mask, frame: f });
      gotCtr += ctr;
    }
    // 손톱 수가 많은 색, 같으면 둘레와 더 뚜렷이 다른 색
    if (got.length > best.length || (got.length === best.length && got.length > 0 && gotCtr > bestCtr)) {
      best = got;
      bestCtr = gotCtr;
    }
    if (best.length >= 3) break;
  }
  return best;
}
/** 사진은 거울상이 아니라 왼손·오른손 판단이 뒤바뀌기도 한다: 손등 방향 두 가지 중 손톱이 더 잘 보이는 쪽 */
function bestQuads(hp: { p: { x: number; y: number }[]; z: number[]; handed: string }): ReturnType<typeof nailQuads> {
  const a = nailQuads(hp, 1), b = nailQuads(hp, -1);
  const sum = (q: typeof a) => q.reduce((t, x) => t + x.vis, 0);
  return sum(a) >= sum(b) ? a : b;
}
async function findRefNails(): Promise<void> {
  const src = lastSource;
  if (!src || refNails?.src === src) return;
  const s = Math.min(1, 1024 / Math.max(src.width, src.height));
  const w = Math.round(src.width * s), h = Math.round(src.height * s);
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const cx = cv.getContext('2d', { willReadFrequently: true })!;
  cx.drawImage(src, 0, 0, w, h);
  const rgba = cx.getImageData(0, 0, w, h).data;
  // ① 손 점이 잡히면 손가락 끝(손톱 자리)마다 SAM 으로 오린다: 색 덩어리로 찾으면 손가락 사이 그늘·배경이 잡히기도 한다
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  const handsRef = await photoAnalyzer.handPoints(cv).catch(() => []);
  const quadsRef = handsRef.flatMap(bestQuads).filter((q) => q.vis > 0.3);
  if (quadsRef.length >= 2) {
    setStatus('고른 사진에서 손톱을 찾는 중…');
    const samR = await samImage(cv);
    const got: { mask: Uint8Array; frame: NailFrame }[] = [];
    for (const q of quadsRef) {
      const r = await samR.segment(q.c.x, q.c.y);
      let area = 0;
      for (let i = 0; i < r.mask.length; i++) area += r.mask[i];
      // 사진 가장자리에 닿은 양(넓게 닿으면 배경)
      let edgeN = 0;
      for (let x = 0; x < w; x++) edgeN += r.mask[x] + r.mask[(h - 1) * w + x];
      for (let y = 0; y < h; y++) edgeN += r.mask[y * w] + r.mask[y * w + w - 1];
      // 가까이 찍은 긴 네일은 손 점의 손톱 크기가 실제보다 훨씬 작게 나와 크기로는 못 거른다:
      // 사진의 8% 이하, 가장자리에 넓게 닿지 않음(배경), 손톱 모양(길쭉함·꽉 참)으로 거른다
      if (r.score < 0.6 || area < 30 || area > w * h * 0.08 || edgeN > Math.sqrt(area) * 0.5) continue;
      hullFill(r.mask, w, h);
      const f = nailFrame(r.mask, w, h, q.dir, rgba);
      if (!f) continue;
      const len = f.t1 - f.t0, wid = 2 * Math.max(...f.prof);
      const elong = len / Math.max(1, wid), solid = area / Math.max(1, len * wid);
      if (elong < 1.1 || elong > 5 || solid < 0.5) continue;
      // 둘레와 색이 거의 같으면 손톱이 아니다(손 점이 엉뚱한 곳에 잡혀 손등·배경 덩어리를 오렸다)
      if (maskContrast(rgba, w, h, r.mask) < 8) continue;
      got.push({ mask: r.mask, frame: f });
    }
    // 몇 개만 찾았으면(손 점이 일부만·엉뚱하게 잡힘) 색 덩어리 방식도 해 보고 더 많이 찾은 쪽을 쓴다
    if (got.length >= 4) {
      refNails = { src, rgba, w, h, nails: rankNails(rgba, w, h, got) };
      return;
    }
    if (got.length >= 2) {
      const byColor = await nailsByColor(rgba, w, h, () => samImage(cv), '고른 사진에서 손톱을 찾는 중…');
      refNails = { src, rgba, w, h, nails: rankNails(rgba, w, h, byColor.length > got.length ? byColor : got) };
      return;
    }
  }
  // ② 손 점이 없으면(손톱만 가까이 찍힌 사진 등) 색 덩어리로
  const best = await nailsByColor(rgba, w, h, () => samImage(cv), '고른 사진에서 손톱을 찾는 중…');
  // 스타일 AI가 읽은 네일 색(연핑크 등)과 색도가 크게 다르면 손가락 사이 그늘·배경(흰 털옷 조각)을 잡은 것: 버리고 그 색의 가상 손톱으로
  if (best.length && nailLabelColor) {
    const lab = new Float32Array(3), lab2 = new Float32Array(3);
    const rs: number[] = [], gs: number[] = [], bs: number[] = [];
    for (const n of best) for (let i = 0; i < n.mask.length; i += 3) if (n.mask[i]) { rs.push(rgba[i * 4]); gs.push(rgba[i * 4 + 1]); bs.push(rgba[i * 4 + 2]); }
    const med = (a: number[]): number => a.sort((x, y) => x - y)[a.length >> 1];
    rgb2lab(med(rs), med(gs), med(bs), lab);
    rgb2lab(nailLabelColor[0] * 255, nailLabelColor[1] * 255, nailLabelColor[2] * 255, lab2);
    // 색도(a·b) 거리 13 넘거나 밝기 60 넘게 다르면: 밝기만 보면 어두운 배경의 크롬 실버(가운데 색 L 30, 이름 색 L 81, Δab 11)까지 버려졌다.
    // 흰 털옷 조각(회색, Δab 14.9) 은 버리고 누드 핑크 진짜 손톱(Δab 12.0) 은 남는 경계
    console.log('[nail] 고른 손톱 색 vs AI 이름 색', JSON.stringify({ nails: best.length, lab: Array.from(lab).map((v) => +v.toFixed(1)), label: Array.from(lab2).map((v) => +v.toFixed(1)) }));
    if (Math.hypot(lab[1] - lab2[1], lab[2] - lab2[2]) > 13 || Math.abs(lab[0] - lab2[0]) > 60) {
      refNails = { src, rgba, w, h, nails: [] };
      return;
    }
  }
  refNails = { src, rgba, w, h, nails: rankNails(rgba, w, h, best) };
  // 사진 분석이 손을 못 찾아 손톱 색을 배경에서 잘못 읽었을 수 있다 → 찾은 손톱 픽셀의 가운데 색으로 바로잡는다
  if (best.length && chosen.nail.color) {
    const rs: number[] = [], gs: number[] = [], bs: number[] = [];
    for (const n of best) for (let i = 0; i < n.mask.length; i += 3) if (n.mask[i]) { rs.push(rgba[i * 4]); gs.push(rgba[i * 4 + 1]); bs.push(rgba[i * 4 + 2]); }
    const med = (a: number[]): number => a.sort((x, y) => x - y)[a.length >> 1] / 255;
    chosen.nail.color = lin([med(rs), med(gs), med(bs)]);
  }
}
/** 내 손톱: 손 점이 잡히면 손톱 자리를 자동으로, 아니면 눌러 달라고 안내 */
let nailsEnsuring: Promise<void> | null = null;
let nailNoArtNoted: ImageBitmap | null = null;
function ensureNails(): Promise<void> {
  nailsEnsuring ??= ensureNailsOnce().finally(() => (nailsEnsuring = null));
  return nailsEnsuring;
}
async function ensureNailsOnce(): Promise<void> {
  if (!PHOTO_MODE || !photoOriginal || swapMode) return;
  if (!refNailsBusy) refNailsBusy = findRefNails().finally(() => (refNailsBusy = null));
  await refNailsBusy;
  if (!refNailArt()) {
    if (lastSource && nailNoArtNoted !== lastSource) {
      nailNoArtNoted = lastSource;
      setStatus('고른 사진에서 손톱 모양을 찾지 못해 색만 입혔어요');
    }
    return;
  }
  if (nailsAuto === photoOriginal) return;
  nailsAuto = photoOriginal;
  // 사진용 손 찾기(정지 사진 모드, 크기·여백 바꿔 여러 번)를 먼저: 실시간 추적은 영상용이라 사진에서도 실행 때마다
  // 손 점이 조금씩 달라(긴 네일 손은 한곳에 몰리기도) 찾는 손톱 수가 2~4개로 들쭉날쭉했다.
  // 못 찾으면 실시간 손 점이 잡힐 때까지 잠깐(실제 시간으로 2초까지만: 큰 사진은 기다림 한 번이 길어진다)
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  let handsMine = await photoAnalyzer.handPoints(stillCanvas()).catch(() => []);
  if (!handsMine.length) {
    const th0 = performance.now();
    while (!engine.lastHands.length && performance.now() - th0 < 2000) await new Promise((r) => setTimeout(r, 100));
    handsMine = engine.lastHands;
  }
  // 사진은 거울상이 아니라 왼손·오른손 판단이 뒤바뀌기도 한다: 손등 방향 두 가지 중 손톱이 더 잘 보이는 쪽
  const quads = handsMine.flatMap(bestQuads).filter((q) => q.vis > 0.3);
  if (quads.length) {
    setStatus('손톱을 찾는 중…');
    const sam = await mySamImage();
    // 손 점으로 정한 손톱 크기(길이·폭)를 넘는 덩어리(손가락·손 전체가 잡힌 것)는 버린다
    // 손톱 크기 기준: 손가락 끝마디로 잰 값은 가까이 찍은 손에서 실제보다 훨씬 작게 나와, 손 크기(손목→가운데 손가락 뿌리)의 18%와 큰 쪽
    const st = stillRGBA();
    for (const hp of handsMine) {
      const hs = Math.hypot(hp.p[9].x - hp.p[0].x, hp.p[9].y - hp.p[0].y);
      // 손이 사진 짧은 변의 15% 아래면(전신 사진) 손 둘레만 잘라 키운 SAM 으로
      // (작은 손은 긴 네일이 아닌 맨 손톱이 대부분이고 손가락 옆면을 길게 잡기 쉬워 길이 기준을 2.5배로 좁힌다)
      const small = hs < Math.min(st.w, st.h) * 0.15;
      const hsam = small ? await handSamImage(hp) : sam;
      for (const q of bestQuads(hp).filter((x) => x.vis > 0.3)) {
        await addMyNail(hsam, q.c.x, q.c.y, q.dir, true, { c: q.c, size: Math.max(q.len, q.width, hs * 0.18), maxExt: small ? 2.5 : 5 });
      }
    }
  }
  console.debug('[nail] pose', myNails.length, 'hands', handsMine.length);
  dropSkinNails();
  // 손 점으로 몇 개만 찾았으면(손가락이 겹쳐 손 점이 어긋남) 찾은 손톱 색과 비슷한 손 둘레 덩어리를 더 찾는다
  // (빨간 매니큐어 손톱 넷 중 둘만 바뀌면 섞여 보여 어색하다)
  if (myNails.length && myNails.length < 10) await moreNailsLikeMine(handsMine);
  // 손 점으로 4개 미만이면(가까이 찍은 손·긴 네일은 손 점이 엉뚱한 곳에 몰리기도 한다) 색으로도 찾아 더 많이 찾은 쪽:
  // 매니큐어 바른 손톱은 피부와 색이 뚜렷이 다르다(전에는 하나라도 찾으면 색으로 안 찾아 손톱 하나만 바뀌었다)
  let colorFound = !myNails.length;
  if (myNails.length < 4) {
    const byPose = myNails.splice(0);
    const base = stillRGBA();
    // 손 점으로 찾을 때와 같은 기준: 사진의 3% 넘는 덩어리는 손톱이 아니고, 손이 잡혔으면 손 둘레(손 크기 50% 여백: 서로 낀 손은 점이 어긋난다) 안만
    // (서로 낀 손 사진에서 손톱 색과 비슷한 어두운 머리카락이 큰 손톱으로 칠해졌다)
    const boxes = handsMine.map((hp) => {
      const xs = hp.p.map((q) => q.x), ys = hp.p.map((q) => q.y);
      const m = Math.hypot(hp.p[9].x - hp.p[0].x, hp.p[9].y - hp.p[0].y) * 0.5;
      return { x0: Math.min(...xs) - m, y0: Math.min(...ys) - m, x1: Math.max(...xs) + m, y1: Math.max(...ys) + m };
    });
    let found = await nailsByColor(base.rgba, base.w, base.h, () => mySamImage(), '손톱을 찾는 중…', boxes, true);
    console.debug('[nail] color(boxes)', found.length, 'byPose', byPose.length);
    // 손 점으로 하나 이하만 찾았으면 손 점 자체가 어긋났을 수 있다: 사진 전체에서도 찾아 더 많이 찾은 쪽
    // (긴 네일 손을 가까이 찍으면 손 찾기가 구석에 작은 손을 잡아, 그 둘레만 세느라 위쪽 손톱 셋을 못 봤다)
    if (byPose.length <= 1 && boxes.length) {
      const all = await nailsByColor(base.rgba, base.w, base.h, () => mySamImage(), '손톱을 찾는 중…', [], true);
      console.debug('[nail] color(all)', all.length);
      if (all.length > found.length) {
        found = all;
        boxes.length = 0;
      }
    }
    // 너무 큰 덩어리 기준: 사진의 3%, 단 찾은 손톱 넓이 가운데 값의 1.6배가 더 크면 그쪽(8%까지: 가까이 찍은 긴 네일은 하나가 사진의 4%)
    const areas = found.map((n) => n.mask.reduce((t, v) => t + v, 0)).sort((x, y) => x - y);
    let maxA = Math.max(base.w * base.h * 0.03, Math.min(base.w * base.h * 0.08, (areas[areas.length >> 1] ?? 0) * 1.6));
    // 손이 잡혔으면 손 크기로도 묶는다: 손톱은 손바닥 길이(손목→가운뎃손가락 뿌리)² 의 2~4%(긴 네일) → 10% 까지.
    // 전신 사진(00055)에서 사진 전체 색 찾기가 어두운 보라 소매(7,700px, 사진의 1%)를 손톱 하나로 잡아 소매에 네일 아트가 칠해졌음
    if (handsMine.length) {
      const hm = Math.max(...handsMine.map((hp) => Math.hypot(hp.p[9].x - hp.p[0].x, hp.p[9].y - hp.p[0].y)));
      maxA = Math.min(maxA, hm * hm * 0.1);
    }
    console.debug('[nail] maxA', Math.round(maxA), 'areas', areas.map((v) => Math.round(v)).join(','));
    for (const n of found) {
      let a = 0, sx = 0, sy = 0;
      for (let i = 0; i < n.mask.length; i++) if (n.mask[i]) { a++; sx += i % base.w; sy += (i / base.w) | 0; }
      if (!a || a > maxA) continue;
      const cx = sx / a, cy = sy / a;
      if (boxes.length && !boxes.some((b) => cx >= b.x0 && cx <= b.x1 && cy >= b.y0 && cy <= b.y1)) continue;
      if (maskContrast(base.rgba, base.w, base.h, n.mask) >= 4) {
        growNailMask(n.mask, base.w, base.h, base.rgba, Math.min(4, Math.max(2, Math.round(Math.sqrt(a) * 0.08))));
        myNails.push(n);
      }
    }
    if (myNails.length > byPose.length) colorFound = true;
    else myNails.splice(0, myNails.length, ...byPose);
  }
  // 색으로 몇 개만 찾았어도(서로 낀 손) 그 손톱 색으로 나머지를 더 찾는다
  if (colorFound && myNails.length && myNails.length < 10) await moreNailsLikeMine(handsMine);
  await glintsToNails();
  dropSkinNails();
  dropOversizeNails();
  setStatus(myNails.length ? '빠진 손톱은 눌러서 더하고, 잘못 잡힌 곳은 다시 누르면 빠져요' : '손톱을 하나씩 눌러 주세요(누른 손톱에 고른 사진의 네일이 입혀져요)');
  nailGen++;
  applyAll();
}
/**
 * 매니큐어 바른 손톱이 있으면, 색이 손 피부(손톱 둘레 띠들의 가운데 색)와 거의 같은 '손톱'은 손가락 살을 잡은 것으로 보고 뺀다.
 * (손 점 손끝 자리에 SAM 이 손가락 살을 손톱 크기로 잡아 네일이 피부에 칠해졌다. 맨 손톱도 같이 빠지지만 피부에 칠하는 것보다 낫다)
 */
function dropSkinNails(): void {
  if (myNails.length < 2) return;
  const { w, h, rgba } = stillRGBA();
  const lab = new Float32Array(3);
  const ring: number[][] = [[], [], []];
  const means = myNails.map((m) => {
    const c = [0, 0, 0];
    let n = 0;
    for (let i = 0; i < m.mask.length; i += 2) {
      if (!m.mask[i]) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      c[0] += lab[0]; c[1] += lab[1]; c[2] += lab[2]; n++;
    }
    // 둘레 띠(4~8px 밖)
    let x0 = w, y0 = h, x1 = 0, y1 = 0;
    for (let i = 0; i < m.mask.length; i++) if (m.mask[i]) { const x = i % w, y = (i / w) | 0; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    for (let y = Math.max(0, y0 - 8); y <= Math.min(h - 1, y1 + 8); y += 2) for (let x = Math.max(0, x0 - 8); x <= Math.min(w - 1, x1 + 8); x += 2) {
      const i = y * w + x;
      if (m.mask[i]) continue;
      let near = false;
      for (let d = 4; d <= 8 && !near; d += 4) for (const [dx, dy] of [[d, 0], [-d, 0], [0, d], [0, -d]]) { const xx = x + dx, yy = y + dy; if (xx >= 0 && yy >= 0 && xx < w && yy < h && m.mask[yy * w + xx]) { near = true; break; } }
      if (!near) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      ring[0].push(lab[0]); ring[1].push(lab[1]); ring[2].push(lab[2]);
    }
    return n ? c.map((v) => v / n) : null;
  });
  if (!ring[0].length) return;
  const skin = ring.map((v) => v.sort((p, q) => p - q)[v.length >> 1]);
  const dE = means.map((m) => (m ? Math.hypot((m[0] - skin[0]) * 0.5, m[1] - skin[1], m[2] - skin[2]) : 0));
  if (!dE.some((v) => v > 30)) return;
  for (let k = myNails.length - 1; k >= 0; k--) if (dE[k] < 15) myNails.splice(k, 1);
}
/**
 * 같은 사진 손톱들 넓이 가운데 값의 1.6배 넘는 '손톱'은 SAM 이 손끝 살까지 잡은 것: 뺀다(3개 이상일 때만, 한 손 크기가 비슷하다는 가정).
 * 단 살이 섞이지 않은 매니큐어 손톱(둘레와 색 차 25 이상, 매니큐어 색으로 다듬을 게 없음)은 남긴다:
 * 정면으로 보이는 엄지 손톱은 옆으로 본 손톱들보다 3배 가까이 크다
 */
function dropOversizeNails(): void {
  if (myNails.length < 3) return;
  const { w, h, rgba } = stillRGBA();
  const areas = myNails.map((m) => m.mask.reduce((a, v) => a + v, 0));
  const med = [...areas].sort((a, b) => a - b)[areas.length >> 1];
  for (let k = myNails.length - 1; k >= 0; k--) {
    if (areas[k] <= med * 1.6) continue;
    const m = myNails[k].mask;
    if (maskContrast(rgba, w, h, m) >= 25 && splitPolished(m, w, h, rgba)[0] === m) continue;
    myNails.splice(k, 1);
  }
}
async function moreNailsLikeMine(hands: { p: { x: number; y: number }[] }[]): Promise<void> {
  const base = stillRGBA();
  const { w, h, rgba } = base;
  const lab = new Float32Array(3);
  // 손톱마다 색과 둘레(피부) 색: 둘레와 뚜렷이 다른(매니큐어 바른) 손톱 색만 씨앗으로
  // (빨간 손톱과 맨 손톱을 평균하면 분홍 = 피부색이 되어 손가락 끝을 손톱으로 잡았다)
  const seeds: [number, number, number][] = [];
  let area = 0;
  for (const m of myNails) {
    let n = 0, a = 0;
    const c = [0, 0, 0];
    for (let i = 0; i < m.mask.length; i++) {
      if (!m.mask[i]) continue;
      a++;
      if (i % 3) continue;
      rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
      n++; c[0] += lab[0]; c[1] += lab[1]; c[2] += lab[2];
    }
    area += a;
    if (!n) continue;
    const mean: [number, number, number] = [c[0] / n, c[1] / n, c[2] / n];
    if (maskContrast(rgba, w, h, m.mask) >= 25) seeds.push(mean);
  }
  if (!seeds.length) return;
  const avg = area / myNails.length, side = Math.sqrt(avg);
  const boxes = hands.map((hp) => {
    const xs = hp.p.map((q) => q.x), ys = hp.p.map((q) => q.y);
    // 서로 낀 손은 손 점이 한 손만·어긋나게 잡혀 다른 손 손톱이 둘레 밖이다: 넓게(손 크기 80%), 대신 색을 엄격히
    const m = Math.hypot(hp.p[9].x - hp.p[0].x, hp.p[9].y - hp.p[0].y) * 0.8;
    return { x0: Math.min(...xs) - m, y0: Math.min(...ys) - m, x1: Math.max(...xs) + m, y1: Math.max(...ys) + m };
  });
  const hs = Math.max(0, ...hands.map((hp) => Math.hypot(hp.p[9].x - hp.p[0].x, hp.p[9].y - hp.p[0].y)));
  const centers = myNails.map((m) => {
    let n = 0, sx = 0, sy = 0;
    for (let i = 0; i < m.mask.length; i++) if (m.mask[i]) { n++; sx += i % w; sy += (i / w) | 0; }
    return { x: sx / Math.max(1, n), y: sy / Math.max(1, n) };
  });
  const G = 3, gw = Math.floor(w / G), gh = Math.floor(h / G);
  const hit = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    const px = x * G + 1, py = y * G + 1, i = py * w + px;
    if (boxes.length && !boxes.some((b) => px >= b.x0 && px <= b.x1 && py >= b.y0 && py <= b.y1)) continue;
    if (myNails.some((m) => m.mask[i])) continue;
    rgb2lab(rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2], lab);
    // 밝기도 온전히 본다: 손가락 사이 그늘(손톱보다 어둡고 색이 비슷)이 손톱으로 잡혔다
    if (seeds.some((s) => Math.hypot(lab[0] - s[0], lab[1] - s[1], lab[2] - s[2]) < 12)) hit[y * gw + x] = 1;
  }
  const seen = new Uint8Array(gw * gh);
  const cands: { x: number; y: number; a: number }[] = [];
  for (let s0 = 0; s0 < hit.length; s0++) {
    if (!hit[s0] || seen[s0]) continue;
    const st = [s0];
    seen[s0] = 1;
    let c = 0, sx = 0, sy = 0;
    while (st.length) {
      const q = st.pop()!, qx = q % gw, qy = (q / gw) | 0;
      c++; sx += qx; sy += qy;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = qx + dx, ny = qy + dy, j = ny * gw + nx;
        if (nx >= 0 && ny >= 0 && nx < gw && ny < gh && hit[j] && !seen[j]) { seen[j] = 1; st.push(j); }
      }
    }
    const a = c * G * G;
    // 찾은 손톱 넓이의 0.1~4배만(겹쳐 끝만 보이는 손톱부터 가까이 정면으로 보이는 엄지 손톱까지:
    // 옆으로 본 손톱만 찾았으면 엄지는 3배 가까이 된다). 작은 덩어리는 SAM 이 손톱 전체로 넓힌다
    if (a < Math.max(150, avg * 0.1) || a > avg * 4) continue;
    const cx = (sx / c) * G + 1, cy = (sy / c) * G + 1;
    // 손톱은 손 하나 크기 안에 모여 있다: 이미 찾은 손톱에서 손 크기보다 먼 덩어리(다른 손가락 그늘·소매)는 뺀다
    if (hs > 0 && !centers.some((q) => Math.hypot(q.x - cx, q.y - cy) <= hs)) continue;
    cands.push({ x: cx, y: cy, a });
  }
  if (!cands.length) return;
  // 작은 손(전신 사진)은 가까운 손 둘레만 잘라 키운 SAM 과 좁은 길이 기준으로(사진 전체 SAM 은 손가락 옆면을 길게 잡았다)
  const small = hs > 0 && hs < Math.min(w, h) * 0.15;
  const hsams = new Map<number, Promise<SamImage>>();
  const samFor = (x: number, y: number): Promise<SamImage> => {
    if (!small) return mySamImage();
    let k = 0, bd = Infinity;
    hands.forEach((hp, j) => { const d = Math.hypot(hp.p[9].x - x, hp.p[9].y - y); if (d < bd) { bd = d; k = j; } });
    if (!hsams.has(k)) hsams.set(k, handSamImage(hands[k]));
    return hsams.get(k)!;
  };
  for (const c of cands.sort((p, q) => q.a - p.a).slice(0, 8)) {
    const before = myNails.length;
    await addMyNail(await samFor(c.x, c.y), c.x, c.y, null, true, { c: { x: c.x, y: c.y }, size: side, maxExt: small ? 2.5 : 5 });
    // SAM 이 손가락까지 잡아 찾은 손톱보다 훨씬 크면 버린다(매니큐어 색 쪽으로 다듬은 뒤라 4배까지는 손톱)
    while (myNails.length > before) {
      const m = myNails[myNails.length - 1].mask;
      let a = 0;
      for (let i = 0; i < m.length; i++) a += m[i];
      if (a <= avg * 4) break;
      myNails.pop();
    }
  }
}
async function addMyNail(
  sam: SamImage, x: number, y: number, hint: { x: number; y: number } | null, quiet = false,
  expect: { c: { x: number; y: number }; size: number; maxExt?: number } | null = null,
): Promise<boolean> {
  const base = stillRGBA();
  const r = await sam.segment(x, y);
  let area = 0, sx = 0, sy = 0, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < r.mask.length; i++) {
    if (!r.mask[i]) continue;
    area++;
    const px = i % base.w, py = (i - px) / base.w;
    sx += px; sy += py;
    if (px < x0) x0 = px; if (px > x1) x1 = px; if (py < y0) y0 = py; if (py > y1) y1 = py;
  }
  // 손톱은 작다: 사진의 3% 넘거나, (손 점이 있으면) 예상 손톱 크기보다 훨씬 크거나 가운데가 멀면 손가락·손이 잡힌 것.
  // 작은 손(전신 사진, maxExt 를 좁혀 부름)은 너무 작은 조각의 기준을 예상 손톱 크기로: 사진 넓이 0.03% 로 자르면
  // 전신 사진의 손톱(30픽셀 남짓)이 모두 버려졌다
  const smallHand = !!expect && (expect.maxExt ?? 5) < 5;
  const minA = smallHand ? Math.max(8, expect!.size * expect!.size * 0.15) : base.w * base.h * 0.0003;
  let bad = r.score < 0.5 || area < minA || area > base.w * base.h * 0.03;
  if (!bad && expect) {
    const ext = Math.max(x1 - x0, y1 - y0);
    const cd = Math.hypot(sx / area - expect.c.x, sy / area - expect.c.y);
    // 손 점의 손톱 크기는 손가락 끝마디 기준이라 긴 네일보다 작다(실제 손톱은 2~4배): 손가락·손 전체(7배 넘게)만 거른다
    bad = ext > expect.size * (expect.maxExt ?? 5) || area > expect.size * expect.size * 8 || cd > expect.size * 2.5;
    // 둘레 손가락 피부와 색이 거의 같으면 손톱이 아니라 손가락을 잡은 것
    const ctr = bad ? -1 : maskContrast(base.rgba, base.w, base.h, r.mask);
    if (!bad && ctr < 4) bad = true;
  }
  if (bad) {
    if (!quiet) setStatus('손톱을 찾지 못했어요. 손톱 가운데를 눌러 주세요');
    return false;
  }
  // 매니큐어 색 쪽 픽셀만으로 다듬고 떨어진 손톱은 나눈다(옆으로 본 두 손가락의 손톱 둘 + 사이 살이 한 덩어리로 잡혔다)
  let added = false;
  for (const part of splitPolished(r.mask, base.w, base.h, base.rgba)) {
    hullFill(part, base.w, base.h);
    let pa = 0;
    for (let i = 0; i < part.length; i++) pa += part[i];
    // 마스크가 실제 손톱보다 조금 작아 원래 매니큐어 색이 테두리로 비치지 않게 손톱 색 쪽 가장자리까지 넓힌다
    growNailMask(part, base.w, base.h, base.rgba, Math.min(4, Math.max(2, Math.round(Math.sqrt(pa) * 0.08))));
    // 이미 잡은 손톱과 절반 넘게 겹치면 같은 손톱(두 번 그리면 진해지고 테두리가 두꺼워진다)
    let dup = false;
    for (const m of myNails) {
      let ov = 0, a2 = 0;
      for (let i = 0; i < part.length; i++) if (part[i]) { a2++; if (m.mask[i]) ov++; }
      if (ov > a2 * 0.5) { dup = true; break; }
    }
    if (dup) continue;
    const f = nailFrame(part, base.w, base.h, hint, base.rgba);
    if (!f) continue;
    myNails.push({ mask: part, frame: f });
    added = true;
  }
  return added;
}
// 사진 모드에서 네일을 켰을 때 화면을 누르면: 손톱 더하기(이미 잡힌 손톱이면 빼기)
viewEl.addEventListener('pointerdown', (ev) => {
  if (!PHOTO_MODE || swapMode || tattooPicking || !photoOriginal || !chosen.nail.color || off.has('nail') || !refNailArt()) return;
  const cv = viewEl.querySelector('canvas')!;
  const r = cv.getBoundingClientRect();
  // object-fit: contain 으로 그려지므로 실제 그림이 차지하는 상자를 구한다
  const sc = Math.min(r.width / cv.width, r.height / cv.height);
  const ox = r.left + (r.width - cv.width * sc) / 2, oy = r.top + (r.height - cv.height * sc) / 2;
  const x = (ev.clientX - ox) / sc, y = (ev.clientY - oy) / sc;
  if (x < 0 || y < 0 || x >= cv.width || y >= cv.height) return;
  const i = Math.floor(y) * cv.width + Math.floor(x);
  const k = myNails.findIndex((n) => n.mask[i]);
  if (k >= 0) {
    myNails.splice(k, 1);
    nailGen++;
    applyAll();
    return;
  }
  void (async () => {
    setStatus('손톱 오려 내는 중…');
    const ok = await addMyNail(await mySamImage(), x, y, null);
    if (ok) {
      nailGen++;
      setStatus('');
      applyAll();
    }
  })();
});

// 자동 시험용: 사진 모드 손톱 상태
(window as unknown as { irisHair: unknown }).irisHair = () => ({ mine: mineHair, ref: refHair });
(window as unknown as { irisTattoo: unknown }).irisTattoo = tattoo;
(window as unknown as { irisAnalyzer: unknown }).irisAnalyzer = () => (photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE));
(window as unknown as { irisTatState: unknown }).irisTatState = () => ({ planOk: tatPlanOk, sameRef: tatPlan?.ref === refTat, baked: tatBaked(), refTat: !!refTat, user: refTat?.user, designSame: refTat?.design === tattoo.design, stillKey: stillHairKey, busy: !!stillHairBusy });
(window as unknown as { irisNail: unknown }).irisNail = { get ref() { return refNails; }, get mine() { return myNails; }, ctr: () => { const b = stillRGBA(); return myNails.map((n) => +maskContrast(b.rgba, b.w, b.h, n.mask).toFixed(1)); } };

// ---- 사진 속에 내 얼굴(face-swap.ts): 고른 사진은 그대로, 얼굴만 내 얼굴로 바꾼 뒤 그 사진의 화장을 다시 입힌다 ----
const btnSwapEl = $<HTMLButtonElement>('btn-swap');
let swapMode = false;
let swapSmall = false;
let swapCache: { mine: ImageBitmap; ref: ImageBitmap; bmp: ImageBitmap } | null = null;
/** 그림을 긴 변 maxSide 이하 캔버스로 */
function toCanvas(img: ImageBitmap, maxSide: number): HTMLCanvasElement {
  const s = Math.min(1, maxSide / Math.max(img.width, img.height));
  const cv = document.createElement('canvas');
  cv.width = Math.round(img.width * s);
  cv.height = Math.round(img.height * s);
  cv.getContext('2d')!.drawImage(img, 0, 0, cv.width, cv.height);
  return cv;
}
/** 사진 속 얼굴 피부·이목구비 확률(캔버스 크기). hairPad: 머리카락 둘레를 이만큼(분할 칸) 넓혀 뺀다. 이마에 걸친 가는 가닥이 금처럼 옮겨지지 않게 */
/**
 * 얼굴 둘레 피부 지도(0~1). trustFeatures: 이목구비 둘레(눈·눈썹·눈꺼풀·입술·콧구멍, 얼굴 폭 4% 흐림)는 분할이 '피부 아님'
 * (속눈썹·쌍꺼풀 그늘을 머리카락으로)이라 해도 피부로 믿는다(안경 테 자리는 빼고): 얼굴 바꾸기에서 그 자리를 틈으로 보고
 * 둘레 피부로 메워 눈꺼풀 위에 평평하고 들쭉날쭉한 살색 조각이 생겼다
 */
async function skinOf(c: HTMLCanvasElement, pts: { x: number; y: number }[], hairPad = 0, trustFeatures = false): Promise<Float32Array | null> {
  const w = Math.hypot(pts[454].x - pts[234].x, pts[454].y - pts[234].y);
  const cx = (pts[234].x + pts[454].x) / 2, cy = (pts[10].y + pts[152].y) / 2;
  const size = w * 2.2;
  const rect = { x: cx - size / 2, y: cy - size / 2, size };
  try {
    const p = await parseFace(c, rect, setStatus);
    const face = new Float32Array(p.size * p.size);
    for (let i = 0; i < face.length; i++) face[i] = p.face[i] / 255;
    // 안경 테·코받침 자리는 피부가 아니다: 얇은 테는 분할이 피부로 보기도 해 콧등 위에 늘어난 네모 조각으로 남았다
    // → 안경 확률만큼 빼고(2px 넓혀) 얼굴 바꾸기에서 가는 틈 메우기가 둘레 피부로 채우게 한다
    // (안경 확률은 알까지 덮는다: 눈 둘레(눈 폭 1.5배로 넓힘)는 알 너머로 보이는 내 눈·피부라 남기고 테·코받침·다리만 뺀다)
    // 눈 둘레(눈 폭 1.5배·높이 1.8배): 안경알 너머로 보이는 내 눈·피부라 안경 확률과 무관하게 남긴다
    let kb: Float32Array | null = null;
    if (p.glasses) {
      const keep = new Float32Array(p.size * p.size);
      const k = p.size / rect.size;
      for (const ids of [[...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)]]) {
        const q = ids.map((j) => pts[j]);
        const ex = q.reduce((t, v) => t + v.x, 0) / q.length, ey = q.reduce((t, v) => t + v.y, 0) / q.length;
        const m = fillPoly(q.map((v) => ({ x: (ex + (v.x - ex) * 1.5 - rect.x) * k, y: (ey + (v.y - ey) * 1.8 - rect.y) * k })), p.size, p.size);
        for (let i = 0; i < keep.length; i++) if (m[i]) keep[i] = 1;
      }
      kb = boxBlur(keep, p.size, p.size, Math.max(1, Math.round(p.size * 0.01)));
      const g = new Float32Array(p.size * p.size);
      for (let i = 0; i < g.length; i++) g[i] = p.glasses[i] / 255;
      const gb = boxBlur(g, p.size, p.size, 2);
      // 안경 확률은 알 전체를 덮는다(분할의 '안경' 클래스). 테·코받침은 알 영역의 가장자리이므로 가장자리에서 2% 넘게 안쪽은
      // 투명 안경이면 알 너머 내 피부·눈이다 → 피부로 되살린다. 선글라스(알 속 밝기가 둘레 피부의 60% 아래)는 그대로 뺀다
      // (가는 테 안경 얼굴을 바꾸면 알 영역 전체가 빠져 눈이 고른 사진 눈으로 남고, 콧대 둘레는 메울 피부가 없어 밝은 네모 조각이 줄지어 남았다)
      const g1 = new Float32Array(p.size * p.size);
      for (let i = 0; i < g.length; i++) g1[i] = g[i] > 0.5 ? 1 : 0;
      const gin = boxBlur(g1, p.size, p.size, Math.max(2, Math.round(p.size * 0.02)));
      const oc = document.createElement('canvas');
      oc.width = p.size; oc.height = p.size;
      oc.getContext('2d')!.drawImage(c, rect.x, rect.y, rect.size, rect.size, 0, 0, p.size, p.size);
      const px = oc.getContext('2d')!.getImageData(0, 0, p.size, p.size).data;
      let li = 0, ni = 0, ls = 0, ns = 0;
      for (let i = 0; i < g.length; i++) {
        const l = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
        if (gin[i] >= 0.97) { li += l; ni++; } else if (face[i] > 0.8 && g[i] < 0.2) { ls += l; ns++; }
      }
      const clear = ni > 20 && ns > 20 && li / ni > 0.6 * (ls / ns) ? 1 : 0;
      for (let i = 0; i < face.length; i++) {
        const inner = clear * Math.max(0, Math.min(1, (gin[i] - 0.9) / 0.07));
        face[i] *= 1 - Math.min(1, gb[i] * 3) * (1 - Math.min(1, kb[i])) * (1 - inner);
        if (inner > face[i]) face[i] = inner;
      }
      // 분할이 못 잡는 가는 금속 브리지·코받침(밝은 반짝임이나 어두운 선, 1~2px): 안경 둘레(알 영역에서 3% 안)의
      // 가는 능선(헤시안 한 방향만 큼, 밝은 선·어두운 선 모두)을 피부에서 빼 얼굴 바꾸기의 틈 메우기가 둘레 피부로 채우게 한다
      // 눈 둘레(속눈썹·쌍꺼풀 선)는 뺀다. (콧대 위에 밝은 네모 조각이 줄지어 남았음)
      const near = boxBlur(g1, p.size, p.size, Math.max(2, Math.round(p.size * 0.03)));
      const lum = new Float32Array(p.size * p.size);
      for (let i = 0; i < lum.length; i++) lum[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
      const S = boxBlur(lum, p.size, p.size, 1);
      const ridge = new Float32Array(p.size * p.size);
      const n = p.size;
      for (let y = 2; y < n - 2; y++) for (let x = 2; x < n - 2; x++) {
        const i = y * n + x;
        if (near[i] < 0.02 || kb[i] > 0.5) continue;
        const hxx = S[i + 2] - 2 * S[i] + S[i - 2], hyy = S[i + 2 * n] - 2 * S[i] + S[i - 2 * n];
        const hxy = (S[i + 2 + 2 * n] - S[i + 2 - 2 * n] - S[i - 2 + 2 * n] + S[i - 2 - 2 * n]) / 4;
        const tr = hxx + hyy, dd = Math.sqrt((hxx - hyy) ** 2 + 4 * hxy * hxy);
        const l1 = (tr + dd) / 2, l2 = (tr - dd) / 2;
        if ((l1 > 10 && Math.abs(l2) < l1 * 0.35) || (l2 < -10 && Math.abs(l1) < -l2 * 0.35)) ridge[i] = 1;
      }
      const rb2 = boxBlur(ridge, p.size, p.size, 1);
      for (let i = 0; i < face.length; i++) if (rb2[i] > 0.1) face[i] *= 1 - Math.min(1, rb2[i] * 3);
    }
    if (hairPad > 0) {
      const hair = new Float32Array(p.size * p.size);
      for (let i = 0; i < hair.length; i++) hair[i] = p.hair[i] / 255;
      const hb = boxBlur(hair, p.size, p.size, hairPad);
      for (let i = 0; i < face.length; i++) face[i] *= 1 - Math.min(1, hb[i] * 4);
    }
    if (trustFeatures) {
      const feat = new Float32Array(p.size * p.size);
      const toP = (i: number) => ({ x: ((pts[i].x - rect.x) / rect.size) * p.size, y: ((pts[i].y - rect.y) / rect.size) * p.size });
      for (const ids of [[...EYE_R_UPPER, ...EYE_R_LOWER.slice(1, -1)], [...EYE_L_UPPER, ...EYE_L_LOWER.slice(1, -1)], [...BROW_R_LOWER, ...BROW_R_UPPER], [...BROW_L_LOWER, ...BROW_L_UPPER], [...BROW_R_LOWER, ...[...EYE_R_UPPER].reverse()], [...BROW_L_LOWER, ...[...EYE_L_UPPER].reverse()], LIPS_OUTER, [64, 98, 240, 2, 460, 327, 294, 4]]) {
        const fm = fillPoly(ids.map(toP), p.size, p.size);
        for (let i = 0; i < fm.length; i++) if (fm[i]) feat[i] = 1;
      }
      const fb = boxBlur(feat, p.size, p.size, Math.max(2, Math.round((w / size) * p.size * 0.04)));
      // 안경 확률이 높은 곳(테·코받침)은 빼되, 눈 둘레는 안경알 너머 내 눈이라 그대로 인정한다
      // (안경 쓴 얼굴은 알 전체가 안경 확률 ≥ 77 이라 눈·눈꺼풀이 내 피부에서 빠져, 바뀐 얼굴에 고른 사진 눈이 그대로 남고
      //  콧대 둘레는 메울 내 피부가 없어 고른 사진 피부가 비친 밝은 네모 조각이 줄지어 남았다)
      for (let i = 0; i < face.length; i++) if (fb[i] > 0.3 && (!p.glasses || p.glasses[i] < 77 || (kb && kb[i] > 0.5))) face[i] = 1;
    }
    const m = new Float32Array(c.width * c.height);
    for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
      const u = Math.floor(((x - rect.x) / rect.size) * p.size), v = Math.floor(((y - rect.y) / rect.size) * p.size);
      if (u >= 0 && v >= 0 && u < p.size && v < p.size) m[y * c.width + x] = face[v * p.size + u];
    }
    return m;
  } catch (err) {
    console.warn('얼굴 분할 실패(윤곽만 사용)', err);
    return null;
  }
}

/** 사진처럼 서버의 전용 얼굴 바꾸기 모델(inswapper)이 켜져 있으면 그걸로: 얼굴을 다시 그려 코·입이 무너지지 않는다 */
async function makeSwapServer(mine: ImageBitmap, ref: ImageBitmap): Promise<ImageBitmap | null> {
  const h = await checkGenServer();
  if (!h?.realistic?.swap) return null;
  setStatus('AI 생성 서버의 얼굴 바꾸기 모델로 그리는 중…');
  try {
    const res = await swapFaces(toDataUrl(bitmapToImageData(mine), 1600), toDataUrl(bitmapToImageData(ref), 2400));
    const blob = await (await fetch(res.image)).blob();
    return await createImageBitmap(blob);
  } catch (e) {
    setStatus(`서버 얼굴 바꾸기를 못 해서(${(e as Error).message}) 브라우저 방식으로 바꿔요`);
    await new Promise((r) => setTimeout(r, 30));
    return null;
  }
}
async function makeSwap(mine: ImageBitmap, ref: ImageBitmap): Promise<ImageBitmap | string> {
  if (swapCache?.mine === mine && swapCache.ref === ref) return swapCache.bmp;
  const srv = await makeSwapServer(mine, ref);
  if (srv) {
    swapSmall = false;
    swapCache = { mine, ref, bmp: srv };
    return srv;
  }
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  setStatus('두 사진의 얼굴을 찾는 중…');
  const mc = toCanvas(mine, 1600), rc = toCanvas(ref, 2400);
  const [mp, rp] = [await photoAnalyzer.faceLandmarks(mc), await photoAnalyzer.faceLandmarks(rc)];
  if (!mp) return '내 사진에서 얼굴을 찾지 못했어요';
  if (!rp) return '고른 사진에서 얼굴을 찾지 못했어요(얼굴이 보이는 사진이어야 해요)';
  // 두 사진 얼굴 주변을 정밀 분할: 고른 사진의 앞머리·안경은 그대로 두고, 내 앞머리는 옮기지 않으려고
  setStatus('얼굴 자리를 나누는 중…');
  const fw = Math.hypot(rp[454].x - rp[234].x, rp[454].y - rp[234].y);
  const skin = await skinOf(rc, rp);
  const mySkin = await skinOf(mc, mp, 5, true);
  const hands = await photoAnalyzer.bodySkin(rc).catch(() => null);
  setStatus('얼굴을 바꾸는 중…');
  await new Promise((r) => setTimeout(r, 30));
  const md = mc.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, mc.width, mc.height).data;
  const rd = rc.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, rc.width, rc.height).data;
  let out = swapFace(rd, rc.width, rc.height, rp, md, mc.width, mc.height, mp, FACE_OVAL, skin, hands, mySkin);
  // 바뀐 얼굴은 고른 사진 얼굴 자리 그대로라, 고른 사진 화장(립 색·아이라인·블러셔 모양)을 같은 점으로 바로 옮긴다
  // (실시간 화장을 얹으면 고른 사진 화장 위에 둥근 블러셔가 겹쳐 무거워진다)
  setStatus('그 사진의 화장을 바뀐 얼굴에 옮기는 중…');
  await new Promise((r) => setTimeout(r, 30));
  // 사진 분석이 화장을 못 읽었으면(전신 사진처럼 얼굴이 작아 화장 칩이 없음) 진하기만큼 모두 옮긴다:
  // 얼굴 바꾸기는 그 사진 얼굴의 화장 차이를 그대로 옮기므로 잰 색이 필요 없다(안 하면 빨간 립·아이라인이 빠져 창백했다)
  const v = Math.min(1, strength);
  const amts = lastStyle?.makeup ? mkAmounts() : { lip: v, eye: v, skin: v };
  out = transferMakeup(out, rc.width, rc.height, rp, rd, rc.width, rc.height, rp, FACE_OVAL, amts, skin, skin, hands, undefined, 1);
  const bmp = await createImageBitmap(new ImageData(out as unknown as Uint8ClampedArray<ArrayBuffer>, rc.width, rc.height));
  const myW = Math.hypot(mp[454].x - mp[234].x, mp[454].y - mp[234].y);
  swapSmall = fw / Math.max(1, myW) > 2;
  swapCache = { mine, ref, bmp };
  return bmp;
}
async function setSwap(on: boolean): Promise<void> {
  if (!PHOTO_MODE || !photoOriginal) return;
  if (on) {
    if (!lastSource) {
      setStatus('먼저 📷 사진 고르기로 얼굴이 보이는 사진을 골라 주세요');
      return;
    }
    btnSwapEl.disabled = true;
    try {
      const r = await makeSwap(photoOriginal, lastSource);
      if (typeof r === 'string') {
        setStatus(r);
        return;
      }
      swapMode = true;
      // 고른 사진의 머리·네일·타투는 이미 그 사진에 있으므로 끈다. 화장은 바뀐 내 얼굴에 다시 입힌다
      engine.tracker.setStill(null);
      engine.tracker.hiW = 1024;
      await begin(() => engine.source.openImage(r, 2400));
      for (const k of ['hair', 'nail', 'tattoo', 'hairgen'] as Chip[]) off.add(k);
      applyAll();
      renderChips();
      // 내 사진 얼굴이 작으면(전신 사진 등) 크게 늘려 흐려진다 → 안내
      setStatus(swapSmall ? '고른 사진 속에 내 얼굴을 넣었어요. 내 사진 얼굴이 작아 흐릴 수 있어요: 얼굴이 크게 나온 사진을 쓰면 선명해요' : '고른 사진 속에 내 얼굴을 넣고 그 사진의 화장을 입혔어요');
    } finally {
      btnSwapEl.disabled = false;
    }
  } else {
    swapMode = false;
    engine.tracker.setStill(null);
    mineHair = null;
    stillHairKey = '';
    await begin(() => engine.source.openImage(photoOriginal!, 2400));
    for (const k of ['hair', 'nail', 'tattoo'] as Chip[]) off.delete(k);
    applyAll();
    renderChips();
    setStatus('');
  }
  btnSwapEl.textContent = swapMode ? '↩ 내 사진에 입히기' : '🔁 사진 속에 내 얼굴';
  btnSwapEl.title = swapMode ? '내 사진에 고른 사진의 스타일을 입히는 방식으로' : '고른 사진(머리·화장·옷·배경)은 그대로 두고 얼굴만 내 얼굴로';
}
btnSwapEl.addEventListener('click', () => void setSwap(!swapMode));

async function openMyPhoto(f: File): Promise<void> {
  if (!f.type.startsWith('image/')) {
    setStatus('사진 파일(JPG·PNG)을 골라 주세요');
    return;
  }
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(f, { imageOrientation: 'from-image' });
  } catch (err) {
    setStatus(`사진을 읽지 못했어요: ${(err as Error).message}`);
    return;
  }
  // 사진은 거울처럼 뒤집지 않는다
  $<HTMLInputElement>('opt-mirror').checked = false;
  applyView();
  engine.tracker.setStill(null);
  photoOriginal = bmp;
  swapMode = false;
  btnSwapEl.textContent = '🔁 사진 속에 내 얼굴';
  mineHair = null;
  myNails = [];
  nailsAuto = null;
  stillHairKey = '';
  // 사진은 시간이 넉넉하므로 크게 처리한다(분할 합성 1024, 원본 긴 변 2400 까지)
  engine.tracker.hiW = 1024;
  await begin(() => engine.source.openImage(bmp, 2400));
  applyAll();
  if (!running) return;
  btnMyPhotoEl.hidden = false;
  // 얼굴 바꾸기는 제품 기능이 아니다(헤어·네일·메이크업·타투만 옮긴다): 시험용으로 ?swap=1 일 때만 보인다
  btnSwapEl.hidden = params.get('swap') !== '1';
  setStatus(lastStyle ? '' : '이제 📷 사진 고르기로 입히고 싶은 사진을 골라 주세요');
}
myPhotoFileEl.addEventListener('change', () => {
  const f = myPhotoFileEl.files?.[0];
  myPhotoFileEl.value = '';
  if (f) void openMyPhoto(f);
});
$('btn-myphoto-pick').addEventListener('click', () => myPhotoFileEl.click());
btnMyPhotoEl.addEventListener('click', () => myPhotoFileEl.click());
if (PHOTO_MODE) {
  document.title = 'IRIS 사진 합성';
  $('start-title').textContent = '내 사진 + 스타일 사진 합성';
  $('start-desc').textContent = '① 내 얼굴 사진을 올리고 ② 마음에 드는 메이크업 · 헤어 · 네일 · 타투 사진을 고르면 내 사진에 합쳐 줍니다. AI 생성 서버가 켜져 있으면 ✨ AI 생성으로 머리 모양까지 사진처럼 바꿀 수 있어요. 사진은 끌어다 놓거나 붙여 넣어도 됩니다.';
  $('start-cam').hidden = true;
  $('start-photo').hidden = false;
  const link = $<HTMLAnchorElement>('mode-link').querySelector('a')!;
  link.href = './';
  link.textContent = '📷 웹캠으로 하기 →';
  $('pick-hint').textContent = '입히고 싶은 메이크업 · 헤어 · 네일 · 타투 사진을 골라 주세요';
  $('gen-live').textContent = '▶ 내 사진에 입히기';
}

// ---- AI 생성(사진 한 장): 생성 서버가 켜져 있을 때만 단추가 보인다 ----
const genEl = $<HTMLElement>('gen');
const genStatusEl = $<HTMLElement>('gen-status');
const genCatEl = $<HTMLSelectElement>('gen-cat');
const genEngineEl = $<HTMLSelectElement>('gen-engine');
const genHairModelEl = $<HTMLSelectElement>('gen-hair-model');
/** 헤어 2(Stable-Hair) 서버가 켜져 있는지(마지막 확인) */
let hair2On = false;
const genGrowEl = $<HTMLInputElement>('gen-grow');
const genExtendEl = $<HTMLInputElement>('gen-extend');
const genDescEl = $<HTMLInputElement>('gen-desc');
const genUseRefEl = $<HTMLInputElement>('gen-use-ref');
const genRunEl = $<HTMLButtonElement>('gen-run');
const genResultEl = $<HTMLElement>('gen-result');
const genBeforeEl = $<HTMLImageElement>('gen-before');
const genAfterEl = $<HTMLImageElement>('gen-after');
const genSliderEl = $<HTMLInputElement>('gen-slider');
const genInfoEl = $<HTMLElement>('gen-info');
const genSaveEl = $<HTMLAnchorElement>('gen-save');
const genUrlEl = $<HTMLInputElement>('opt-gen-url');
genUrlEl.value = genServerUrl();
let genUrlTimer = 0;
const applyGenUrl = (): void => {
  const v = genUrlEl.value.trim().replace(/\/+$/, '');
  if (v && v !== genServerUrl()) setGenServerUrl(v);
  void refreshGenStatus();
};
genUrlEl.addEventListener('change', applyGenUrl);
// 붙여넣기만 하고 창을 닫아도 바로 확인되게
genUrlEl.addEventListener('input', () => {
  clearTimeout(genUrlTimer);
  genUrlTimer = window.setTimeout(applyGenUrl, 800);
});
// 서버를 나중에 켜도 단추가 나타나게: 단추가 숨어 있는 동안 15초마다 다시 확인
setInterval(() => {
  if (btnGenEl.hidden && genEl.hidden) void refreshGenStatus();
}, 15000);

async function refreshGenStatus(): Promise<boolean> {
  genStatusEl.textContent = '확인 중…';
  $('gen-use-ref-row').hidden = !lastStyle;
  const h = await checkGenServer();
  btnGenEl.hidden = !h;
  if (!h) {
    genStatusEl.textContent = '지금은 쓸 수 없어요(AI 생성 서버가 꺼져 있음)';
    genRunEl.disabled = true;
    return false;
  }
  genStatusEl.textContent = h.dry_run ? '시험 모드' : h.model_loaded ? '준비됨' : '준비됨 · 첫 생성은 조금 더 걸려요';
  // 꺼진 엔진은 고를 수 없게. 사진처럼 서버가 켜져 있으면 헤어·메이크업은 그 엔진을 기본으로
  const real = h.realistic ?? null;
  const opt = (v: string): HTMLOptionElement | null => genEngineEl.querySelector(`option[value="${v}"]`);
  const o = opt('anydoor');
  if (o) o.disabled = !h.anydoor;
  const r = opt('realistic');
  if (r) r.disabled = !real;
  if (!h.anydoor && genEngineEl.value === 'anydoor') genEngineEl.value = 'default';
  if (!real && genEngineEl.value === 'realistic') genEngineEl.value = 'default';
  if (real && !genEngineEl.dataset.touched && (genCatEl.value === 'hair' || genCatEl.value === 'makeup')) genEngineEl.value = 'realistic';
  hair2On = !!real?.hair2;
  const sh = genHairModelEl.querySelector<HTMLOptionElement>('option[value="stablehair"]');
  if (sh) sh.disabled = !hair2On;
  if (!hair2On && genHairModelEl.value === 'stablehair') genHairModelEl.value = 'auto';
  genEngineEl.title = real ? `사진처럼 서버 연결됨(헤어 ${real.hair ? '○' : '×'} · 헤어 2 ${real.hair2 ? '○' : '×'} · 메이크업 ${real.makeup ? '○' : '×'})` : h.anydoor ? 'AnyDoor 연결됨' : '사진처럼·AnyDoor 서버는 지금 꺼져 있어요';
  syncGenCategory();
  genRunEl.disabled = false;
  return true;
}

/** 생성 대상에 맞춰 설명(영어)과 옵션 줄을 채운다 */
function syncGenCategory(): void {
  const cat = genCatEl.value as GenCategory;
  // 메이크업은 사진처럼 엔진 전용, 헤어 모양은 사진처럼 엔진이면 길이 조절이 없다
  const real = genEngineEl.value === 'realistic' || cat === 'makeup';
  if (cat === 'makeup' && genEngineEl.value !== 'realistic') genEngineEl.value = 'realistic';
  $('gen-grow-row').hidden = cat !== 'hair' || real;
  $('gen-hair-model-row').hidden = cat !== 'hair' || !real;
  $('gen-extend-row').hidden = cat !== 'nail';
  $('gen-use-ref-row').hidden = !lastStyle || real;
  genRunEl.textContent = real ? '📸 지금 모습에 사진처럼 옮기기' : '📸 지금 모습으로 생성';
  // 스타일 AI가 읽은 속성을 영어 설명으로(사용자가 고칠 수 있다)
  const auto = cat === 'makeup' ? '' : promptFromStyleAI(lastStyle?.ai?.raw, cat);
  if (auto && !genDescEl.dataset.edited) genDescEl.value = auto;
  if (cat === 'nail' && lastStyle?.nail) genExtendEl.value = String(nailLength);
}
genCatEl.addEventListener('change', syncGenCategory);
genEngineEl.addEventListener('change', () => {
  genEngineEl.dataset.touched = '1';
  syncGenCategory();
});
genDescEl.addEventListener('input', () => {
  genDescEl.dataset.edited = genDescEl.value ? '1' : '';
});
btnGenEl.addEventListener('click', () => {
  genEl.hidden = !genEl.hidden;
  if (!genEl.hidden) {
    // 고른 사진에서 찾은 것을 기본 대상으로
    genCatEl.value = lastStyle?.tattoo ? 'tattoo' : lastStyle?.nail ? 'nail' : lastStyle?.makeup && !lastStyle.hair ? 'makeup' : 'hair';
    syncGenCategory();
    void refreshGenStatus();
  }
});
$('gen-close').addEventListener('click', () => {
  genEl.hidden = true;
});

// ---- 지금 화면을 사진으로 저장: 사진을 골랐으면 [고른 사진 | 원래 나 | 적용된 나] 비교 한 장 ----
/** 효과를 잠시 모두 끄고 한 장 찍는다(원래 모습) */
async function captureNatural(): Promise<ImageData> {
  // 사진 모드: 염색을 사진에 구워 넣었으므로 원래 사진을 그대로
  if (PHOTO_MODE && photoOriginal && engine.source.kind === 'image') {
    const cv = document.createElement('canvas');
    cv.width = engine.source.width;
    cv.height = engine.source.height;
    const cx = cv.getContext('2d')!;
    // 비율을 지켜 가운데에(얼굴 바꾸기 모드는 화면이 고른 사진 크기라 내 사진과 비율이 다르다)
    const s = Math.min(cv.width / photoOriginal.width, cv.height / photoOriginal.height);
    const dw = photoOriginal.width * s, dh = photoOriginal.height * s;
    cx.fillStyle = '#111';
    cx.fillRect(0, 0, cv.width, cv.height);
    cx.drawImage(photoOriginal, (cv.width - dw) / 2, (cv.height - dh) / 2, dw, dh);
    return cx.getImageData(0, 0, cv.width, cv.height);
  }
  const saved = { look: engine.look, hair: engine.hair, nail: engine.nail, tattoo: engine.tattoo };
  engine.look = {};
  engine.hair = null;
  engine.nail = null;
  engine.tattoo = null;
  try {
    return (await engine.captureNext()).image;
  } finally {
    Object.assign(engine, saved);
  }
}
function frameCanvas(img: ImageData): HTMLCanvasElement {
  const cv = document.createElement('canvas');
  cv.width = img.width;
  cv.height = img.height;
  const ctx = cv.getContext('2d')!;
  const tmp = document.createElement('canvas');
  tmp.width = img.width;
  tmp.height = img.height;
  tmp.getContext('2d')!.putImageData(img, 0, 0);
  if (viewEl.classList.contains('mirror')) {
    ctx.translate(cv.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(tmp, 0, 0);
  // 생성된 머리 스티커(2D 오버레이)도 함께
  const ov = $<HTMLCanvasElement>('overlay');
  if (hairSticker && !off.has('hairgen') && ov.width === cv.width && ov.height === cv.height) ctx.drawImage(ov, 0, 0);
  return cv;
}
function download(cv: HTMLCanvasElement, name: string): void {
  const a = document.createElement('a');
  a.href = cv.toDataURL('image/jpeg', 0.92);
  a.download = `${name}-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.jpg`;
  a.click();
}
btnSnapEl.addEventListener('click', async () => {
  if (!running) return;
  btnSnapEl.disabled = true;
  try {
    const after = frameCanvas((await engine.captureNext()).image);
    if (!lastStyle) {
      download(after, 'iris');
      setStatus('사진을 저장했어요');
    } else {
      const before = frameCanvas(await captureNatural());
      const ref = lastStyle.thumb;
      const H = after.height;
      const refW = Math.round((ref.width / ref.height) * H);
      const gap = Math.round(H * 0.015);
      const label = Math.round(H * 0.07);
      const out = document.createElement('canvas');
      out.width = refW + after.width * 2 + gap * 2;
      out.height = H + label;
      const ctx = out.getContext('2d')!;
      ctx.fillStyle = '#101216';
      ctx.fillRect(0, 0, out.width, out.height);
      const panels: [CanvasImageSource, number, number, string][] = [
        [ref, 0, refW, '고른 사진'],
        [before, refW + gap, after.width, '원래 모습'],
        [after, refW + gap + after.width + gap, after.width, 'IRIS 적용'],
      ];
      ctx.font = `600 ${Math.round(label * 0.5)}px -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (const [img, x, w, text] of panels) {
        ctx.drawImage(img, x, 0, w, H);
        ctx.fillStyle = '#e9ecf1';
        ctx.fillText(text, x + w / 2, H + label / 2);
      }
      download(out, 'iris-비교');
      setStatus('전후 비교 사진을 저장했어요');
    }
  } finally {
    btnSnapEl.disabled = false;
  }
  setTimeout(() => {
    if (statusEl.textContent?.endsWith('저장했어요')) setStatus('');
  }, 2500);
});
genSliderEl.addEventListener('input', () => {
  genAfterEl.style.clipPath = `inset(0 0 0 ${genSliderEl.value}%)`;
});

/** ImageBitmap → ImageData(생성 서버로 보낼 때) */
function bitmapToImageData(bmp: ImageBitmap): ImageData {
  const cv = document.createElement('canvas');
  cv.width = bmp.width;
  cv.height = bmp.height;
  const ctx = cv.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0);
  return ctx.getImageData(0, 0, cv.width, cv.height);
}

/** 마지막 생성 결과(자동 시험용) */
let lastGen: { elapsed_ms: number; prompt: string; model: string } | null = null;
/** 실시간에 입히기용: 생성에 쓴 캡처의 얼굴 점(생성 사진 좌표)과 결과 */
let lastGenLive: { category: GenCategory; image: string; mask: string; facePts: { x: number; y: number; z?: number }[] | null } | null = null;
const genLiveEl = $<HTMLButtonElement>('gen-live');

// ---- 생성된 머리 스티커: 생성 결과의 머리카락 부분을 얼굴 점에 붙여 두고, 고개 움직임을 따라 같이 움직인다 ----
/** 얼굴에서 잘 안 움직이는 점들(윤곽·콧대)로 닮음 변환(크기·회전·이동)을 구한다 */
const STICKER_PTS = [...FACE_OVAL, 1, 4, 6, 168, 197, 5, 195];
/** p0: 캡처 때 얼굴 점(2D 닮음 변환용), p3·mesh: 깊이가 있으면 3D 로 휘기(고개를 돌려도 머리 모양 유지) */
let hairSticker: { img: ImageBitmap; p0: { x: number; y: number }[]; p3: Vec3[] | null; mesh: WarpMesh | null } | null = null;
let stickerGL: StickerGL | null = null;
let stickerXY: Float32Array | undefined;
/** 고개 자세를 조금 부드럽게(깊이 떨림이 회전으로 번지지 않게) */
let stickerPose: Pose3 | null = null;

function similarity(from: { x: number; y: number }[], to: { x: number; y: number }[]): { a: number; b: number; tx: number; ty: number } | null {
  const n = Math.min(from.length, to.length);
  if (n < 3) return null;
  let cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;
  for (let i = 0; i < n; i++) {
    cx0 += from[i].x; cy0 += from[i].y; cx1 += to[i].x; cy1 += to[i].y;
  }
  cx0 /= n; cy0 /= n; cx1 /= n; cy1 /= n;
  let num = 0, cross = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dx = from[i].x - cx0, dy = from[i].y - cy0, ex = to[i].x - cx1, ey = to[i].y - cy1;
    num += dx * ex + dy * ey;
    cross += dx * ey - dy * ex;
    den += dx * dx + dy * dy;
  }
  if (den < 1e-6) return null;
  const a = num / den, b = cross / den;
  return { a, b, tx: cx1 - (a * cx0 - b * cy0), ty: cy1 - (b * cx0 + a * cy0) };
}

async function buildSticker(image: string, mask: string): Promise<{ bmp: ImageBitmap; box: { x0: number; y0: number; x1: number; y1: number } | null }> {
  const load = (src: string): Promise<HTMLImageElement> => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
  const [img, m] = await Promise.all([load(image), load(mask)]);
  const w = img.naturalWidth, h = img.naturalHeight;
  const mc = document.createElement('canvas');
  mc.width = w; mc.height = h;
  const mctx = mc.getContext('2d')!;
  mctx.filter = `blur(${Math.max(2, Math.round(Math.min(w, h) * 0.012))}px)`;
  mctx.drawImage(m, 0, 0, w, h);
  const alpha = mctx.getImageData(0, 0, w, h).data;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, w, h);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let i = 0; i < alpha.length; i += 4) {
    const a = alpha[i];
    d.data[i + 3] = a;
    if (a > 4) {
      const px = (i >> 2) % w, py = (i >> 2) / w | 0;
      if (px < x0) x0 = px;
      if (px > x1) x1 = px;
      if (py < y0) y0 = py;
      if (py > y1) y1 = py;
    }
  }
  ctx.putImageData(d, 0, 0);
  return { bmp: await createImageBitmap(c), box: x1 >= x0 ? { x0, y0, x1: x1 + 1, y1: y1 + 1 } : null };
}

/** 3D 로 휘어 그리기. 깊이가 없거나 WebGL2 가 안 되면 false(평면 스티커로) */
function drawSticker3D(ctx: CanvasRenderingContext2D, face: NonNullable<Parameters<NonNullable<typeof engine.overlayDraw>>[1]>, w: number, h: number): boolean {
  const st = hairSticker;
  if (!st?.p3 || !st.mesh || !face.z) return false;
  const now: Vec3[] = STICKER_PTS.map((i) => ({ x: face.p[i].x, y: face.p[i].y, z: face.z![i] }));
  const P = similarity3(st.p3, now);
  if (!P) return false;
  if (!stickerPose) stickerPose = P;
  else {
    // 회전·크기만 가볍게 거른다(이동은 거르면 늦어 보인다)
    const k = 0.6;
    const R = stickerPose.R.map((r0, i) => r0 + (P.R[i] - r0) * k);
    stickerPose = { s: stickerPose.s + (P.s - stickerPose.s) * k, R, t: P.t };
  }
  // 거른 회전으로 이동을 다시 맞춘다(중심이 어긋나지 않게)
  const Pf: Pose3 = { ...stickerPose, t: { x: 0, y: 0, z: 0 } };
  let ox = 0, oy = 0;
  for (let i = 0; i < now.length; i++) {
    const q = applyPose(Pf, st.p3[i]);
    ox += now[i].x - q.x;
    oy += now[i].y - q.y;
  }
  Pf.t = { x: ox / now.length, y: oy / now.length, z: 0 };
  const deg = rotationDeg(P.R);
  // 많이 돌리면 붙인 머리가 어색해진다 → 35°~55° 에서 서서히 사라지게
  const fade = 1 - Math.min(1, Math.max(0, (deg - 35) / 20));
  if (fade <= 0) return true;
  stickerGL ??= new StickerGL();
  if (!stickerGL.ready) stickerGL.setImage(st.img);
  stickerXY = projectMesh(st.mesh, Pf, stickerXY);
  const cut = cutMesh(FACE_CUT.map((i) => face.p[i]));
  // 정면에서는 지우지 않고(생성 머리가 뺨을 덮는 것도 살림), 돌릴수록 얼굴 자리를 지운다
  const cutA = Math.min(1, Math.max(0, (deg - 5) / 15));
  const out = stickerGL.draw(w, h, { xy: stickerXY, uv: st.mesh.uv, index: st.mesh.index }, fade * face.confidence, cut, cutA);
  ctx.drawImage(out, 0, 0);
  return true;
}

engine.overlayDraw = (ctx, face, w, h) => {
  if (!hairSticker || off.has('hairgen') || !face) return;
  try {
    if (drawSticker3D(ctx, face, w, h)) return;
  } catch (err) {
    console.warn('3D 머리 붙이기 실패 → 평면으로', err);
    hairSticker.mesh = null;
  }
  const t = similarity(hairSticker.p0, STICKER_PTS.map((i) => face.p[i]));
  if (!t) return;
  ctx.globalAlpha = face.confidence;
  ctx.setTransform(t.a, t.b, -t.b, t.a, t.tx, t.ty);
  ctx.drawImage(hairSticker.img, 0, 0);
};

/** 생성 결과를 웹캠 화면에서 계속 따라오게: 메이크업은 사진 따라하기(내 얼굴 기준이라 정확), 헤어는 머리 스티커 */
async function applyGenLive(): Promise<void> {
  const g = lastGenLive;
  if (!g) return;
  genLiveEl.disabled = true;
  try {
    if (g.category === 'hair') {
      if (!g.facePts) {
        setStatus('생성할 때 얼굴 점이 없어 따라오게 할 수 없어요. 얼굴이 보이는 상태에서 다시 생성해 주세요');
        return;
      }
      const { bmp, box } = await buildSticker(g.image, g.mask);
      const fp = g.facePts;
      const has3 = fp.length >= 468 && fp[1].z !== undefined;
      const p3: Vec3[] | null = has3 ? fp.map((q) => ({ x: q.x, y: q.y, z: q.z ?? 0 })) : null;
      const mesh = p3 && box ? buildWarpMesh(headProxy(p3), bmp.width, bmp.height, box) : null;
      hairSticker = { img: bmp, p0: STICKER_PTS.map((i) => fp[i]), p3: p3 ? STICKER_PTS.map((i) => p3[i]) : null, mesh };
      stickerPose = null;
      if (stickerGL) stickerGL.ready = false;
      off.delete('hairgen');
      // 생성된 머리가 덮으므로 실시간 염색은 끈다
      off.add('hair');
      applyAll();
      renderChips();
      setStatus('생성된 머리가 화면을 따라옵니다(정면에서 조금 움직일 때 자연스러워요)');
    } else {
      const blob = await (await fetch(g.image)).blob();
      await applyPhoto(await createImageBitmap(blob), g.category === 'makeup' ? 'makeup' : 'auto');
      setStatus('생성 결과의 화장을 실시간으로 입혔어요');
    }
    genEl.hidden = true;
  } catch (err) {
    setStatus(`실시간에 입히지 못했어요: ${(err as Error).message}`);
  } finally {
    genLiveEl.disabled = false;
  }
}
genLiveEl.addEventListener('click', () => void applyGenLive());

async function runGen(): Promise<void> {
  if (!running) {
    setStatus(PHOTO_MODE ? '먼저 내 사진을 올려 주세요' : '먼저 카메라를 시작해 주세요');
    return;
  }
  const cat = genCatEl.value as GenCategory;
  genRunEl.disabled = true;
  genStatusEl.textContent = '지금 모습을 찍는 중…';
  try {
    // 합성된 화면(실시간 메이크업 포함)을 한 장 찍는다
    const cap = await engine.captureNext();
    const image = toDataUrl(cap.image, 1024);
    // 실시간에 입히기용: 캡처의 얼굴 점을 생성 사진 좌표(최대 1024)로
    const capScale = Math.min(1, 1024 / Math.max(cap.image.width, cap.image.height));
    const capZ = cap.face?.z;
    const capFace = cap.face ? cap.face.p.map((q, i) => ({ x: q.x * capScale, y: q.y * capScale, z: capZ ? capZ[i] * capScale : undefined })) : null;
    const realistic = genEngineEl.value === 'realistic' || cat === 'makeup';
    if (realistic && !lastStyle) {
      genStatusEl.textContent = '먼저 📷 사진 고르기로 옮길 사진을 골라 주세요';
      genRunEl.disabled = false;
      return;
    }
    // 사진처럼 엔진은 고른 사진 원본(큰 것)을 보낸다: 얼굴이 클수록 잘 옮겨진다
    const reference = realistic && lastSource ? toDataUrl(bitmapToImageData(lastSource), 1024) : genUseRefEl.checked && lastStyle ? lastStyle.thumb.toDataURL('image/jpeg', 0.9) : undefined;
    // 타투: 사진에서 고른(또는 자동으로 찾은) 도안 영역을 그대로 보낸다(서버의 추정이 끈·머리카락을 고르는 것 방지)
    let reference_mask: string | undefined;
    const ts = lastStyle?.tattooSrc;
    if (reference && cat === 'tattoo' && ts && lastStyle) {
      const m = document.createElement('canvas');
      m.width = lastStyle.thumb.width;
      m.height = lastStyle.thumb.height;
      const mc = m.getContext('2d')!;
      mc.fillStyle = '#000';
      mc.fillRect(0, 0, m.width, m.height);
      mc.fillStyle = '#fff';
      mc.fillRect(ts.rect.x, ts.rect.y, ts.rect.w, ts.rect.h);
      reference_mask = m.toDataURL('image/png');
    }
    genBeforeEl.src = image;
    const hairModel = genHairModelEl.value as 'auto' | 'stablehair' | 'hairfast';
    const slowHair = hairModel === 'stablehair' && hair2On;
    const working = genEngineEl.value === 'anydoor' ? '만드는 중… (1~2분)' : realistic ? (cat === 'makeup' ? '화장을 옮기는 중… (10~30초)' : slowHair ? '머리를 지우고 새 머리를 그리는 중… (10~20초)' : '머리 모양·색을 옮기는 중… (몇 초)') : '만드는 중… (10~30초)';
    genStatusEl.textContent = working;
    const r = await genGenerate(
      {
        category: cat,
        image,
        reference,
        reference_mask,
        desc: genDescEl.value.trim() || undefined,
        place: tattoo.place,
        grow: Number(genGrowEl.value),
        extend: Number(genExtendEl.value),
        engine: realistic ? 'realistic' : genEngineEl.value === 'anydoor' ? 'anydoor' : 'default',
        hair_model: hairModel,
      },
      undefined,
      (ms, line) => {
        genStatusEl.textContent = `${working} ${Math.round(ms / 1000)}초${line ? ' · ' + line.replace(/^\d\d:\d\d:\d\d /, '') : ''}`;
      },
    );
    genAfterEl.src = r.image;
    genAfterEl.style.clipPath = `inset(0 0 0 ${genSliderEl.value}%)`;
    genSaveEl.href = r.image;
    genInfoEl.textContent = `${(r.elapsed_ms / 1000).toFixed(1)}초 · ${r.model === 'dry-run' ? '시험 모드' : r.model === 'anydoor' ? 'AnyDoor 합성' : r.model === 'realistic' ? (r.hair_model === 'stablehair' ? '사진처럼 옮김(헤어 2)' : '사진처럼 옮김') : r.model}`;
    genResultEl.hidden = false;
    genStatusEl.textContent = '완성! 막대를 움직여 전후를 비교해 보세요';
    lastGen = { elapsed_ms: r.elapsed_ms, prompt: r.prompt, model: r.model };
    lastGenLive = { category: cat, image: r.image, mask: r.mask, facePts: capFace };
    genLiveEl.hidden = !(cat === 'hair' || cat === 'makeup');
  } catch (err) {
    genStatusEl.textContent = `만들지 못했어요: ${(err as Error).message}`;
  } finally {
    genRunEl.disabled = false;
  }
}
genRunEl.addEventListener('click', () => void runGen());
(window as unknown as { irisGen: unknown }).irisGen = {
  run: runGen, get last() { return lastGen; }, status: refreshGenStatus,
  // 자동 시험용: 생성 결과를 흉내 내어 실시간에 입히기
  setLive(v: typeof lastGenLive) { lastGenLive = v; }, live: applyGenLive, get sticker() { return hairSticker; },
};

/** 되먹임 기록(자동 시험·디버그용) */
let lastMatch: Record<string, unknown>[] = [];

/**
 * 사진 되먹임: 화장 전 내 얼굴과 화장 후 내 얼굴을 같은 방법으로 재서,
 * 사진과 같은 정도가 되도록 색·진하기를 네 번 고친다.
 */
async function matchToPhoto(photo: FaceMeasure): Promise<void> {
  const saved = engine.look;
  engine.look = {};
  const nat = await engine.captureNext();
  engine.look = saved;
  if (!nat.face) return;
  const my = measureFace(nat.image, nat.face.p);
  lastMatch = [];
  // 비교 위치: 광대 자리는 머리카락이 섞여 불안정하므로 눈 밑(사과존) 또는 볼 가운데만 쓴다
  const spot = (m: FaceMeasure): RGB | null => (blushPos < 0.35 ? m.blushApple : m.blushMid);
  const natBlush = blushPos < 0.35 ? NATURAL_APPLE : NATURAL_MID;
  const darker = (m: FaceMeasure): RGB | null => [m.shadowIn, m.shadowOut].filter((x): x is RGB => !!x).sort((a, b) => luma(a) - luma(b))[0] ?? null;
  for (let it = 0; it < 4; it++) {
    const cap = await engine.captureNext();
    if (!cap.face) break;
    const now = measureFace(cap.image, cap.face.p);
    const log: Record<string, unknown> = { it };
    if (chosen.blush.color && spot(photo) && spot(now) && spot(my)) {
      const m = matchTint({ color: chosen.blush.color, amount: chosen.blush.amount }, spot(photo)!, spot(now)!, natBlush, spot(my)!);
      chosen.blush.color = m.amount > 0 ? m.color : null;
      chosen.blush.amount = m.amount > 0 ? m.amount : 0.35;
      log.blush = { want: spot(photo), got: spot(now), amount: m.amount };
    }
    const two = shadowOut && photo.shadowIn && now.shadowIn && my.shadowIn && photo.shadowOut && now.shadowOut && my.shadowOut;
    if (chosen.shadow.color && two) {
      // 두 색 섀도: 앞머리 쪽과 눈꼬리 쪽을 따로 맞춘다. 사진에 그쪽 섀도가 없으면 피부색(안 칠함)
      const mi = matchTint({ color: chosen.shadow.color, amount: chosen.shadow.amount }, photo.shadowIn!, now.shadowIn!, NATURAL_SHADOW, my.shadowIn!);
      const mo = matchTint({ color: shadowOut!, amount: chosen.shadow.amount }, photo.shadowOut!, now.shadowOut!, NATURAL_SHADOW, my.shadowOut!);
      chosen.shadow.color = mi.amount > 0 ? mi.color : SKIN_REF;
      shadowOut = mo.amount > 0 ? mo.color : SKIN_REF;
      chosen.shadow.amount = Math.max(mi.amount, mo.amount) || 0.45;
      if (mi.amount === 0 && mo.amount === 0) chosen.shadow.color = null;
      log.shadow = { wantIn: photo.shadowIn, gotIn: now.shadowIn, wantOut: photo.shadowOut, gotOut: now.shadowOut, amount: chosen.shadow.amount, color: chosen.shadow.color, colorOut: shadowOut };
    } else if (chosen.shadow.color && darker(photo) && darker(now) && darker(my)) {
      const m = matchTint({ color: chosen.shadow.color, amount: chosen.shadow.amount }, darker(photo)!, darker(now)!, NATURAL_SHADOW, darker(my)!);
      chosen.shadow.color = m.amount > 0 ? m.color : null;
      chosen.shadow.amount = m.amount > 0 ? m.amount : 0.45;
      log.shadow = { want: darker(photo), got: darker(now), my: darker(my), natural: NATURAL_SHADOW, amount: m.amount, color: chosen.shadow.color };
    }
    if (chosen.lip.color && photo.lip && now.lip) {
      const want = lin(lipTarget(photo.lip, photo.skin));
      const got = lin(lipTarget(now.lip, now.skin));
      const m = matchLip({ color: chosen.lip.color, amount: chosen.lip.amount }, want, got);
      chosen.lip.color = m.color;
      log.lip = { want, got };
    }
    if (chosen.brow.color && photo.brow && now.brow && my.brow) {
      const m = matchDarkness({ color: chosen.brow.color, amount: chosen.brow.amount }, luma(photo.brow), luma(now.brow), NATURAL_BROW_L, luma(my.brow));
      chosen.brow.amount = m.amount;
      log.brow = { want: luma(photo.brow), got: luma(now.brow), amount: m.amount };
    }
    if (chosen.liner.color && photo.linerL !== null && now.linerL !== null && my.linerL !== null) {
      const m = matchDarkness({ color: chosen.liner.color, amount: chosen.liner.amount }, photo.linerL, now.linerL, NATURAL_LINER_L, my.linerL);
      chosen.liner.amount = m.amount;
      log.liner = { want: photo.linerL, got: now.linerL, amount: m.amount };
    }
    lastMatch.push(log);
    applyLookToEngine();
  }
  renderChips();
}

/** 자동 시험용: 지금 화면을 재서 사진 측정값과 나란히 돌려준다 */
async function checkPhoto(): Promise<Record<string, unknown> | null> {
  if (!lastStyle?.measure) return null;
  const cap = await engine.captureNext();
  if (!cap.face) return null;
  const now = measureFace(cap.image, cap.face.p);
  const pick = (m: FaceMeasure): Record<string, unknown> => ({
    apple: m.blushApple, mid: m.blushMid, bone: m.blushBone, shadowIn: m.shadowIn, shadowOut: m.shadowOut,
    lip: m.lip ? lin(lipTarget(m.lip, m.skin)) : null, linerL: m.linerL, browL: m.brow ? luma(m.brow) : null,
  });
  return { photo: pick(lastStyle.measure), result: pick(now), match: lastMatch, settings: { blush: chosen.blush, shadow: { ...chosen.shadow, colorOut: shadowOut }, lip: chosen.lip, brow: chosen.brow, liner: chosen.liner, blushPos } };
}
(window as unknown as { irisPhoto: unknown }).irisPhoto = { apply: applyPhoto, get last() { return lastStyle; }, get match() { return lastMatch; }, measure: measureFace, check: checkPhoto };

// ---- 타투 자리: 화면을 눌러 고르기 ----
tattooPickEl.addEventListener('click', () => {
  tattooPicking = !tattooPicking;
  tattooPickEl.classList.toggle('on', tattooPicking);
  setStatus(tattooPicking ? '화면에서 타투를 넣을 곳을 눌러 주세요' : '');
});
viewEl.addEventListener('pointerdown', (ev) => {
  if (!tattooPicking) return;
  const cv = viewEl.querySelector('canvas')!;
  const r = cv.getBoundingClientRect();
  if (!r.width || !r.height) return;
  // 거울 모드는 CSS로 뒤집으므로 화면 좌표를 다시 뒤집는다
  let fx = (ev.clientX - r.left) / r.width;
  const fy = (ev.clientY - r.top) / r.height;
  if (viewEl.classList.contains('mirror')) fx = 1 - fx;
  const hit = engine.tattooPlaceAt(fx * cv.width, fy * cv.height);
  if (!hit) {
    setStatus('팔·다리·목·가슴 위를 눌러 주세요. 몸이 화면에 보여야 해요');
    return;
  }
  tattoo.place = hit.place;
  tattoo.along = hit.along;
  tattoo.across = hit.across;
  // 사진 모드: 자리를 직접 고르면 팔 대 팔 옮기기 대신 그 자리에 굽는다
  if (PHOTO_MODE) refTat = null;
  applyTattooToEngine();
  tattooPicking = false;
  tattooPickEl.classList.remove('on');
  setStatus(`${PLACE_LABELS[hit.place]}에 옮겼어요`);
});

// ---- 사진 위에서 영역 드래그해 고르기(타투 도안 영역, 사진 속 한 사람·한 부분) ----
type Rect = { x: number; y: number; w: number; h: number };
function openCrop(
  image: CanvasImageSource,
  w: number,
  h: number,
  init: Rect | null,
  hintText: string,
  buttons: { label: string; cls: string; run: (rect: Rect | null) => string | void }[],
): void {
  let rect: Rect | null = init;
  const ov = document.createElement('div');
  ov.className = 'crop-overlay';
  const hint = document.createElement('div');
  hint.textContent = hintText;
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d')!;
  const paint = (): void => {
    ctx.drawImage(image, 0, 0, w, h);
    if (!rect) return;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(0, 0, w, rect.y);
    ctx.fillRect(0, rect.y + rect.h, w, h - rect.y - rect.h);
    ctx.fillRect(0, rect.y, rect.x, rect.h);
    ctx.fillRect(rect.x + rect.w, rect.y, w - rect.x - rect.w, rect.h);
    ctx.strokeStyle = '#ffd34d';
    ctx.lineWidth = Math.max(2, w / 300);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
  };
  const toImg = (e: PointerEvent): { x: number; y: number } => {
    const r = cv.getBoundingClientRect();
    return { x: Math.min(w - 1, Math.max(0, ((e.clientX - r.left) / r.width) * w)), y: Math.min(h - 1, Math.max(0, ((e.clientY - r.top) / r.height) * h)) };
  };
  let start: { x: number; y: number } | null = null;
  cv.addEventListener('pointerdown', (e) => {
    cv.setPointerCapture(e.pointerId);
    start = toImg(e);
    rect = null;
  });
  cv.addEventListener('pointermove', (e) => {
    if (!start) return;
    const p = toImg(e);
    rect = { x: Math.round(Math.min(start.x, p.x)), y: Math.round(Math.min(start.y, p.y)), w: Math.round(Math.abs(p.x - start.x)), h: Math.round(Math.abs(p.y - start.y)) };
    paint();
  });
  cv.addEventListener('pointerup', () => {
    start = null;
  });
  const row = document.createElement('div');
  row.className = 'row';
  const close = (): void => ov.remove();
  for (const b of buttons) {
    const el = document.createElement('button');
    el.textContent = b.label;
    if (b.cls) el.className = b.cls;
    el.addEventListener('click', () => {
      if (b.cls === 'ghost') return close();
      const msg = b.run(rect);
      if (msg) hint.textContent = msg;
      else close();
    });
    row.append(el);
  }
  ov.append(hint, cv, row);
  document.body.append(ov);
  paint();
}

tattooCropEl.addEventListener('click', () => {
  const ts = lastStyle?.tattooSrc;
  if (!ts) return;
  const { src } = ts;
  const photo = document.createElement('canvas');
  photo.width = src.w;
  photo.height = src.h;
  photo.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(src.d), src.w, src.h), 0, 0);
  const use = (r: Rect | null): string | void => {
    const t = tattooFromSource(src, r, PHOTO_MODE ? 1600 : 512);
    // 잉크가 거의 없는 영역(배경·맨살)을 고르면 빈 도안이 되어 아무것도 안 보인다
    let inkN = 0;
    if (t) {
      const c = t.design.canvas!;
      const d = c.getContext('2d', { willReadFrequently: true })!.getImageData(0, 0, c.width, c.height).data;
      for (let i = 3; i < d.length; i += 4) if (d[i] > 64) inkN++;
    }
    if (!t || inkN < 30) return '이 영역에서 타투 선을 찾지 못했어요. 피부 위 타투를 감싸 주세요';
    tattoo.design = t.design;
    if (lastStyle) {
      lastStyle.tattoo = t.design;
      lastStyle.tattooSrc = { src, rect: t.rect, inkMask: t.inkMask, paper: ts.paper };
    }
    // 사진 모드: 고른 영역으로 팔 대 팔 옮기기를 다시(도안 시트는 실시간 타투로), 사진도 다시 굽는다
    if (PHOTO_MODE && refTat) refTat = ts.paper ? null : { ...refTat, design: t.design, rect: t.rect, user: !!r };
    off.delete('tattoo');
    applyTattooToEngine();
    if (PHOTO_MODE) scheduleStillHair();
    renderChips();
    setStatus('고른 영역으로 도안을 바꿨어요');
  };
  openCrop(photo, src.w, src.h, { ...ts.rect }, '타투 부분을 드래그해서 감싸 주세요', [
    { label: '자동으로 찾기', cls: '', run: () => use(null) },
    { label: '취소', cls: 'ghost', run: () => undefined },
    { label: '이 영역으로', cls: 'primary', run: (r) => (!r || r.w < 8 || r.h < 8 ? '먼저 타투 부분을 드래그해서 감싸 주세요' : use(r)) },
  ]);
});

// 사진 속 한 사람·한 부분만 골라 다시 읽기(여러 명, 메이크업과 헤어가 따로 있는 사진)
$('ref-crop').addEventListener('click', () => {
  const src = lastSource;
  const st = lastStyle;
  if (!src || !st) return;
  const t = st.thumb;
  openCrop(t, t.width, t.height, null, st.faces && st.faces > 1 ? `얼굴이 ${st.faces}명이에요. 원하는 사람을 드래그해서 감싸 주세요` : '원하는 사람이나 부분을 드래그해서 감싸 주세요', [
    { label: '취소', cls: 'ghost', run: () => undefined },
    {
      label: '이 부분만 읽기',
      cls: 'primary',
      run: (r) => {
        if (!r || r.w < 16 || r.h < 16) return '먼저 원하는 부분을 드래그해서 감싸 주세요';
        // 썸네일 좌표 → 원본 좌표로 키워 잘라 낸다(작은 영역도 원본 해상도로 분석)
        const k = src.width / t.width;
        const cut = document.createElement('canvas');
        cut.width = Math.max(2, Math.round(r.w * k));
        cut.height = Math.max(2, Math.round(r.h * k));
        cut.getContext('2d')!.drawImage(src, r.x * k, r.y * k, r.w * k, r.h * k, 0, 0, cut.width, cut.height);
        void createImageBitmap(cut).then((bmp) => applyPhoto(bmp, 'auto', undefined, src));
      },
    },
  ]);
});

sizeEl.addEventListener('input', () => {
  tattoo.size = Number(sizeEl.value);
  applyTattooToEngine();
});
amountEl.addEventListener('input', () => {
  strength = Number(amountEl.value);
  applyAll();
});

// ---- 시작 ----

async function begin(open: () => Promise<void>): Promise<void> {
  try {
    setStatus('영상 여는 중…');
    await open();
    startEl.hidden = true;
    await engine.tracker.configure(trackerConfig, setStatus);
    if (NAIL_MODEL_PARAM) await engine.setNailModel(NAIL_MODEL_PARAM, setStatus);
    setStatus('');
    engine.start();
    running = true;
    btnSnapEl.hidden = false;
  } catch (err) {
    const msg = err instanceof DOMException && err.name === 'NotAllowedError' ? '카메라 권한이 거부되었습니다. 브라우저 주소창에서 권한을 허용해 주세요.' : String(err);
    setStatus(msg);
    startEl.hidden = false;
  }
}

/** 앞(셀피)·뒤 카메라 */
let facing: 'user' | 'environment' = 'user';
const btnFlipEl = $<HTMLButtonElement>('btn-flip');
function cameraRequest(): { width: number; height: number; fps: number; facing: 'user' | 'environment' } {
  const [w, h] = $<HTMLSelectElement>('opt-res').value.split('x').map(Number);
  return { width: w, height: h, fps: 30, facing };
}
async function openCameraAndStart(): Promise<void> {
  await begin(() => engine.source.openCamera(cameraRequest()));
  // 카메라가 둘 이상인 기기(휴대폰)에서만 전환 단추
  try {
    const n = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput').length;
    btnFlipEl.hidden = !(running && engine.source.kind === 'camera' && n >= 2);
  } catch {
    btnFlipEl.hidden = true;
  }
}
btnFlipEl.addEventListener('click', () => {
  facing = facing === 'user' ? 'environment' : 'user';
  // 뒤 카메라는 거울이 아니다
  $<HTMLInputElement>('opt-mirror').checked = facing === 'user';
  applyView();
  void openCameraAndStart();
});
/** 느린 기기: 처리 시간이 계속 길면 한 번만 화질을 한 단계 낮춘다 */
let autoLowered = false;
setInterval(() => {
  if (autoLowered || !running || engine.source.kind !== 'camera' || engine.metrics.proc.count < 60) return;
  const res = $<HTMLSelectElement>('opt-res');
  if (engine.metrics.proc.quantile(0.5) < 110 || res.value === '640x480') return;
  autoLowered = true;
  res.value = res.value === '1920x1080' ? '1280x720' : '640x480';
  if (trackerConfig.segEvery < 2) {
    trackerConfig.segEvery = 2;
    $<HTMLSelectElement>('opt-seg-every').value = '2';
  }
  void openCameraAndStart().then(() => setStatus('이 기기에 맞춰 화질을 조금 낮췄어요(설정에서 바꿀 수 있어요)'));
}, 5000);

$('btn-camera').addEventListener('click', () => void openCameraAndStart());
$('btn-sample').addEventListener('click', () => {
  $<HTMLInputElement>('opt-mirror').checked = false;
  applyView();
  void begin(() => engine.source.openFile(SAMPLE_VIDEO));
});
$<HTMLInputElement>('file-video').addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;
  $<HTMLInputElement>('opt-mirror').checked = false;
  applyView();
  void begin(() => engine.source.openFile(file));
});

// ---- 설정 ----

function applyView(): void {
  viewEl.classList.toggle('mirror', $<HTMLInputElement>('opt-mirror').checked);
}

function applySettings(): void {
  const s = engine.settings;
  s.useSeg = $<HTMLInputElement>('opt-seg').checked;
  s.refine = $<HTMLInputElement>('opt-refine').checked;
  s.debugLandmarks = $<HTMLInputElement>('opt-dbg-lm').checked;
  s.debugSeg = $<HTMLInputElement>('opt-dbg-seg').checked;
  applyView();
}

async function applyTracker(): Promise<void> {
  trackerConfig.delegate = $<HTMLSelectElement>('opt-delegate').value as Delegate;
  trackerConfig.segEvery = Number($<HTMLSelectElement>('opt-seg-every').value);
  trackerConfig.hairModel = $<HTMLInputElement>('opt-hair-model').checked ? HAIR_MODEL_URL : '';
  if (!running) return;
  await engine.tracker.configure(trackerConfig, setStatus);
  engine.metrics.reset();
  setStatus('');
}

for (const id of ['opt-mirror', 'opt-seg', 'opt-refine', 'opt-dbg-lm', 'opt-dbg-seg']) {
  $(id).addEventListener('input', applySettings);
}
for (const id of ['opt-delegate', 'opt-seg-every', 'opt-hair-model']) {
  $(id).addEventListener('change', () => void applyTracker());
}
$('opt-res').addEventListener('change', () => {
  if (running && engine.source.kind === 'camera') void openCameraAndStart();
});

$('btn-settings').addEventListener('click', () => {
  panelEl.hidden = !panelEl.hidden;
});
$('btn-close').addEventListener('click', () => {
  panelEl.hidden = true;
});
$('btn-hud').addEventListener('click', () => {
  hudEl.hidden = !hudEl.hidden;
  $('btn-hud').classList.toggle('on', !hudEl.hidden);
});

// ---- 지연 측정 ----

$('btn-loopback').addEventListener('click', () => {
  if (!running) {
    loopbackEl.textContent = '먼저 카메라를 시작하세요.';
    return;
  }
  panelEl.hidden = true;
  loopbackEl.textContent = '측정 중… (화면이 깜빡입니다)';
  engine.startLoopback((lb) => {
    panelEl.hidden = false;
    loopbackFailures = lb.failures;
    if (lb.phase === 'failed') {
      loopbackEl.textContent = lb.message;
      return;
    }
    const s = summarize(lb.samples);
    loopbackSummary = s;
    const proc = engine.metrics.proc.quantile(0.5);
    loopbackEl.textContent = [
      `화면→카메라→브라우저 왕복: 중앙 ${fmt(s.p50)} ms · 95% ${fmt(s.p95)} ms · 최대 ${fmt(s.max)} ms (n=${s.n}, 실패 ${lb.failures})`,
      `처리 시간(중앙): ${fmt(proc)} ms`,
      `추정 전체 지연(움직임→화면) ≈ ${fmt(s.p50 + proc)} ms`,
      '※ 추정치입니다. 정확한 값은 슬로모션 촬영으로 교차 확인하세요.',
    ].join('\n');
  });
});

$('btn-export').addEventListener('click', () => {
  const data = {
    createdAt: new Date().toISOString(),
    userAgent: navigator.userAgent,
    screen: { w: screen.width, h: screen.height, dpr: devicePixelRatio },
    source: engine.source.kind,
    camera: engine.source.settings,
    video: { w: engine.source.width, h: engine.source.height },
    tracker: { ...trackerConfig, activeDelegate: engine.tracker.delegate },
    settings: engine.settings,
    look: engine.look,
    loopback: loopbackSummary ? { ...loopbackSummary, failures: loopbackFailures } : null,
    minutes: engine.metrics.minutes,
    frames: engine.metrics.records,
  };
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `iris-측정-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ---- 주기적 화면 갱신(HUD·안내) ----

setInterval(() => {
  if (!running) return;
  const now = performance.now();
  if (engine.lastFace) lastFaceSeen = now;
  // 네일·타투만 쓸 때는 손이나 몸이 보이면 얼굴 안내를 띄우지 않는다
  guideEl.hidden = now - lastFaceSeen < 1200 || engine.lastHands.length > 0 || engine.lastPose !== null || engine.loopback?.running === true;
  if (!hudEl.hidden) {
    const lines = engine.metrics.hudLines(now);
    const src = engine.source;
    lines.unshift(`입력 ${src.width}×${src.height} · ${engine.tracker.delegate}`);
    if (loopbackSummary) {
      lines.push(`루프백 왕복 ${fmt(loopbackSummary.p50)} ms → 추정 전체 ${fmt(loopbackSummary.p50 + engine.metrics.proc.quantile(0.5))} ms`);
    }
    hudEl.textContent = lines.join('\n');
  }
}, 250);

// ---- 초기화 ----

function applyParams(): void {
  if (params.get('hud') === '1') {
    hudEl.hidden = false;
    $('btn-hud').classList.add('on');
  }
  const debug = (params.get('debug') ?? '').split(',');
  $<HTMLInputElement>('opt-dbg-lm').checked = debug.includes('lm');
  $<HTMLInputElement>('opt-dbg-seg').checked = debug.includes('seg');
  $<HTMLSelectElement>('opt-delegate').value = trackerConfig.delegate;
  $<HTMLSelectElement>('opt-seg-every').value = String(trackerConfig.segEvery);
  $<HTMLInputElement>('opt-hair-model').checked = trackerConfig.hairModel !== '';
  applySettings();
  // 아래는 모두 시험용: 화면에는 메뉴가 없고 자동 시험·평가 도구가 주소로 켠다
  const look = params.get('look') as LookName | null;
  if (look && look in LOOKS) selectLook(look);
  if (params.has('over')) overlip = Number(params.get('over'));
  if (params.has('pearl')) pearl = Number(params.get('pearl'));
  const ls = params.get('lstyle');
  if (ls === 'full' || ls === 'gradient' || ls === 'blur') lipStyle = ls;
  const hp = params.get('hair');
  if (hp) {
    const list = PALETTES.hair;
    const item = list[Number(hp)] ?? list.find((p) => p.name === hp);
    if (item) {
      chosen.hair.color = item.color;
      hairTip = item.tip ?? null;
    }
  }
  const np = params.get('nail');
  if (np) {
    const list = PALETTES.nail;
    const item = list[Number(np)] ?? list.find((p) => p.name === np);
    if (item) {
      chosen.nail.color = item.color;
      const ns = params.get('nstyle') as NailStyle | null;
      if (ns && ns in NAIL_COLOR2) nailStyle = ns;
      const nl = params.get('nlen');
      if (nl) nailLength = Number(nl);
    }
  }
  const tp = params.get('tattoo');
  const d = tp ? builtinDesigns().find((x) => x.id === tp) : undefined;
  if (d) {
    tattoo.design = d;
    const pl = params.get('place');
    if (pl && pl in PLACE_LABELS) tattoo.place = pl as TattooPlace;
    const ts = params.get('tsize');
    if (ts) tattoo.size = Number(ts);
  }
  applyAll();
  renderChips();
  const ref = params.get('ref');
  if (ref) {
    const mode = (params.get('refmode') ?? 'auto') as PhotoMode;
    void fetch(ref)
      .then((res) => res.blob())
      .then((b) => createImageBitmap(b))
      .then((bmp) => applyPhoto(bmp, mode))
      .catch((err) => setStatus(`사진을 열지 못했어요: ${(err as Error).message}`));
  }
}

applyParams();
try {
  localStorage.removeItem('iris.hairModel');
} catch {
  /* 무시 */
}
void refreshGenStatus();
{
  const src = params.get('src');
  if (src === 'sample') {
    $('btn-sample').click();
  } else if (src) {
    // 시험용: 같은 사이트 안의 동영상 주소로 바로 시작
    $<HTMLInputElement>('opt-mirror').checked = false;
    applyView();
    void begin(() => engine.source.openFile(new URL(src, base).href));
  }
}
