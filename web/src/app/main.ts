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
import type { LipStyle } from '../beauty/face-regions.ts';
import { builtinDesigns, designFromFile, type TattooDesign } from '../beauty/tattoo-designs.ts';
import { PLACE_LABELS, type TattooPlace } from '../beauty/tattoo-place.ts';
import { NATURAL_APPLE, NATURAL_BROW_L, NATURAL_LINER_L, NATURAL_MID, NATURAL_SHADOW, PhotoAnalyzer, SKIN_REF, tattooFromSource, type PhotoMode, type StyleResult } from '../beauty/photo-style.ts';
import { FACE_OVAL, measureFace, type FaceMeasure } from '../beauty/face-measure.ts';
import { matchDarkness, matchLip, matchTint } from '../beauty/style-match.ts';
import { analyzeStyle } from '../beauty/style-ai.ts';
import { hintsFromStyleAI, promptFromStyleAI, type StyleHints } from '../beauty/style-attributes.ts';
import { checkGenServer, generate as genGenerate, genServerUrl, setGenServerUrl, toDataUrl, type GenCategory } from '../beauty/gen-mode.ts';
import { gam, lin, lipTarget, luma } from '../beauty/style-math.ts';
import type { NailStyle } from '../beauty/nail.ts';
import { BeautyEngine } from '../engine/engine.ts';
import { fmt, summarize } from '../engine/stats.ts';
import { DEFAULT_TRACKER_CONFIG, type Delegate, type TrackerConfig } from '../engine/tracker.ts';
import { StickerGL } from '../engine/sticker-gl.ts';
import { parseFace } from '../beauty/face-parse.ts';
import { closeHoles, guidedFilter, hairDist, recolorHair, type HairDist } from '../beauty/still-hair.ts';
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
  engine.tattoo = d && !off.has('tattoo') ? { design: d, place: tattoo.place, size: tattoo.size, amount: amt(tattoo.amount), ink: INK, along: tattoo.along, across: tattoo.across } : null;
  // 타투는 몸 관절점이 필요하다: 처음 켤 때 자세 추적을 켠다(모델을 한 번 내려받음)
  if (d && !trackerConfig.pose) {
    trackerConfig.pose = true;
    if (running) void engine.tracker.configure(trackerConfig, setStatus).then(() => setStatus(''));
  }
}

function applyLookToEngine(): void {
  const look: MakeupLook = {};
  for (const part of Object.keys(chosen) as PartName[]) {
    const c = chosen[part];
    if (!c.color || off.has(part) || part === 'hair' || part === 'nail') continue;
    if (part === 'lip') look.lip = { color: c.color, amount: amt(c.amount), gloss, over: overlip, style: lipStyle };
    else if (part === 'blush') look.blush = { color: c.color, amount: amt(c.amount), pos: blushPos, size: blushSize };
    else if (part === 'shadow') look.shadow = { color: c.color, amount: amt(c.amount), pearl, colorOut: shadowOut ?? undefined };
    else look[part] = { color: c.color, amount: amt(c.amount) };
  }
  engine.look = look;
  const h = chosen.hair;
  engine.hair = h.color && !off.has('hair') ? { color: h.color, amount: amt(h.amount), tip: hairTip ?? undefined } : null;
  // 사진 모드: 염색은 셰이더 대신 고른 사진의 실제 머리색 분포를 올마다 옮겨 사진에 굽는다(still-hair.ts)
  if (PHOTO_MODE && engine.source.kind === 'image') {
    engine.hair = null;
    scheduleStillHair();
  }
  const n = chosen.nail;
  engine.nail = n.color && !off.has('nail') ? { color: n.color, color2: NAIL_COLOR2[nailStyle], style: nailStyle, amount: amt(n.amount), length: nailLength } : null;
  // 네일은 손 점이 필요하다: 처음 켤 때 손 추적을 켠다
  if (n.color && !trackerConfig.hands) {
    trackerConfig.hands = true;
    if (running) void engine.tracker.configure(trackerConfig, setStatus).then(() => setStatus(''));
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
    } catch (err) {
      console.warn('스타일 AI를 쓰지 못했습니다(색만 적용)', err);
    }
  }
  // 사람이 없는 도안 그림(흰 바탕·투명 PNG)은 그대로 타투 도안으로
  if (!r.makeup && !r.hair && !r.nail && !r.tattoo && file && looksLikeDesign(r.thumb)) {
    try {
      r.tattoo = await designFromFile(file);
      r.summary = '그림을 타투 도안으로 가져왔어요';
    } catch (err) {
      console.warn('도안으로 열지 못했습니다', err);
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
    tattoo.along = tattoo.across = undefined;
    setOff(['tattoo'], !on('tattoo'));
    applyTattooToEngine();
  }
  lastMatch = [];
  renderChips();
  refImgEl.src = r.thumb.toDataURL('image/jpeg', 0.8);
  refTextEl.textContent = (r.ai?.headline ? `${r.summary} · ${r.ai.headline}` : r.summary) + (r.faces && r.faces > 1 ? ` · 얼굴 ${r.faces}명 중 큰 얼굴(다른 사람은 ✂️ 영역)` : '');
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
let refHair: { src: ImageBitmap; dist: HairDist | null } | null = null;
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
  for (let i = 0; i < 80 && !(r = engine.tracker.headRect(engine.source.width, engine.source.height)); i++) await new Promise((res) => setTimeout(res, 100));
  if (!r || photoOriginal !== orig) return null;
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
  // 아주 어두운 머리카락 덩어리는 분할·필터가 놓쳐 작은 구멍(원래 색 얼룩)이 된다 → 머리 크기에 맞춰 닫기
  const closed = closeHoles(alpha, W, H, Math.max(3, Math.round(rect.size / 90)));
  for (let i = 0; i < alpha.length; i++) alpha[i] = closed[i];
  const dist = hairDist(rgba, alpha);
  if (!dist) return null;
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
  refHair = { src, dist: hairDist(p.rgba, a, 0.7, 200) };
  return refHair.dist;
}
async function updateStillHair(): Promise<void> {
  if (!PHOTO_MODE || !photoOriginal || engine.source.kind !== 'image') return;
  if (stillHairBusy) {
    await stillHairBusy;
    return updateStillHair();
  }
  const h = chosen.hair;
  // 분포를 그대로 옮기므로 진하기 1 이 '고른 사진 그대로'(분석의 amount 는 셰이더용이라 쓰지 않는다)
  const amount = h.color && !off.has('hair') && lastSource ? strength : 0;
  const key = `${amount.toFixed(3)}|${lastSource ? 1 : 0}`;
  if (key === stillHairKey && refHair?.src === lastSource) return;
  // 작업이 곧바로 끝나는 경우(할 일 없음)에도 표시가 남지 않게, 끝난 뒤 '그 작업이면' 지운다
  const job = (async () => {
    try {
      if (amount <= 0) {
        engine.source.replaceStill(photoOriginal!);
        stillHairKey = key;
        return;
      }
      const mine = await mineHairMatte();
      const ref = await refHairDist();
      if (!mine || !ref) {
        // 정밀 방식이 안 되면(얼굴·머리카락을 못 찾음) 실시간 염색으로
        engine.hair = { color: h.color!, amount: amt(h.amount), tip: hairTip ?? undefined };
        setStatus(mine ? '고른 사진에서 머리카락을 찾지 못해 머리색만 옮겼어요' : '');
        return;
      }
      setStatus('머리카락 색을 올마다 옮기는 중…');
      const out = recolorHair(mine.rgba, mine.w, mine.h, mine.alpha, mine.dist, ref, Math.min(1, amount));
      const bmp = await createImageBitmap(new ImageData(out as unknown as Uint8ClampedArray<ArrayBuffer>, mine.w, mine.h));
      engine.source.replaceStill(bmp);
      stillHairKey = key;
      setStatus('');
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
}

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
  mineHair = null;
  stillHairKey = '';
  // 사진은 시간이 넉넉하므로 크게 처리한다(분할 합성 1024, 원본 긴 변 2400 까지)
  engine.tracker.hiW = 1024;
  await begin(() => engine.source.openImage(bmp, 2400));
  applyAll();
  if (!running) return;
  btnMyPhotoEl.hidden = false;
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
    cx.drawImage(photoOriginal, 0, 0, cv.width, cv.height);
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
    setStatus('팔·목·가슴 위를 눌러 주세요. 몸이 화면에 보여야 해요');
    return;
  }
  tattoo.place = hit.place;
  tattoo.along = hit.along;
  tattoo.across = hit.across;
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
    const t = tattooFromSource(src, r);
    if (!t) return '이 영역에서 타투 선을 찾지 못했어요. 조금 넓게 다시 감싸 주세요';
    tattoo.design = t.design;
    if (lastStyle) {
      lastStyle.tattoo = t.design;
      lastStyle.tattooSrc = { src, rect: t.rect, inkMask: t.inkMask };
    }
    off.delete('tattoo');
    applyTattooToEngine();
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
