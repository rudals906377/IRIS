// 화면 구성과 사용자 조작. 엔진(engine/)을 불러 카메라·메이크업·설정을 연결한다.
//
// 개발·시험용 주소 인자:
//   ?src=sample            예시 영상으로 바로 시작
//   ?src=<경로>             같은 사이트의 동영상으로 바로 시작(시험용)
//   ?look=<룩 이름>         처음 적용할 룩(daily, idol, coral, red, smoky, rose, glam, clear)
//   ?over=-1~1 ?pearl=0~1   입술 라인(오버립), 아이섀도 펄
//   ?lstyle=full|gradient|blur  립 모양
//   ?hair=<번호|이름>       헤어 색
//   ?ref=<사진 주소>&refmode=auto|makeup|hair|nail|tattoo   참고 사진을 분석해 적용(시험용)
//   ?nail=<번호|이름>&nstyle=solid|french|gradient|glitter|dots|chrome|jelly|cateye|aurora&nlen=0~1   네일
//   ?tattoo=<도안 id>&place=<위치>&tsize=0~1   타투(예: tattoo=moon&place=forearmL)
//   ?hud=1                  측정 표시 켜기
//   ?debug=lm,seg           개발자 표시(얼굴 점, 분할)
//   ?delegate=CPU|GPU       처리 장치

import { LOOKS, PALETTES, PART_LABELS, type LookName, type PartName } from '../beauty/palettes.ts';
import type { MakeupLook, RGB } from '../beauty/makeup.ts';
import type { LipStyle } from '../beauty/face-regions.ts';
import { builtinDesigns, designFromFile, type TattooDesign } from '../beauty/tattoo-designs.ts';
import { PLACE_LABELS, type TattooPlace } from '../beauty/tattoo-place.ts';
import { NATURAL_APPLE, NATURAL_BROW_L, NATURAL_LINER_L, NATURAL_MID, NATURAL_SHADOW, PhotoAnalyzer, type PhotoMode, type StyleResult } from '../beauty/photo-style.ts';
import { measureFace, type FaceMeasure } from '../beauty/face-measure.ts';
import { matchDarkness, matchLip, matchTint } from '../beauty/style-match.ts';
import { analyzeStyle } from '../beauty/style-ai.ts';
import { hintsFromStyleAI, promptFromStyleAI, type StyleHints } from '../beauty/style-attributes.ts';
import { checkGenServer, generate as genGenerate, genServerUrl, setGenServerUrl, toDataUrl, type GenCategory } from '../beauty/gen-mode.ts';
import { gam, lin, lipTarget, luma } from '../beauty/style-math.ts';
import type { NailStyle } from '../beauty/nail.ts';
import { BeautyEngine } from '../engine/engine.ts';
import { fmt, summarize } from '../engine/stats.ts';
import { DEFAULT_TRACKER_CONFIG, type Delegate, type TrackerConfig } from '../engine/tracker.ts';

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
const viewEl = $<HTMLDivElement>('view');
const panelEl = $<HTMLElement>('panel');
const tabsEl = $<HTMLElement>('tabs');
const btnPhotoEl = $<HTMLButtonElement>('btn-photo');
const btnGenEl = $<HTMLButtonElement>('btn-gen');
const swatchesEl = $<HTMLElement>('swatches');
const amountEl = $<HTMLInputElement>('amount');
const amountRowEl = $<HTMLElement>('amount-row');
const glossRowEl = $<HTMLElement>('gloss-row');
const glossEl = $<HTMLInputElement>('gloss');
const overRowEl = $<HTMLElement>('over-row');
const lstyleRowEl = $<HTMLElement>('lstyle-row');
const lstyleEl = $<HTMLSelectElement>('lstyle');
const overEl = $<HTMLInputElement>('over');
const pearlRowEl = $<HTMLElement>('pearl-row');
const pearlEl = $<HTMLInputElement>('pearl');
const loopbackEl = $<HTMLPreElement>('loopback-result');
const placeRowEl = $<HTMLElement>('place-row');
const placeEl = $<HTMLSelectElement>('place');
const sizeRowEl = $<HTMLElement>('size-row');
const sizeEl = $<HTMLInputElement>('size');
const tattooFileEl = $<HTMLInputElement>('tattoo-file');
const styleFileEl = $<HTMLInputElement>('style-file');
const refEl = $<HTMLElement>('ref');
const refImgEl = $<HTMLImageElement>('ref-img');
const refTextEl = $<HTMLElement>('ref-text');
const nstyleRowEl = $<HTMLElement>('nstyle-row');
const nstyleEl = $<HTMLSelectElement>('nstyle');
const nlenRowEl = $<HTMLElement>('nlen-row');
const nlenEl = $<HTMLInputElement>('nlen');

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
  // 자체 머리카락 모델: ?hairmodel=models/hair-matte-256.onnx (또는 설정 체크박스)
  hairModel: params.get('hairmodel') ?? (localStorage.getItem('iris.hairModel') || ''),
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

// ---- 메이크업 선택 ----

/** 부위별로 마지막에 고른 색(없음이면 null)과 진하기 */
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
/** 입술 라인(-1~1)과 아이섀도 펄(0~1) */
let overlip = 0;
let lipStyle: LipStyle = 'full';
let pearl = 0;
let tab: PartName | 'look' | 'tattoo' = 'look';

// ---- 타투 선택 ----
const designs: TattooDesign[] = builtinDesigns();
const tattoo = {
  design: null as TattooDesign | null,
  place: 'forearmL' as TattooPlace,
  size: 0.5,
  amount: 0.85,
};
const INK: [number, number, number] = [0.16, 0.17, 0.2];
for (const [k, label] of Object.entries(PLACE_LABELS)) placeEl.append(new Option(label, k));

function applyTattooToEngine(): void {
  const d = tattoo.design;
  engine.tattoo = d ? { design: d, place: tattoo.place, size: tattoo.size, amount: tattoo.amount, ink: INK } : null;
  // 타투는 몸 관절점이 필요하다: 처음 켤 때 자세 추적을 켠다(모델을 한 번 내려받음)
  if (d && !trackerConfig.pose) {
    trackerConfig.pose = true;
    if (running) void engine.tracker.configure(trackerConfig, setStatus).then(() => setStatus(`처리 장치 ${engine.tracker.delegate}`));
  }
}
let currentLook: LookName | null = null;

function applyLookToEngine(): void {
  const look: MakeupLook = {};
  for (const part of Object.keys(chosen) as PartName[]) {
    const c = chosen[part];
    if (!c.color || part === 'hair' || part === 'nail') continue;
    if (part === 'lip') look.lip = { color: c.color, amount: c.amount, gloss, over: overlip, style: lipStyle };
    else if (part === 'blush') look.blush = { color: c.color, amount: c.amount, pos: blushPos, size: blushSize };
    else if (part === 'shadow') look.shadow = { color: c.color, amount: c.amount, pearl };
    else look[part] = { color: c.color, amount: c.amount };
  }
  engine.look = look;
  const h = chosen.hair;
  engine.hair = h.color ? { color: h.color, amount: h.amount, tip: hairTip ?? undefined } : null;
  const n = chosen.nail;
  engine.nail = n.color ? { color: n.color, color2: NAIL_COLOR2[nailStyle], style: nailStyle, amount: n.amount, length: nailLength } : null;
  // 네일은 손 점이 필요하다: 처음 켤 때 손 추적을 켠다
  if (n.color && !trackerConfig.hands) {
    trackerConfig.hands = true;
    if (running) void engine.tracker.configure(trackerConfig, setStatus).then(() => setStatus(`처리 장치 ${engine.tracker.delegate}`));
  }
}

function selectLook(name: LookName): void {
  const l = LOOKS[name];
  currentLook = name;
  // 룩은 메이크업만 바꾼다(헤어 색은 그대로)
  for (const part of Object.keys(chosen) as PartName[]) {
    if (part === 'hair' || part === 'nail') continue;
    const v = l.parts[part];
    chosen[part].color = v ? v.color : null;
    if (v) chosen[part].amount = v.amount;
  }
  gloss = l.gloss ?? gloss;
  glossEl.value = String(gloss);
  pearl = l.pearl ?? 0;
  pearlEl.value = String(pearl);
  lipStyle = l.lipStyle ?? 'full';
  lstyleEl.value = lipStyle;
  blushPos = 0.5;
  blushSize = 1;
  applyLookToEngine();
  renderRail();
}

const hex = (c: RGB): string => '#' + c.map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
const fromHex = (h: string): RGB => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255) as RGB;
const same = (a: RGB | null, b: RGB | null): boolean => !!a && !!b && a.every((v, i) => Math.abs(v - b[i]) < 0.004);

function swatch(label: string, color: RGB | null, selected: boolean, onClick: () => void): HTMLButtonElement {
  const b = document.createElement('button');
  b.className = 'swatch' + (selected ? ' selected' : '');
  b.title = label;
  const dot = document.createElement('span');
  dot.className = 'dot' + (color ? '' : ' none');
  if (color) dot.style.background = hex(color);
  b.append(dot, document.createTextNode(label));
  b.addEventListener('click', onClick);
  return b;
}

function renderRail(): void {
  tabsEl.replaceChildren(
    ...(['look', 'base', 'lip', 'shadow', 'blush', 'contour', 'liner', 'brow', 'hair', 'nail', 'tattoo'] as const).map((t) => {
      const b = document.createElement('button');
      b.className = 'tab' + (tab === t ? ' on' : '');
      b.textContent = t === 'look' ? '룩' : t === 'tattoo' ? '타투' : PART_LABELS[t];
      b.addEventListener('click', () => {
        tab = t;
        renderRail();
      });
      return b;
    }),
  );
  // 사진 따라하기가 첫 번째 동작이 되도록 고정 버튼(사진 따라하기·생성)을 탭 앞에 둔다(탭을 다시 그려도 사라지지 않게)
  tabsEl.prepend(btnPhotoEl, btnGenEl);
  if (tab === 'look') {
    swatchesEl.replaceChildren(
      ...(Object.keys(LOOKS) as LookName[]).map((name) => {
        const l = LOOKS[name];
        return swatch(l.label, l.parts.lip?.color ?? null, currentLook === name, () => selectLook(name));
      }),
    );
    amountRowEl.hidden = true;
    glossRowEl.hidden = true;
    overRowEl.hidden = true;
    lstyleRowEl.hidden = true;
    pearlRowEl.hidden = true;
    placeRowEl.hidden = true;
    sizeRowEl.hidden = true;
    nstyleRowEl.hidden = true;
    nlenRowEl.hidden = true;
    return;
  }
  nstyleRowEl.hidden = tab !== 'nail';
  nlenRowEl.hidden = tab !== 'nail';
  if (tab === 'tattoo') {
    renderTattooRail();
    return;
  }
  placeRowEl.hidden = true;
  sizeRowEl.hidden = true;
  const part = tab;
  const c = chosen[part];
  const items = [
    swatch('없음', null, !c.color, () => {
      c.color = null;
      if (part === 'hair') hairTip = null;
      currentLook = null;
      applyLookToEngine();
      renderRail();
    }),
    ...PALETTES[part].map((p) =>
      swatch(p.name, p.tip ?? p.color, same(c.color, p.color) && (part !== 'hair' || same(hairTip, p.tip ?? null) || (!hairTip && !p.tip)), () => {
        c.color = p.color;
        if (part === 'hair') hairTip = p.tip ?? null;
        currentLook = null;
        applyLookToEngine();
        renderRail();
      }),
    ),
  ];
  // 직접 고르기
  const pick = document.createElement('label');
  pick.className = 'swatch';
  pick.title = '직접 고르기';
  const input = document.createElement('input');
  input.type = 'color';
  input.value = c.color ? hex(c.color) : '#c0404a';
  input.addEventListener('input', () => {
    c.color = fromHex(input.value);
    if (part === 'hair') hairTip = null;
    currentLook = null;
    applyLookToEngine();
  });
  const dot = document.createElement('span');
  dot.className = 'dot custom';
  pick.append(input, dot, document.createTextNode('직접'));
  items.push(pick as unknown as HTMLButtonElement);
  swatchesEl.replaceChildren(...items);
  amountRowEl.hidden = false;
  amountEl.value = String(c.amount);
  glossRowEl.hidden = part !== 'lip';
  overRowEl.hidden = part !== 'lip';
  lstyleRowEl.hidden = part !== 'lip';
  pearlRowEl.hidden = part !== 'shadow';
}

function renderTattooRail(): void {
  const pick = (d: TattooDesign | null): void => {
    tattoo.design = d;
    applyTattooToEngine();
    renderRail();
  };
  const items = [
    swatch('없음', null, !tattoo.design, () => pick(null)),
    ...designs.map((d) => {
      const b = swatch(d.name, null, tattoo.design === d, () => pick(d));
      const dot = b.querySelector<HTMLElement>('.dot')!;
      dot.className = 'dot design';
      dot.style.backgroundImage = `url(${d.canvas.toDataURL()})`;
      return b;
    }),
  ];
  const up = swatch('내 도안', null, tattoo.design?.id === 'upload', () => tattooFileEl.click());
  up.querySelector<HTMLElement>('.dot')!.className = 'dot custom';
  items.push(up);
  swatchesEl.replaceChildren(...items);
  amountRowEl.hidden = false;
  amountEl.value = String(tattoo.amount);
  glossRowEl.hidden = true;
  overRowEl.hidden = true;
  lstyleRowEl.hidden = true;
  pearlRowEl.hidden = true;
  placeRowEl.hidden = false;
  placeEl.value = tattoo.place;
  sizeRowEl.hidden = false;
  sizeEl.value = String(tattoo.size);
}

nlenEl.addEventListener('input', () => {
  nailLength = Number(nlenEl.value);
  applyLookToEngine();
});
nstyleEl.addEventListener('change', () => {
  nailStyle = nstyleEl.value as NailStyle;
  applyLookToEngine();
});

// ---- 참고 사진 따라하기 ----
let photoAnalyzer: PhotoAnalyzer | null = null;
/** 마지막 분석 결과(개발 도구·자동 시험용) */
let lastStyle: StyleResult | null = null;

/** 지금 탭에 따라 사진에서 무엇을 가져올지 정한다 */
function photoMode(): PhotoMode {
  if (tab === 'hair') return 'hair';
  if (tab === 'nail') return 'nail';
  if (tab === 'tattoo') return 'tattoo';
  return 'auto';
}

async function applyPhoto(source: ImageBitmap | HTMLImageElement, mode: PhotoMode): Promise<StyleResult> {
  photoAnalyzer ??= new PhotoAnalyzer(WASM_BASE);
  setStatus('사진 분석 중…');
  const r = await photoAnalyzer.analyze(source, mode, setStatus);
  // 스타일 AI: 무늬·모양·기법(색은 위에서 잰 값을 쓴다). 실패해도 색 적용은 계속한다
  if ($<HTMLInputElement>('opt-style-ai').checked && source instanceof ImageBitmap) {
    try {
      const cat = mode === 'auto' ? 'auto' : mode;
      const ai = await analyzeStyle(source, cat, (p) => {
        if (p.status === 'progress' && p.file && p.progress !== undefined) setStatus(`스타일 AI 모델 내려받는 중 ${Math.round(p.progress)}%`);
        else if (p.status === 'ready') setStatus('스타일 AI 분석 중…');
      });
      r.ai = { hints: hintsFromStyleAI(ai), headline: ai.headline, description: ai.description_ko, raw: ai };
      applyHints(r, r.ai.hints);
    } catch (err) {
      console.warn('스타일 AI를 쓰지 못했습니다(색만 적용)', err);
    }
  }
  lastStyle = r;
  const m = r.makeup;
  if (m) {
    // 사진의 메이크업으로 바꾼다(사진에 없는 부위는 지움). 피부 보정은 사진이 보정된 경우가 많아 약하게 켠다
    currentLook = null;
    for (const part of ['lip', 'blush', 'shadow', 'liner', 'brow', 'contour'] as const) {
      const v = m[part];
      chosen[part].color = v ? v.color : null;
      if (v) chosen[part].amount = v.amount;
    }
    blushPos = m.blush?.pos ?? 0.5;
    blushSize = m.blush?.size ?? 1;
    chosen.base.color = c('#d1a38a');
    chosen.base.amount = 0.35;
    if (m.lip) {
      gloss = m.lip.gloss;
      glossEl.value = String(gloss);
      lipStyle = m.lip.style;
      lstyleEl.value = lipStyle;
    }
    pearl = 0;
    pearlEl.value = '0';
  }
  if (r.hair) {
    chosen.hair.color = r.hair.color;
    chosen.hair.amount = r.hair.amount;
    hairTip = r.hair.tip;
  }
  if (r.nail) {
    chosen.nail.color = r.nail.color;
    nailStyle = r.nail.style;
    nstyleEl.value = nailStyle;
  }
  applyLookToEngine();
  if (r.tattoo) {
    tattoo.design = r.tattoo;
    applyTattooToEngine();
  }
  renderRail();
  refImgEl.src = r.thumb.toDataURL('image/jpeg', 0.8);
  refTextEl.textContent = r.ai?.headline ? `${r.summary} · AI: ${r.ai.headline}` : r.summary;
  refEl.hidden = false;
  setStatus(r.summary);
  if (r.makeup && r.measure) {
    // 영상이 아직 시작 전이면(사진 분석이 먼저 끝난 경우) 잠시 기다린다
    for (let i = 0; i < 100 && !running; i++) await new Promise((res) => setTimeout(res, 100));
    if (running) {
      try {
        await matchToPhoto(r.measure);
      } catch (err) {
        console.error('되먹임 실패', err);
        lastMatch.push({ error: String(err) });
      }
    }
  }
  return r;
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
    if (k.blush === 'none' && (m.blush?.redness ?? 0) < 0.15) m.blush = undefined;
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
    if (h.nail.length !== undefined) {
      nailLength = h.nail.length;
      nlenEl.value = String(nailLength);
    }
  }
  if (r.tattoo && h.tattoo) {
    if (h.tattoo.place) tattoo.place = h.tattoo.place;
    if (h.tattoo.size !== undefined) tattoo.size = h.tattoo.size;
  }
}

// ---- 생성 모드(사진 한 장): 내 컴퓨터의 생성 서버로 헤어 모양·네일아트·타투를 그린다 ----
const genEl = $<HTMLElement>('gen');
const genStatusEl = $<HTMLElement>('gen-status');
const genCatEl = $<HTMLSelectElement>('gen-cat');
const genEngineEl = $<HTMLSelectElement>('gen-engine');
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
genUrlEl.addEventListener('change', () => {
  setGenServerUrl(genUrlEl.value.trim() || genServerUrl());
  genUrlEl.value = genServerUrl();
  void refreshGenStatus();
});

async function refreshGenStatus(): Promise<boolean> {
  genStatusEl.textContent = '서버 확인 중…';
  const h = await checkGenServer();
  if (!h) {
    genStatusEl.textContent = `생성 서버가 꺼져 있어요 (${genServerUrl()}) — tools/genserver/README.md 참고`;
    genRunEl.disabled = true;
    return false;
  }
  genStatusEl.textContent = h.dry_run ? '연결됨 (시험 모드: 영역만 표시)' : `연결됨 · ${h.device === 'cuda' ? 'GPU' : h.device.toUpperCase()}${h.model_loaded ? '' : ' · 첫 생성 때 모델을 불러와요'}${h.anydoor ? ' · AnyDoor 준비' : ''}`;
  // AnyDoor 서버가 꺼져 있으면 엔진 선택을 기본으로 되돌린다
  if (!h.anydoor && genEngineEl.value === 'anydoor') genEngineEl.value = 'default';
  genEngineEl.title = h.anydoor ? 'AnyDoor 서버 연결됨' : 'AnyDoor 서버(8766)가 꺼져 있어요 — tools/anydoor/run.bat';
  genRunEl.disabled = false;
  return true;
}

/** 생성 대상에 맞춰 설명(영어)과 옵션 줄을 채운다 */
function syncGenCategory(): void {
  const cat = genCatEl.value as GenCategory;
  $('gen-grow-row').hidden = cat !== 'hair';
  $('gen-extend-row').hidden = cat !== 'nail';
  // 스타일 AI가 읽은 속성을 영어 설명으로(사용자가 고칠 수 있다)
  const auto = promptFromStyleAI(lastStyle?.ai?.raw, cat);
  if (auto && !genDescEl.dataset.edited) genDescEl.value = auto;
  if (cat === 'nail' && lastStyle?.nail) genExtendEl.value = String(nailLength);
}
genCatEl.addEventListener('change', syncGenCategory);
genDescEl.addEventListener('input', () => {
  genDescEl.dataset.edited = genDescEl.value ? '1' : '';
});
btnGenEl.addEventListener('click', () => {
  genEl.hidden = !genEl.hidden;
  if (!genEl.hidden) {
    // 지금 탭에 맞는 대상을 기본으로
    if (tab === 'nail') genCatEl.value = 'nail';
    else if (tab === 'tattoo') genCatEl.value = 'tattoo';
    else if (tab === 'hair') genCatEl.value = 'hair';
    syncGenCategory();
    void refreshGenStatus();
  }
});
$('gen-close').addEventListener('click', () => {
  genEl.hidden = true;
});
genSliderEl.addEventListener('input', () => {
  genAfterEl.style.clipPath = `inset(0 0 0 ${genSliderEl.value}%)`;
});

/** 마지막 생성 결과(자동 시험용) */
let lastGen: { elapsed_ms: number; prompt: string; model: string } | null = null;

async function runGen(): Promise<void> {
  if (!running) {
    setStatus('먼저 카메라나 영상을 시작하세요');
    return;
  }
  const cat = genCatEl.value as GenCategory;
  genRunEl.disabled = true;
  genStatusEl.textContent = '지금 모습을 찍는 중…';
  try {
    // 합성된 화면(실시간 메이크업 포함)을 한 장 찍는다
    const cap = await engine.captureNext();
    const image = toDataUrl(cap.image, 1024);
    const reference = genUseRefEl.checked && lastStyle ? lastStyle.thumb.toDataURL('image/jpeg', 0.9) : undefined;
    genBeforeEl.src = image;
    const working = genEngineEl.value === 'anydoor' ? '합성 중… (AnyDoor, 한 자리에 30~120초)' : '생성 중… (GPU 5~30초, CPU는 몇 분)';
    genStatusEl.textContent = working;
    const r = await genGenerate(
      {
        category: cat,
        image,
        reference,
        desc: genDescEl.value.trim() || undefined,
        place: tattoo.place,
        grow: Number(genGrowEl.value),
        extend: Number(genExtendEl.value),
        engine: genEngineEl.value === 'anydoor' ? 'anydoor' : 'default',
      },
      undefined,
      (ms, line) => {
        genStatusEl.textContent = `${working} ${Math.round(ms / 1000)}초${line ? ' · ' + line.replace(/^\d\d:\d\d:\d\d /, '') : ''}`;
      },
    );
    genAfterEl.src = r.image;
    genAfterEl.style.clipPath = `inset(0 0 0 ${genSliderEl.value}%)`;
    genSaveEl.href = r.image;
    genInfoEl.textContent = `${(r.elapsed_ms / 1000).toFixed(1)}초 · ${r.model === 'dry-run' ? '시험 모드' : r.model === 'anydoor' ? 'AnyDoor 합성' : r.model}`;
    genResultEl.hidden = false;
    genStatusEl.textContent = '완료 — 가운데 막대를 움직여 전후를 비교하세요';
    lastGen = { elapsed_ms: r.elapsed_ms, prompt: r.prompt, model: r.model };
  } catch (err) {
    genStatusEl.textContent = `생성 실패: ${(err as Error).message}`;
  } finally {
    genRunEl.disabled = false;
  }
}
genRunEl.addEventListener('click', () => void runGen());
(window as unknown as { irisGen: unknown }).irisGen = { run: runGen, get last() { return lastGen; }, status: refreshGenStatus };

/** 되먹임 기록(자동 시험·디버그용) */
let lastMatch: Record<string, unknown>[] = [];

/**
 * 사진 따라하기 되먹임: 화장 전 내 얼굴과 화장 후 내 얼굴을 같은 방법으로 재서,
 * 사진과 같은 정도가 되도록 색·진하기를 세 번 고친다.
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
  for (let it = 0; it < 3; it++) {
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
    if (chosen.shadow.color && darker(photo) && darker(now) && darker(my)) {
      const m = matchTint({ color: chosen.shadow.color, amount: chosen.shadow.amount }, darker(photo)!, darker(now)!, NATURAL_SHADOW, darker(my)!);
      chosen.shadow.color = m.amount > 0 ? m.color : null;
      chosen.shadow.amount = m.amount > 0 ? m.amount : 0.45;
      log.shadow = { want: darker(photo), got: darker(now), amount: m.amount };
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
  renderRail();
}
const c = (h: string): RGB => fromHex(h);

styleFileEl.addEventListener('change', async () => {
  const f = styleFileEl.files?.[0];
  styleFileEl.value = '';
  if (!f) return;
  try {
    await applyPhoto(await createImageBitmap(f), photoMode());
  } catch (err) {
    console.error(err);
    setStatus(`사진을 분석하지 못했습니다: ${(err as Error).message}`);
  }
});
btnPhotoEl.addEventListener('click', () => styleFileEl.click());
$('ref-clear').addEventListener('click', () => {
  refEl.hidden = true;
  lastStyle = null;
});
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
  return { photo: pick(lastStyle.measure), result: pick(now), match: lastMatch, settings: { blush: chosen.blush, shadow: chosen.shadow, lip: chosen.lip, brow: chosen.brow, liner: chosen.liner, blushPos } };
}
(window as unknown as { irisPhoto: unknown }).irisPhoto = { apply: applyPhoto, get last() { return lastStyle; }, get match() { return lastMatch; }, measure: measureFace, check: checkPhoto };

tattooFileEl.addEventListener('change', async () => {
  const f = tattooFileEl.files?.[0];
  tattooFileEl.value = '';
  if (!f) return;
  try {
    tattoo.design = await designFromFile(f);
    applyTattooToEngine();
    renderRail();
  } catch (err) {
    setStatus(`도안을 열지 못했습니다: ${(err as Error).message}`);
  }
});
placeEl.addEventListener('change', () => {
  tattoo.place = placeEl.value as TattooPlace;
  applyTattooToEngine();
});
sizeEl.addEventListener('input', () => {
  tattoo.size = Number(sizeEl.value);
  applyTattooToEngine();
});

amountEl.addEventListener('input', () => {
  if (tab === 'look') return;
  if (tab === 'tattoo') {
    tattoo.amount = Number(amountEl.value);
    applyTattooToEngine();
    return;
  }
  chosen[tab].amount = Number(amountEl.value);
  applyLookToEngine();
});
glossEl.addEventListener('input', () => {
  gloss = Number(glossEl.value);
  applyLookToEngine();
});
lstyleEl.addEventListener('change', () => {
  lipStyle = lstyleEl.value as LipStyle;
  applyLookToEngine();
});
overEl.addEventListener('input', () => {
  overlip = Number(overEl.value);
  applyLookToEngine();
});
pearlEl.addEventListener('input', () => {
  pearl = Number(pearlEl.value);
  applyLookToEngine();
});

// ---- 시작 ----

async function begin(open: () => Promise<void>): Promise<void> {
  try {
    setStatus('영상 여는 중…');
    await open();
    startEl.hidden = true;
    await engine.tracker.configure(trackerConfig, setStatus);
    if (NAIL_MODEL_PARAM) await engine.setNailModel(NAIL_MODEL_PARAM, setStatus);
    setStatus(`처리 장치 ${engine.tracker.delegate}`);
    engine.start();
    running = true;
  } catch (err) {
    const msg = err instanceof DOMException && err.name === 'NotAllowedError' ? '카메라 권한이 거부되었습니다. 브라우저 주소창에서 권한을 허용해 주세요.' : String(err);
    setStatus(msg);
    startEl.hidden = false;
  }
}

function cameraRequest(): { width: number; height: number; fps: number } {
  const [w, h] = $<HTMLSelectElement>('opt-res').value.split('x').map(Number);
  return { width: w, height: h, fps: 30 };
}

$('btn-camera').addEventListener('click', () => void begin(() => engine.source.openCamera(cameraRequest())));
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
  try {
    localStorage.setItem('iris.hairModel', trackerConfig.hairModel);
  } catch {
    /* 저장 못 해도 동작 */
  }
  if (!running) return;
  await engine.tracker.configure(trackerConfig, setStatus);
  engine.metrics.reset();
  setStatus(`처리 장치 ${engine.tracker.delegate}`);
}

for (const id of ['opt-mirror', 'opt-seg', 'opt-refine', 'opt-dbg-lm', 'opt-dbg-seg']) {
  $(id).addEventListener('input', applySettings);
}
for (const id of ['opt-delegate', 'opt-seg-every', 'opt-hair-model']) {
  $(id).addEventListener('change', () => void applyTracker());
}
$('opt-res').addEventListener('change', () => {
  if (running && engine.source.kind === 'camera') void begin(() => engine.source.openCamera(cameraRequest()));
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
  const look = params.get('look') as LookName | null;
  selectLook(look && look in LOOKS ? look : 'daily');
  // ?over=-1~1(입술 라인), ?pearl=0~1(아이섀도 펄)
  if (params.has('over')) overEl.value = String((overlip = Number(params.get('over'))));
  if (params.has('pearl')) pearlEl.value = String((pearl = Number(params.get('pearl'))));
  const ls = params.get('lstyle');
  if (ls === 'full' || ls === 'gradient' || ls === 'blur') lstyleEl.value = lipStyle = ls;
  applyLookToEngine();
  // ?hair=<색상표 번호 또는 이름>
  const hp = params.get('hair');
  if (hp) {
    const list = PALETTES.hair;
    const item = list[Number(hp)] ?? list.find((p) => p.name === hp);
    if (item) {
      chosen.hair.color = item.color;
      hairTip = item.tip ?? null;
      applyLookToEngine();
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
      nstyleEl.value = nailStyle;
      const nl = params.get('nlen');
      if (nl) nlenEl.value = String((nailLength = Number(nl)));
      applyLookToEngine();
    }
  }
  const tp = params.get('tattoo');
  const d = tp ? designs.find((x) => x.id === tp) : undefined;
  if (d) {
    tattoo.design = d;
    const pl = params.get('place');
    if (pl && pl in PLACE_LABELS) tattoo.place = pl as TattooPlace;
    const ts = params.get('tsize');
    if (ts) tattoo.size = Number(ts);
    applyTattooToEngine();
  }
  const ref = params.get('ref');
  if (ref) {
    const mode = (params.get('refmode') ?? 'auto') as PhotoMode;
    void fetch(ref)
      .then((res) => res.blob())
      .then((b) => createImageBitmap(b))
      .then((bmp) => applyPhoto(bmp, mode))
      .catch((err) => setStatus(`참고 사진을 열지 못했습니다: ${(err as Error).message}`));
  }
}

applyParams();
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
