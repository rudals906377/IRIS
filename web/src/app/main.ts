// 화면 구성과 사용자 조작. 엔진(engine/)을 불러 카메라·메이크업·설정을 연결한다.
//
// 개발·시험용 주소 인자:
//   ?src=sample            예시 영상으로 바로 시작
//   ?src=<경로>             같은 사이트의 동영상으로 바로 시작(시험용)
//   ?look=<룩 이름>         처음 적용할 룩(daily, coral, red, smoky, rose, clear)
//   ?hair=<번호|이름>       헤어 색
//   ?nail=<번호|이름>&nstyle=solid|french|gradient|glitter|dots   네일
//   ?tattoo=<도안 id>&place=<위치>&tsize=0~1   타투(예: tattoo=moon&place=forearmL)
//   ?hud=1                  측정 표시 켜기
//   ?debug=lm,seg           개발자 표시(얼굴 점, 분할)
//   ?delegate=CPU|GPU       처리 장치

import { LOOKS, PALETTES, PART_LABELS, type LookName, type PartName } from '../beauty/palettes.ts';
import type { MakeupLook, RGB } from '../beauty/makeup.ts';
import { builtinDesigns, designFromFile, type TattooDesign } from '../beauty/tattoo-designs.ts';
import { PLACE_LABELS, type TattooPlace } from '../beauty/tattoo-place.ts';
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
const swatchesEl = $<HTMLElement>('swatches');
const amountEl = $<HTMLInputElement>('amount');
const amountRowEl = $<HTMLElement>('amount-row');
const glossRowEl = $<HTMLElement>('gloss-row');
const glossEl = $<HTMLInputElement>('gloss');
const loopbackEl = $<HTMLPreElement>('loopback-result');
const placeRowEl = $<HTMLElement>('place-row');
const placeEl = $<HTMLSelectElement>('place');
const sizeRowEl = $<HTMLElement>('size-row');
const sizeEl = $<HTMLInputElement>('size');
const tattooFileEl = $<HTMLInputElement>('tattoo-file');
const nstyleRowEl = $<HTMLElement>('nstyle-row');
const nstyleEl = $<HTMLSelectElement>('nstyle');

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
};

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
const NAIL_COLOR2: Record<NailStyle, RGB> = { solid: [1, 1, 1], french: [0.97, 0.96, 0.94], gradient: [0.96, 0.9, 0.88], glitter: [0.95, 0.8, 0.45], dots: [0.97, 0.96, 0.94] };
/** 헤어 그라데이션 끝 색(옴브레) */
let hairTip: RGB | null = null;
let gloss = 0.3;
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
    if (part === 'lip') look.lip = { color: c.color, amount: c.amount, gloss };
    else look[part] = { color: c.color, amount: c.amount };
  }
  engine.look = look;
  const h = chosen.hair;
  engine.hair = h.color ? { color: h.color, amount: h.amount, tip: hairTip ?? undefined } : null;
  const n = chosen.nail;
  engine.nail = n.color ? { color: n.color, color2: NAIL_COLOR2[nailStyle], style: nailStyle, amount: n.amount } : null;
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
    ...(['look', 'lip', 'shadow', 'blush', 'liner', 'brow', 'hair', 'nail', 'tattoo'] as const).map((t) => {
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
  if (tab === 'look') {
    swatchesEl.replaceChildren(
      ...(Object.keys(LOOKS) as LookName[]).map((name) => {
        const l = LOOKS[name];
        return swatch(l.label, l.parts.lip?.color ?? null, currentLook === name, () => selectLook(name));
      }),
    );
    amountRowEl.hidden = true;
    glossRowEl.hidden = true;
    placeRowEl.hidden = true;
    sizeRowEl.hidden = true;
    nstyleRowEl.hidden = true;
    return;
  }
  nstyleRowEl.hidden = tab !== 'nail';
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
  placeRowEl.hidden = false;
  placeEl.value = tattoo.place;
  sizeRowEl.hidden = false;
  sizeEl.value = String(tattoo.size);
}

nstyleEl.addEventListener('change', () => {
  nailStyle = nstyleEl.value as NailStyle;
  applyLookToEngine();
});

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

// ---- 시작 ----

async function begin(open: () => Promise<void>): Promise<void> {
  try {
    setStatus('영상 여는 중…');
    await open();
    startEl.hidden = true;
    await engine.tracker.configure(trackerConfig, setStatus);
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
  if (!running) return;
  await engine.tracker.configure(trackerConfig, setStatus);
  engine.metrics.reset();
  setStatus(`처리 장치 ${engine.tracker.delegate}`);
}

for (const id of ['opt-mirror', 'opt-seg', 'opt-refine', 'opt-dbg-lm', 'opt-dbg-seg']) {
  $(id).addEventListener('input', applySettings);
}
for (const id of ['opt-delegate', 'opt-seg-every']) {
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
  applySettings();
  const look = params.get('look') as LookName | null;
  selectLook(look && look in LOOKS ? look : 'daily');
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
