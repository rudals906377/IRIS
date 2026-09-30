// 화면 구성과 사용자 조작. 엔진(engine/)을 불러 카메라·메이크업·설정을 연결한다.
//
// 개발·시험용 주소 인자:
//   ?src=sample            예시 영상으로 바로 시작
//   ?src=<경로>             같은 사이트의 동영상으로 바로 시작(시험용)
//   ?look=<룩 이름>         처음 적용할 룩(daily, coral, red, smoky, rose, clear)
//   ?hud=1                  측정 표시 켜기
//   ?debug=lm,seg           개발자 표시(얼굴 점, 분할)
//   ?delegate=CPU|GPU       처리 장치

import { LOOKS, PALETTES, PART_LABELS, type LookName, type PartName } from '../beauty/palettes.ts';
import type { MakeupLook, RGB } from '../beauty/makeup.ts';
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
};
let gloss = 0.3;
let tab: PartName | 'look' = 'look';
let currentLook: LookName | null = null;

function applyLookToEngine(): void {
  const look: MakeupLook = {};
  for (const part of Object.keys(chosen) as PartName[]) {
    const c = chosen[part];
    if (!c.color) continue;
    if (part === 'lip') look.lip = { color: c.color, amount: c.amount, gloss };
    else look[part] = { color: c.color, amount: c.amount };
  }
  engine.look = look;
}

function selectLook(name: LookName): void {
  const l = LOOKS[name];
  currentLook = name;
  for (const part of Object.keys(chosen) as PartName[]) {
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
    ...(['look', 'lip', 'shadow', 'blush', 'liner', 'brow'] as const).map((t) => {
      const b = document.createElement('button');
      b.className = 'tab' + (tab === t ? ' on' : '');
      b.textContent = t === 'look' ? '룩' : PART_LABELS[t];
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
    return;
  }
  const part = tab;
  const c = chosen[part];
  const items = [
    swatch('없음', null, !c.color, () => {
      c.color = null;
      currentLook = null;
      applyLookToEngine();
      renderRail();
    }),
    ...PALETTES[part].map((p) =>
      swatch(p.name, p.color, same(c.color, p.color), () => {
        c.color = p.color;
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

amountEl.addEventListener('input', () => {
  if (tab === 'look') return;
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
  guideEl.hidden = now - lastFaceSeen < 1200 || engine.loopback?.running === true;
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
