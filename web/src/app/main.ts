// 화면 구성과 사용자 조작. 엔진(engine/)을 불러 카메라·상품·설정을 연결한다.
//
// 개발·테스트용 주소 인자:
//   ?src=sample            예시 영상으로 바로 시작
//   ?src=<경로>             같은 사이트의 동영상으로 바로 시작(시험용)
//   ?product=<상품 id>      처음 선택할 상품
//   ?hud=1                  측정 표시 켜기
//   ?debug=lm,seg,occ       개발자 표시
//   ?pose=lite|full|heavy   자세 모델
//   ?delegate=CPU|GPU       처리 장치

import { TryOnEngine } from '../engine/engine.ts';
import { loadCatalog, type ProductInfo } from '../engine/garment.ts';
import { fmt, summarize } from '../engine/stats.ts';
import { DEFAULT_TRACKER_CONFIG, type Delegate, type PoseModel, type TrackerConfig } from '../engine/tracker.ts';

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const params = new URLSearchParams(location.search);
const base = new URL('./', location.href);
const WASM_BASE = new URL('mediapipe/wasm', base).href;
const PRODUCT_BASE = new URL('products/', base).href;
const canMp4 = document.createElement('video').canPlayType('video/mp4; codecs="avc1.4d401e"') !== '';
const SAMPLE_VIDEO = new URL(canMp4 ? 'samples/sample-person.mp4' : 'samples/sample-person.webm', base).href;

const statusEl = $<HTMLDivElement>('status');
const hudEl = $<HTMLPreElement>('hud');
const guideEl = $<HTMLDivElement>('guide');
const startEl = $<HTMLDivElement>('start');
const viewEl = $<HTMLDivElement>('view');
const panelEl = $<HTMLElement>('panel');
const railEl = $<HTMLElement>('rail');
const loopbackEl = $<HTMLPreElement>('loopback-result');

const engine = new TryOnEngine({
  video: $<HTMLVideoElement>('video'),
  canvas: $<HTMLCanvasElement>('gl'),
  overlay: $<HTMLCanvasElement>('overlay'),
  wasmBase: WASM_BASE,
  productBase: PRODUCT_BASE,
});
// 개발 도구·자동 테스트에서 상태를 확인할 수 있게 노출
(window as unknown as { iris: TryOnEngine }).iris = engine;

const coarse = matchMedia('(pointer: coarse)').matches;
const trackerConfig: TrackerConfig = {
  ...DEFAULT_TRACKER_CONFIG,
  poseModel: (params.get('pose') as PoseModel) ?? (coarse ? 'lite' : 'full'),
  delegate: (params.get('delegate') as Delegate) ?? 'GPU',
  segEvery: coarse ? 2 : 1,
};

let running = false;
let lastPersonSeen = performance.now();
let loopbackSummary: ReturnType<typeof summarize> | null = null;
let loopbackFailures = 0;

function setStatus(msg: string): void {
  statusEl.textContent = msg;
}

// ---- 상품 목록 ----

let products: ProductInfo[] = [];

async function initCatalog(): Promise<void> {
  try {
    const catalog = await loadCatalog(PRODUCT_BASE);
    products = catalog.products;
  } catch (err) {
    setStatus(String(err));
    return;
  }
  railEl.replaceChildren(
    ...products.map((p) => {
      const b = document.createElement('button');
      b.className = 'product';
      b.dataset.id = p.id;
      b.title = p.name;
      const img = document.createElement('img');
      img.src = new URL(p.thumb, PRODUCT_BASE).href;
      img.alt = p.name;
      img.loading = 'lazy';
      b.append(img, document.createTextNode(p.name));
      b.addEventListener('click', () => void selectProduct(p.id));
      return b;
    }),
  );
  const first = params.get('product') ?? products[0]?.id;
  if (first) await selectProduct(first);
}

async function selectProduct(id: string): Promise<void> {
  const info = products.find((p) => p.id === id);
  if (!info) return;
  for (const el of railEl.querySelectorAll<HTMLButtonElement>('.product')) {
    el.classList.toggle('selected', el.dataset.id === id);
  }
  setStatus(`${info.name} 준비 중…`);
  const t0 = performance.now();
  try {
    await engine.setProduct(info);
    setStatus(`${info.name} · 준비 ${fmt(performance.now() - t0, 0)} ms`);
  } catch (err) {
    setStatus(`상품을 불러오지 못했습니다: ${String(err)}`);
  }
}

// ---- 시작 ----

async function begin(open: () => Promise<void>): Promise<void> {
  try {
    setStatus('영상 여는 중…');
    await open();
    startEl.hidden = true;
    await engine.tracker.configure(trackerConfig, setStatus);
    setStatus(`${engine.product?.name ?? ''} · 처리 장치 ${engine.tracker.delegate}`);
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
  s.shade = Number($<HTMLInputElement>('opt-shade').value);
  s.useSeg = $<HTMLInputElement>('opt-seg').checked;
  s.refine = $<HTMLInputElement>('opt-refine').checked;
  s.fit.seamWidth = Number($<HTMLInputElement>('opt-width').value);
  s.fit.lift = Number($<HTMLInputElement>('opt-lift').value);
  s.debugLandmarks = $<HTMLInputElement>('opt-dbg-lm').checked;
  s.debugSeg = $<HTMLInputElement>('opt-dbg-seg').checked;
  s.debugOcc = $<HTMLInputElement>('opt-dbg-occ').checked;
  applyView();
}

async function applyTracker(): Promise<void> {
  trackerConfig.poseModel = $<HTMLSelectElement>('opt-pose').value as PoseModel;
  trackerConfig.delegate = $<HTMLSelectElement>('opt-delegate').value as Delegate;
  trackerConfig.segEvery = Number($<HTMLSelectElement>('opt-seg-every').value);
  if (!running) return;
  await engine.tracker.configure(trackerConfig, setStatus);
  engine.metrics.reset();
  setStatus(`처리 장치 ${engine.tracker.delegate} · 자세 모델 ${trackerConfig.poseModel}`);
}

for (const id of ['opt-mirror', 'opt-shade', 'opt-seg', 'opt-refine', 'opt-width', 'opt-lift', 'opt-dbg-lm', 'opt-dbg-seg', 'opt-dbg-occ']) {
  $(id).addEventListener('input', applySettings);
}
for (const id of ['opt-pose', 'opt-delegate', 'opt-seg-every']) {
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
      `착용 처리 시간(중앙): ${fmt(proc)} ms`,
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
    product: engine.product?.id,
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
  if (engine.lastBody) lastPersonSeen = now;
  guideEl.hidden = now - lastPersonSeen < 1200 || engine.loopback?.running === true;
  if (!hudEl.hidden) {
    const lines = engine.metrics.hudLines(now);
    const src = engine.source;
    lines.unshift(`입력 ${src.width}×${src.height} · ${engine.tracker.delegate} · 자세 ${trackerConfig.poseModel}`);
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
  $<HTMLInputElement>('opt-dbg-occ').checked = debug.includes('occ');
  $<HTMLSelectElement>('opt-pose').value = trackerConfig.poseModel;
  $<HTMLSelectElement>('opt-delegate').value = trackerConfig.delegate;
  $<HTMLSelectElement>('opt-seg-every').value = String(trackerConfig.segEvery);
  applySettings();
}

applyParams();
void initCatalog().then(() => {
  const src = params.get('src');
  if (src === 'sample') {
    $('btn-sample').click();
  } else if (src) {
    // 시험용: 같은 사이트 안의 동영상 주소로 바로 시작
    $<HTMLInputElement>('opt-mirror').checked = false;
    applyView();
    void begin(() => engine.source.openFile(new URL(src, base).href));
  }
});
