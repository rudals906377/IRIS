// 손톱 라벨 도구: 손 사진을 열면 손 점(MediaPipe)으로 손톱 자리를 짐작해 8각형을 그려 주고,
// 사용자가 꼭짓점을 끌어 손톱 테두리에 맞춘 뒤 JSON으로 저장한다(손톱 전용 모델 학습 데이터).
// 라벨 형식: { version: 1, images: [{ file, width, height, done, nails: [{ pts: [[x, y] × 8 (0~1 비율)] }] }] }

import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { nailQuads, type HandPoints } from '../beauty/nail-place.ts';
import type { Vec2 } from '../engine/math.ts';

type Nail = { pts: [number, number][] };
type Entry = { file: string; width: number; height: number; done: boolean; nails: Nail[]; url?: string; img?: HTMLImageElement };

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const cv = $<HTMLCanvasElement>('cv');
const ctx = cv.getContext('2d')!;
const stage = $<HTMLDivElement>('stage');
const statusEl = $<HTMLSpanElement>('status');
const listEl = $<HTMLUListElement>('list');
const countEl = $<HTMLSpanElement>('count');
const base = new URL('./', location.href);
const WASM_BASE = new URL('mediapipe/wasm', base).href;
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const STORE = 'iris.nailLabels';

let entries: Entry[] = [];
let cur = -1;
let sel = -1; // 선택된 손톱
let hand: HandLandmarker | null = null;
// 보기 변환(사진 → 화면)
let scale = 1;
let ox = 0;
let oy = 0;

// ---- 저장 ----
function save(): void {
  try {
    const slim = entries.map(({ file, width, height, done, nails }) => ({ file, width, height, done, nails }));
    localStorage.setItem(STORE, JSON.stringify(slim));
  } catch {
    /* 용량 초과 등은 무시 */
  }
}
function loadStored(): Map<string, Entry> {
  try {
    const arr = JSON.parse(localStorage.getItem(STORE) || '[]') as Entry[];
    return new Map(arr.map((e) => [e.file, e]));
  } catch {
    return new Map();
  }
}

// ---- 손 모델 ----
async function ensureHand(): Promise<HandLandmarker> {
  if (hand) return hand;
  setStatus('손 모델 불러오는 중…');
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'GPU' },
    runningMode: 'IMAGE',
    numHands: 2,
  });
  return hand;
}

/** 손 점으로 손톱 8각형 짐작(nail-place.ts와 같은 계산) */
async function guess(e: Entry): Promise<Nail[]> {
  const h = await ensureHand();
  if (!e.img) return [];
  const r = h.detect(e.img);
  const out: Nail[] = [];
  r.landmarks.forEach((lm, i) => {
    const hp: HandPoints = {
      p: lm.map((q) => ({ x: q.x * e.width, y: q.y * e.height })),
      z: lm.map((q) => q.z),
      handed: r.handedness[i]?.[0]?.categoryName ?? 'Right',
    };
    // 손바닥이 보여도 일단 그려 준다(사용자가 지움): facingSign 양쪽 다 시도해 보이는 쪽을 쓴다
    const quads = [...nailQuads(hp, 1), ...nailQuads(hp, -1)];
    const seen = new Set<number>();
    for (const q of quads) {
      if (seen.has(q.finger)) continue;
      seen.add(q.finger);
      out.push({ pts: octagon(q.c, q.dir, q.len, q.width, e) });
    }
  });
  return out;
}

/** 손톱 사각형 → 둥근 8각형(비율 좌표). 뿌리 쪽은 둥글게, 끝은 조금 덜 둥글게 */
function octagon(c: Vec2, dir: Vec2, len: number, width: number, e: Entry): [number, number][] {
  const px = -dir.y;
  const py = dir.x;
  const pts: [number, number][] = [];
  // 각도별 반지름(타원 비슷하게): 길이 방향 = len/2, 폭 방향 = width/2
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2;
    const u = Math.cos(a) * (width / 2) * 1.2;
    const v = Math.sin(a) * (len / 2) * 1.2;
    pts.push([(c.x + px * u + dir.x * v) / e.width, (c.y + py * u + dir.y * v) / e.height]);
  }
  return pts;
}

// ---- 화면 ----
function setStatus(s: string): void {
  statusEl.textContent = s;
}
function fit(): void {
  const e = entries[cur];
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  cv.width = W;
  cv.height = H;
  if (!e) return;
  scale = Math.min(W / e.width, H / e.height);
  ox = (W - e.width * scale) / 2;
  oy = (H - e.height * scale) / 2;
}
const toScreen = (p: [number, number], e: Entry): Vec2 => ({ x: ox + p[0] * e.width * scale, y: oy + p[1] * e.height * scale });
const toImage = (x: number, y: number, e: Entry): [number, number] => [(x - ox) / scale / e.width, (y - oy) / scale / e.height];

function draw(): void {
  const e = entries[cur];
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!e || !e.img) return;
  ctx.drawImage(e.img, ox, oy, e.width * scale, e.height * scale);
  e.nails.forEach((n, i) => {
    const on = i === sel;
    ctx.beginPath();
    n.pts.forEach((p, k) => {
      const s = toScreen(p, e);
      if (k === 0) ctx.moveTo(s.x, s.y);
      else ctx.lineTo(s.x, s.y);
    });
    ctx.closePath();
    ctx.fillStyle = on ? 'rgba(240,160,64,0.25)' : 'rgba(80,200,255,0.18)';
    ctx.fill();
    ctx.lineWidth = on ? 2 : 1.5;
    ctx.strokeStyle = on ? '#f0a040' : '#4fc3f7';
    ctx.stroke();
    for (const p of n.pts) {
      const s = toScreen(p, e);
      ctx.beginPath();
      ctx.arc(s.x, s.y, on ? 5 : 3.5, 0, Math.PI * 2);
      ctx.fillStyle = on ? '#f0a040' : '#4fc3f7';
      ctx.fill();
    }
  });
}

function renderList(): void {
  listEl.replaceChildren(
    ...entries.map((e, i) => {
      const li = document.createElement('li');
      li.textContent = `${e.file} (${e.nails.length})`;
      li.className = (i === cur ? 'cur ' : '') + (e.done ? 'done' : '');
      li.addEventListener('click', () => void show(i));
      return li;
    }),
  );
  const done = entries.filter((e) => e.done).length;
  countEl.textContent = entries.length ? `${done}/${entries.length} 완료` : '';
}

async function show(i: number): Promise<void> {
  if (i < 0 || i >= entries.length) return;
  cur = i;
  sel = -1;
  const e = entries[i];
  if (!e.img && e.url) {
    e.img = await loadImage(e.url);
    e.width = e.img.naturalWidth;
    e.height = e.img.naturalHeight;
  }
  if (e.nails.length === 0 && !e.done && e.img) {
    setStatus('손톱 자리 짐작 중…');
    e.nails = await guess(e);
    save();
  }
  fit();
  draw();
  renderList();
  setStatus(`${i + 1}/${entries.length} · ${e.file} · 손톱 ${e.nails.length}개${e.done ? ' · 완료' : ''}`);
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = url;
  });
}

// ---- 입력 ----
$<HTMLInputElement>('files').addEventListener('change', async (ev) => {
  const files = Array.from((ev.target as HTMLInputElement).files ?? []);
  if (files.length === 0) return;
  const stored = loadStored();
  const existing = new Set(entries.map((e) => e.file));
  for (const f of files) {
    if (existing.has(f.name)) continue;
    const prev = stored.get(f.name);
    entries.push({ file: f.name, width: 0, height: 0, done: prev?.done ?? false, nails: prev?.nails ?? [], url: URL.createObjectURL(f) });
  }
  entries.sort((a, b) => a.file.localeCompare(b.file));
  renderList();
  const first = entries.findIndex((e) => !e.done);
  await show(first >= 0 ? first : 0);
});

$('prev').addEventListener('click', () => void show(cur - 1));
$('next').addEventListener('click', () => void show(cur + 1));
$('done').addEventListener('click', () => {
  const e = entries[cur];
  if (!e) return;
  e.done = true;
  save();
  const nxt = entries.findIndex((x, i) => i > cur && !x.done);
  void show(nxt >= 0 ? nxt : Math.min(cur + 1, entries.length - 1));
});
$('auto').addEventListener('click', async () => {
  const e = entries[cur];
  if (!e) return;
  e.nails = await guess(e);
  sel = -1;
  save();
  draw();
  renderList();
});
$('add').addEventListener('click', () => addNail());
$('del').addEventListener('click', () => delNail());
$('export').addEventListener('click', () => {
  const data = { version: 1, images: entries.map(({ file, width, height, done, nails }) => ({ file, width, height, done, nails })) };
  const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'nail-labels.json';
  a.click();
});
$<HTMLInputElement>('import').addEventListener('change', async (ev) => {
  const f = (ev.target as HTMLInputElement).files?.[0];
  if (!f) return;
  const data = JSON.parse(await f.text()) as { images: Entry[] };
  const byName = new Map(entries.map((e) => [e.file, e]));
  for (const e of data.images) {
    const t = byName.get(e.file);
    if (t) {
      t.nails = e.nails;
      t.done = e.done;
    } else entries.push({ ...e, url: undefined });
  }
  save();
  renderList();
  draw();
  setStatus(`라벨 ${data.images.length}장 불러옴`);
});

function addNail(): void {
  const e = entries[cur];
  if (!e) return;
  const c = { x: e.width / 2, y: e.height / 2 };
  const s = Math.min(e.width, e.height) * 0.05;
  e.nails.push({ pts: octagon(c, { x: 0, y: -1 }, s * 1.2, s, e) });
  sel = e.nails.length - 1;
  save();
  draw();
  renderList();
}
function delNail(): void {
  const e = entries[cur];
  if (!e || sel < 0) return;
  e.nails.splice(sel, 1);
  sel = -1;
  save();
  draw();
  renderList();
}

// 드래그: 꼭짓점 또는 손톱 전체
let drag: { nail: number; vert: number; last: Vec2 } | null = null;
const inPoly = (q: Vec2, poly: Vec2[]): boolean => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (a.y > q.y !== b.y > q.y && q.x < ((b.x - a.x) * (q.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
};
cv.addEventListener('pointerdown', (ev) => {
  const e = entries[cur];
  if (!e) return;
  const m = { x: ev.offsetX, y: ev.offsetY };
  // 꼭짓점 먼저(선택된 손톱 우선)
  const order = [...e.nails.keys()].sort((a, b) => (a === sel ? -1 : b === sel ? 1 : 0));
  for (const i of order) {
    const n = e.nails[i];
    for (let k = 0; k < n.pts.length; k++) {
      const s = toScreen(n.pts[k], e);
      if (Math.hypot(s.x - m.x, s.y - m.y) < 8) {
        drag = { nail: i, vert: k, last: m };
        sel = i;
        cv.setPointerCapture(ev.pointerId);
        draw();
        return;
      }
    }
  }
  for (const i of order) {
    if (inPoly(m, e.nails[i].pts.map((p) => toScreen(p, e)))) {
      drag = { nail: i, vert: -1, last: m };
      sel = i;
      cv.setPointerCapture(ev.pointerId);
      draw();
      return;
    }
  }
  sel = -1;
  draw();
});
cv.addEventListener('pointermove', (ev) => {
  const e = entries[cur];
  if (!e || !drag) return;
  const m = { x: ev.offsetX, y: ev.offsetY };
  const n = e.nails[drag.nail];
  const dx = (m.x - drag.last.x) / scale / e.width;
  const dy = (m.y - drag.last.y) / scale / e.height;
  if (drag.vert >= 0) n.pts[drag.vert] = toImage(m.x, m.y, e);
  else n.pts = n.pts.map(([x, y]) => [x + dx, y + dy]);
  drag.last = m;
  draw();
});
cv.addEventListener('pointerup', () => {
  if (drag) save();
  drag = null;
});
// 휠 확대(마우스 위치 기준)
cv.addEventListener('wheel', (ev) => {
  const e = entries[cur];
  if (!e) return;
  ev.preventDefault();
  const f = ev.deltaY < 0 ? 1.15 : 1 / 1.15;
  const mx = ev.offsetX;
  const my = ev.offsetY;
  ox = mx - (mx - ox) * f;
  oy = my - (my - oy) * f;
  scale *= f;
  draw();
}, { passive: false });
window.addEventListener('keydown', (ev) => {
  if ((ev.target as HTMLElement).tagName === 'INPUT') return;
  if (ev.key === 'ArrowLeft') void show(cur - 1);
  else if (ev.key === 'ArrowRight') void show(cur + 1);
  else if (ev.key === 'Enter') $('done').click();
  else if (ev.key === 'Delete' || ev.key === 'Backspace') delNail();
  else if (ev.key.toLowerCase() === 'a') addNail();
});
window.addEventListener('resize', () => {
  fit();
  draw();
});
fit();
