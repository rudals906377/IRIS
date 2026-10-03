// 생성 모드(사진 한 장): 웹캠 한 장을 사용자 컴퓨터의 생성 서버(tools/genserver)로 보내
// 헤어 모양·네일아트·타투를 생성형 AI로 그린 결과를 받는다. 서버는 로컬이라 사진이 인터넷으로 나가지 않는다.

export type GenCategory = 'hair' | 'nail' | 'tattoo';

export interface GenOptions {
  category: GenCategory;
  /** 웹캠 한 장(dataURL) */
  image: string;
  /** 참고 사진(dataURL) */
  reference?: string;
  /** 영어 설명(스타일 AI 속성이나 사용자가 적은 것) */
  desc?: string;
  place?: string;
  grow?: number;
  extend?: number;
  steps?: number;
  strength?: number;
  seed?: number;
  /** 엔진: default(인페인팅) | anydoor(물체 합성) */
  engine?: 'default' | 'anydoor';
}

export interface GenResult {
  image: string;
  mask: string;
  elapsed_ms: number;
  prompt: string;
  model: string;
}

export interface GenHealth {
  ok: boolean;
  dry_run: boolean;
  device: string;
  model_loaded: boolean;
  model: string | null;
  /** AnyDoor 합성 서버(8766)가 켜져 있는지 */
  anydoor?: boolean;
}

export const DEFAULT_GEN_URL = 'http://127.0.0.1:8765';
const KEY = 'iris.genServerUrl';

export function genServerUrl(): string {
  try {
    return localStorage.getItem(KEY) || DEFAULT_GEN_URL;
  } catch {
    return DEFAULT_GEN_URL;
  }
}

export function setGenServerUrl(url: string): void {
  try {
    localStorage.setItem(KEY, url.replace(/\/+$/, ''));
  } catch {
    /* 저장 못 해도 동작 */
  }
}

/** 서버가 켜져 있는지 확인(2초 안에 응답 없으면 꺼진 것으로 본다) */
export async function checkGenServer(url = genServerUrl()): Promise<GenHealth | null> {
  try {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), 2000);
    const r = await fetch(`${url}/health`, { signal: ctrl.signal });
    clearTimeout(t);
    if (!r.ok) return null;
    return (await r.json()) as GenHealth;
  } catch {
    return null;
  }
}

/** 생성 요청. 실패하면 서버가 준 한국어 이유를 담아 던진다 */
export async function generate(opts: GenOptions, url = genServerUrl()): Promise<GenResult> {
  const r = await fetch(`${url}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try {
      detail = ((await r.json()) as { detail?: string }).detail ?? detail;
    } catch {
      /* 본문 없음 */
    }
    throw new Error(detail);
  }
  return (await r.json()) as GenResult;
}

/** 캔버스 그림을 JPEG dataURL로(전송용, 긴 변 maxSide 이하) */
export function toDataUrl(src: HTMLCanvasElement | ImageData, maxSide = 1024, quality = 0.92): string {
  const w = src.width;
  const h = src.height;
  const s = Math.min(1, maxSide / Math.max(w, h));
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w * s));
  c.height = Math.max(1, Math.round(h * s));
  const ctx = c.getContext('2d')!;
  if (src instanceof ImageData) {
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d')!.putImageData(src, 0, 0);
    ctx.drawImage(tmp, 0, 0, c.width, c.height);
  } else {
    ctx.drawImage(src, 0, 0, c.width, c.height);
  }
  return c.toDataURL('image/jpeg', quality);
}
