// IRIS 확장 프로그램 서비스 워커.
// - 툴바 아이콘: 현재 쇼핑몰 페이지에서 상품 사진 후보(대표 이미지 + 큰 이미지)를 모아 착용 창을 연다.
// - 이미지 우클릭 메뉴 "IRIS로 이 옷 입어보기": 그 사진을 첫 후보로 착용 창을 연다.
// 착용 창은 확장 프로그램 안의 웹 앱(index.html)이며, 카메라 영상은 이 컴퓨터 밖으로 나가지 않는다.
// 빌드 시 urls.ts와 함께 하나의 서비스 워커 파일(background.js)로 묶인다.

import { upgradeImageUrl } from './urls.ts';

const MENU_ID = 'iris-try-on';

interface Candidates {
  urls: string[];
  /** 고해상도로 바꾼 주소 → 원래 주소(바꾼 주소가 열리지 않을 때 대신 쓴다) */
  alts?: Record<string, string>;
  title?: string;
  page?: string;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: MENU_ID, title: 'IRIS로 이 옷 입어보기', contexts: ['image'] });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== MENU_ID || !info.srcUrl) return;
  void (async () => {
    const found = tab?.id !== undefined ? await collect(tab.id) : null;
    const first = upgradeImageUrl(info.srcUrl!);
    const rest = (found?.urls ?? []).filter((u) => u !== first);
    const alts = { ...(found?.alts ?? {}), ...(first !== info.srcUrl ? { [first]: info.srcUrl! } : {}) };
    await openTryOn({ urls: [first, ...rest].slice(0, 12), alts, title: found?.title ?? tab?.title, page: tab?.url });
  })();
});

chrome.action.onClicked.addListener((tab) => {
  void (async () => {
    const found = tab.id !== undefined ? await collect(tab.id) : null;
    await openTryOn({ urls: found?.urls ?? [], alts: found?.alts, title: found?.title ?? tab.title, page: tab.url });
  })();
});

async function collect(tabId: number): Promise<Candidates | null> {
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId }, func: collectProductImages });
    const got = res?.result as Candidates | undefined;
    if (!got) return null;
    const alts: Record<string, string> = {};
    const urls: string[] = [];
    for (const u of got.urls) {
      const up = upgradeImageUrl(u);
      if (urls.includes(up)) continue;
      urls.push(up);
      if (up !== u) alts[up] = u;
    }
    return { ...got, urls, alts };
  } catch (err) {
    // chrome:// 같은 페이지에서는 스크립트를 넣을 수 없다
    console.warn('상품 사진 수집 실패', err);
    return null;
  }
}

/**
 * 페이지 안에서 실행된다(다른 함수·변수를 참조하면 안 됨).
 * 상품 사진 후보를 모아 "옷만 찍힌 앞면 사진"일 가능성이 높은 순으로 돌려준다.
 * - 구조화 데이터(JSON-LD Product.image)와 대표 이미지(og:image)
 * - 상세 갤러리 이미지: 지연 로딩(data-src, srcset, <picture><source>)까지 포함
 * - 추천 상품 영역(다른 상품으로 가는 링크 안, 페이지 아래쪽)은 낮은 점수
 * - 나이키·아디다스 파일명 규칙(아디다스 _laydown = 옷만 찍은 사진, _model = 모델 착용 등)
 * 최종 선택은 착용 창이 사진을 실제로 분석해 가장 알맞은 것을 고른다.
 */
function collectProductImages(): { urls: string[]; title?: string } {
  const found = new Map<string, { u: string; score: number; order: number }>();
  let order = 0;
  const sameImageKey = (u: URL): string => {
    // 같은 사진의 다른 크기 변형을 하나로 묶는다(CDN 변환 인자 부분 무시).
    if (u.hostname === 'static.nike.com' || u.hostname === 'assets.adidas.com') {
      return u.hostname + '/' + u.pathname.split('/').slice(-2).join('/');
    }
    return u.hostname + u.pathname;
  };
  const nameHint = (u: URL): number => {
    const p = decodeURIComponent(u.pathname).toLowerCase();
    let k = 1;
    if (/_laydown/.test(p)) k *= 3;
    if (/_0?1_(laydown|standard)/.test(p)) k *= 2; // 앞면
    if (/_0?2_(laydown|standard)/.test(p)) k *= 0.5; // 뒷면
    if (/_(model|hover_model)/.test(p)) k *= 0.6;
    if (/_(detail|41_|42_|43_)/.test(p)) k *= 0.2;
    if (/(logo|icon|sprite|banner|swatch|avatar|badge|payment)/.test(p)) k *= 0.05;
    if (/\.svg$|\.gif$/.test(p)) k *= 0.01;
    return k;
  };
  const add = (raw: string | null | undefined, score: number): void => {
    if (!raw) return;
    raw = raw.trim();
    if (!raw || raw.startsWith('data:') || raw.startsWith('blob:')) return;
    let url: URL;
    try {
      url = new URL(raw, location.href);
    } catch {
      return;
    }
    if (!/^https?:$/.test(url.protocol)) return;
    const k = sameImageKey(url);
    const s = score * nameHint(url);
    const prev = found.get(k);
    if (!prev) found.set(k, { u: url.href, score: s, order: order++ });
    else if (s > prev.score) found.set(k, { ...prev, u: url.href, score: s });
  };
  const bestOfSrcset = (srcset: string | null | undefined): string | null => {
    if (!srcset) return null;
    const list = srcset
      .split(/,\s+(?=\S)/)
      .map((s) => s.trim().split(/\s+/))
      .map(([u, d]) => ({ u, w: parseFloat(d ?? '') || 1 }))
      .filter((x) => x.u);
    list.sort((a, b) => b.w - a.w);
    return list[0]?.u ?? null;
  };

  // 1) 구조화 데이터
  for (const el of Array.from(document.querySelectorAll('script[type="application/ld+json"]'))) {
    try {
      const visit = (node: unknown): void => {
        if (!node || typeof node !== 'object') return;
        if (Array.isArray(node)) return node.forEach(visit);
        const o = node as Record<string, unknown>;
        const t = o['@type'];
        if (t === 'Product' || (Array.isArray(t) && t.includes('Product'))) {
          const imgs = ([] as unknown[]).concat(o.image ?? []);
          imgs.forEach((im, i) => {
            const u = typeof im === 'string' ? im : (im as Record<string, unknown>)?.url;
            if (typeof u === 'string') add(u, 5e6 / (1 + i * 0.15));
          });
        }
        if (o['@graph']) visit(o['@graph']);
      };
      visit(JSON.parse(el.textContent ?? ''));
    } catch {
      /* 잘못된 JSON은 무시 */
    }
  }
  // 2) 대표 이미지
  add(document.querySelector('meta[property="og:image"]')?.getAttribute('content'), 8e6);
  add(document.querySelector('meta[name="twitter:image"]')?.getAttribute('content'), 4e6);

  // 3) 페이지 속 이미지(지연 로딩 포함)
  const here = location.pathname.replace(/\/$/, '');
  const knownCdn = /(^|\.)static\.nike\.com$|(^|\.)assets\.adidas\.com$/;
  for (const img of Array.from(document.querySelectorAll('img'))) {
    if (img.closest('header, nav, footer')) continue;
    const r = img.getBoundingClientRect();
    const pageTop = r.top + scrollY;
    const cands = [
      bestOfSrcset(img.getAttribute('srcset')),
      bestOfSrcset(img.getAttribute('data-srcset')),
      img.getAttribute('data-src'),
      img.getAttribute('data-original'),
      img.currentSrc,
      img.getAttribute('src'),
    ];
    const pic = img.closest('picture');
    if (pic) for (const s of Array.from(pic.querySelectorAll('source'))) cands.unshift(bestOfSrcset(s.getAttribute('srcset') || s.getAttribute('data-srcset')));
    const src = cands.find((c) => c && !c.startsWith('data:'));
    if (!src) continue;
    let host = '';
    try {
      host = new URL(src, location.href).hostname;
    } catch {
      continue;
    }
    const lazy = !img.getAttribute('src') || !img.complete || img.naturalWidth === 0;
    const big = r.width >= 120 && r.height >= 120 && (img.naturalWidth >= 250 || lazy);
    // 갤러리 썸네일: 작게 보여도 원본이 크거나(축소 표시) 알려진 쇼핑몰 CDN이면 같은 상품의 다른 사진으로 받는다.
    const thumb = r.width >= 40 && r.height >= 40 && (img.naturalWidth >= 250 || knownCdn.test(host));
    const hiddenSlide = r.width === 0 && knownCdn.test(host); // 캐러셀에서 가려진 사진
    if (!big && !thumb && !hiddenSlide) continue;
    let score = Math.max(r.width * r.height, 120 * 120);
    // 다른 상품으로 가는 링크 안(추천·최근 본 상품)은 다른 옷이므로 뺀다.
    const a = img.closest('a[href]') as HTMLAnchorElement | null;
    if (a) {
      try {
        const to = new URL(a.href, location.href);
        if (to.pathname.replace(/\/$/, '') !== here) continue;
      } catch {
        /* 무시 */
      }
    }
    // 상세 갤러리는 페이지 위쪽에 있다(한참 아래는 추천·후기 영역).
    if (pageTop > innerHeight * 2.5) continue;
    if (pageTop > innerHeight * 1.6) score *= 0.1;
    add(src, score);
  }
  const urls = [...found.values()]
    .sort((a, b) => b.score - a.score || a.order - b.order)
    .map((c) => c.u)
    .slice(0, 16);
  return { urls, title: document.title };
}

async function openTryOn(c: Candidates): Promise<void> {
  await chrome.storage.session.set({ candidates: { ...c, at: Date.now() } });
  const base = chrome.runtime.getURL('index.html');
  const url = `${base}?from=ext&t=${Date.now()}`;
  // 이미 열린 착용 창이 있으면 그 창을 새 후보로 다시 연다
  const tabs = await chrome.tabs.query({ url: `${base}*` });
  const existing = tabs[0];
  if (existing?.id !== undefined) {
    await chrome.tabs.update(existing.id, { url, active: true });
    if (existing.windowId !== undefined) await chrome.windows.update(existing.windowId, { focused: true });
    return;
  }
  await chrome.windows.create({ url, type: 'popup', width: 1180, height: 860 });
}

// 자동 시험용: 툴바 클릭을 흉내 낼 수 있게 노출(사용자 기능에는 영향 없음)
(globalThis as unknown as { irisTest: unknown }).irisTest = {
  upgradeImageUrl,
  tryTab: async (tabId: number) => {
    const found = await collect(tabId);
    await openTryOn({ urls: found?.urls ?? [], alts: found?.alts, title: found?.title });
    return found;
  },
};
