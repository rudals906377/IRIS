// IRIS 확장 프로그램 서비스 워커.
// - 툴바 아이콘: 현재 쇼핑몰 페이지에서 상품 사진 후보(대표 이미지 + 큰 이미지)를 모아 착용 창을 연다.
// - 이미지 우클릭 메뉴 "IRIS로 이 옷 입어보기": 그 사진을 첫 후보로 착용 창을 연다.
// 착용 창은 확장 프로그램 안의 웹 앱(index.html)이며, 카메라 영상은 이 컴퓨터 밖으로 나가지 않는다.
// 이 파일은 다른 모듈을 가져오지 않는다(서비스 워커 단독 번들).

const MENU_ID = 'iris-try-on';

interface Candidates {
  urls: string[];
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
    await openTryOn({ urls: [first, ...rest].slice(0, 12), title: found?.title ?? tab?.title, page: tab?.url });
  })();
});

chrome.action.onClicked.addListener((tab) => {
  void (async () => {
    const found = tab.id !== undefined ? await collect(tab.id) : null;
    await openTryOn({ urls: found?.urls ?? [], title: found?.title ?? tab.title, page: tab.url });
  })();
});

async function collect(tabId: number): Promise<Candidates | null> {
  try {
    const [res] = await chrome.scripting.executeScript({ target: { tabId }, func: collectProductImages });
    const got = res?.result as Candidates | undefined;
    if (!got) return null;
    return { ...got, urls: [...new Set(got.urls.map(upgradeImageUrl))] };
  } catch (err) {
    // chrome:// 같은 페이지에서는 스크립트를 넣을 수 없다
    console.warn('상품 사진 수집 실패', err);
    return null;
  }
}

/**
 * 페이지 안에서 실행된다(다른 함수·변수를 참조하면 안 됨).
 * 대표 이미지(og:image)를 최우선으로, 화면에 크게 보이는 이미지 순으로 후보를 만든다.
 */
function collectProductImages(): Candidates {
  const scores = new Map<string, number>();
  const add = (raw: string | null | undefined, score: number): void => {
    if (!raw || raw.startsWith('data:')) return;
    let u: string;
    try {
      u = new URL(raw, location.href).href;
    } catch {
      return;
    }
    scores.set(u, Math.max(scores.get(u) ?? 0, score));
  };
  add(document.querySelector('meta[property="og:image"]')?.getAttribute('content'), 1e12);
  for (const img of Array.from(document.images)) {
    const r = img.getBoundingClientRect();
    if (img.naturalWidth < 250 || img.naturalHeight < 250 || r.width < 120 || r.height < 120) continue;
    let src = img.currentSrc || img.src;
    // srcset 중 가장 큰 후보
    if (img.srcset) {
      const best = img.srcset
        .split(',')
        .map((s) => s.trim().split(/\s+/))
        .map(([u, d]) => ({ u, w: parseFloat(d ?? '') || 0 }))
        .sort((a, b) => b.w - a.w)[0];
      if (best?.u) src = best.u;
    }
    const visible = r.bottom > 0 && r.top < innerHeight * 2;
    add(src, r.width * r.height * (visible ? 2 : 1));
  }
  const urls = [...scores.entries()].sort((a, b) => b[1] - a[1]).map(([u]) => u).slice(0, 16);
  return { urls, title: document.title };
}

/** 쇼핑몰별로 더 큰 해상도의 이미지 주소로 바꾼다(모르는 사이트는 그대로). */
function upgradeImageUrl(u: string): string {
  try {
    const url = new URL(u);
    if (url.hostname === 'static.nike.com') {
      // 예: /a/images/t_PDP_936_v1/f_auto,q_auto:eco/… → t_PDP_1728_v1
      url.pathname = url.pathname.replace(/\/t_[A-Za-z0-9_]+_v\d+\//, '/t_PDP_1728_v1/');
      return url.href;
    }
    if (url.hostname === 'assets.adidas.com') {
      // 예: /images/w_600,f_auto,q_auto/… → w_1200
      url.pathname = url.pathname.replace(/\/images\/(?:[wh]_\d+,)+/, '/images/w_1200,');
      return url.href;
    }
    return u;
  } catch {
    return u;
  }
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
  tryTab: async (tabId: number) => {
    const found = await collect(tabId);
    await openTryOn({ urls: found?.urls ?? [], title: found?.title });
    return found;
  },
};
