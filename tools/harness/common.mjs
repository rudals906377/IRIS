// 시험 도구 공통 설정. 환경 변수로 바꿀 수 있다.
//   IRIS_URL      미리보기 서버 주소(기본 http://localhost:4173)  — cd web && npx vite preview --port 4173
//   IRIS_CHROME   크롬 실행 파일(기본: 클라우드 환경의 Playwright Chromium)
import { chromium } from 'playwright';

export const BASE = process.env.IRIS_URL ?? 'http://localhost:4173';
export const CHROME = process.env.IRIS_CHROME ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
export const ARGS = [
  '--ignore-certificate-errors',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--autoplay-policy=no-user-gesture-required',
];

export function launch() {
  return chromium.launch({ executablePath: CHROME, args: ARGS });
}

/** 앱이 영상 프레임을 몇 장 처리하고 얼굴을 찾을 때까지 기다린다. */
export async function waitReady(page, timeout = 90000) {
  await page.waitForFunction(() => window.iris && window.iris.metrics.framesTotal > 4 && window.iris.lastFace, null, { timeout });
}

/** 스크린샷에서 검은 여백을 잘라 낸다(파이썬 없이 쓰려면 생략 가능). */
export function query(obj) {
  return new URLSearchParams(Object.fromEntries(Object.entries(obj).filter(([, v]) => v !== undefined))).toString();
}
