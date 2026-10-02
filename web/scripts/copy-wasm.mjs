// MediaPipe WASM 런타임을 외부 CDN 대신 우리 사이트에서 직접 제공하기 위해 public/ 으로 복사한다.
import { cpSync, mkdirSync } from 'node:fs';

const src = new URL('../node_modules/@mediapipe/tasks-vision/wasm/', import.meta.url);
const dst = new URL('../public/mediapipe/wasm/', import.meta.url);
mkdirSync(dst, { recursive: true });
cpSync(src, dst, { recursive: true });
console.log('copied MediaPipe wasm ->', dst.pathname);
