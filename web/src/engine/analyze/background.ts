// 상품 사진 배경 제거: 단색 또는 부드럽게 변하는(비네팅) 흰색·연회색 배경의 쇼핑몰 상품 사진용.
//
// 방법: 테두리에서 시작해 "매끄럽고(밝기 변화가 작고) 배경색 범위 안인" 픽셀로만 채워 나간다.
// 옷의 외곽선에는 아무리 흰 옷이라도 그림자·봉제선 같은 밝기 변화가 있으므로 그것이 벽이 되어,
// 흰 옷이 흰 배경 위에 있어도 옷 안으로 새어 들어가지 않는다. 옷 안의 흰 무늬도 지워지지 않는다.

export interface RGBAImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

export interface BackgroundResult {
  /** 전경(옷) 마스크 0/1 */
  mask: Uint8Array;
  bg: [number, number, number];
  /** 테두리 색의 흩어짐(클수록 배경이 단색이 아님) */
  spread: number;
  /** 배경으로 인정한 기울기 문턱 */
  gradThreshold: number;
}

function median(values: ArrayLike<number>): number {
  const s = Array.from(values).sort((a, b) => a - b);
  return s[s.length >> 1] ?? 0;
}

/** 3×3 평균으로 JPEG 잡음을 누른 RGB */
function smooth(img: RGBAImage): Float32Array {
  const { data, width: w, height: h } = img;
  const out = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      let n = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const i = (yy * w + xx) * 4;
          r += data[i];
          g += data[i + 1];
          b += data[i + 2];
          n++;
        }
      }
      const o = (y * w + x) * 3;
      out[o] = r / n;
      out[o + 1] = g / n;
      out[o + 2] = b / n;
    }
  }
  return out;
}

/** 채널별 Sobel 크기의 최댓값 */
function gradient(rgb: Float32Array, w: number, h: number): Float32Array {
  const g = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let best = 0;
      for (let c = 0; c < 3; c++) {
        const at = (xx: number, yy: number): number => rgb[(yy * w + xx) * 3 + c];
        const gx = at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1);
        const gy = at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1) - at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1);
        best = Math.max(best, Math.hypot(gx, gy) / 4);
      }
      g[y * w + x] = best;
    }
  }
  return g;
}

export function removeBackground(img: RGBAImage): BackgroundResult {
  const { width: w, height: h } = img;
  const rgb = smooth(img);
  const grad = gradient(rgb, w, h);

  // 테두리 표본
  const m = Math.max(2, Math.round(Math.min(w, h) * 0.015));
  const idx: number[] = [];
  for (let y = 0; y < h; y++) for (let k = 0; k < m; k++) idx.push(y * w + k, y * w + (w - 1 - k));
  for (let x = 0; x < w; x++) for (let k = 0; k < m; k++) idx.push(k * w + x, (h - 1 - k) * w + x);
  const bg: [number, number, number] = [
    median(idx.map((i) => rgb[i * 3])),
    median(idx.map((i) => rgb[i * 3 + 1])),
    median(idx.map((i) => rgb[i * 3 + 2])),
  ];
  const dev = idx.map((i) => Math.hypot(rgb[i * 3] - bg[0], rgb[i * 3 + 1] - bg[1], rgb[i * 3 + 2] - bg[2])).sort((a, b) => a - b);
  const spread = dev[Math.floor(dev.length * 0.9)] ?? 0;
  const borderGrad = idx.map((i) => grad[i]).sort((a, b) => a - b);
  // 배경의 자연스러운 기울기(비네팅·잡음)보다 확실히 큰 변화만 벽으로 본다.
  const gradThreshold = Math.max(3, (borderGrad[Math.floor(borderGrad.length * 0.95)] ?? 0) * 2 + 1.5);
  // 배경색 범위: 비네팅을 감안해 넉넉하게(단, 선명한 색의 옷까지 먹지 않도록 상한)
  const colorRange = Math.max(28, Math.min(70, spread * 3 + 20));

  const pass = (i: number): boolean => {
    if (grad[i] >= gradThreshold) return false;
    const d = Math.hypot(rgb[i * 3] - bg[0], rgb[i * 3 + 1] - bg[1], rgb[i * 3 + 2] - bg[2]);
    return d < colorRange;
  };

  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);
  let head = 0;
  let tail = 0;
  const seed = (i: number): void => {
    if (!visited[i] && pass(i)) {
      visited[i] = 1;
      queue[tail++] = i;
    }
  };
  for (const i of idx) seed(i);
  while (head < tail) {
    const i = queue[head++];
    const x = i % w;
    if (x > 0) seed(i - 1);
    if (x < w - 1) seed(i + 1);
    if (i >= w) seed(i - w);
    if (i < w * (h - 1)) seed(i + w);
  }

  let mask: Uint8Array = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = visited[i] ? 0 : 1;
  // 벽 픽셀이 옷 둘레에 1~2px 테두리로 남으므로 한 번 깎고, 잡티를 지운다.
  mask = erode(mask, w, h);
  mask = open(mask, w, h);
  return { mask: largestComponent(mask, w, h), bg, spread, gradThreshold };
}

function erode(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      out[i] = mask[i] & mask[i - 1] & mask[i + 1] & mask[i - w] & mask[i + w];
    }
  }
  return out;
}

function dilate(mask: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      out[i] = mask[i] | (x > 0 ? mask[i - 1] : 0) | (x < w - 1 ? mask[i + 1] : 0) | (y > 0 ? mask[i - w] : 0) | (y < h - 1 ? mask[i + w] : 0);
    }
  }
  return out;
}

/** 열림(깎고 다시 불림): 가장자리의 1px 가시·잡티 제거 */
export function open(mask: Uint8Array, w: number, h: number): Uint8Array {
  return dilate(erode(mask, w, h), w, h);
}

/** 가장 큰 연결 영역만 남긴다(먼지·그림자 조각 제거). */
export function largestComponent(mask: Uint8Array, w: number, h: number): Uint8Array {
  const label = new Int32Array(w * h);
  const queue = new Int32Array(w * h);
  let best = 0;
  let bestSize = 0;
  let next = 1;
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || label[s]) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = s;
    label[s] = next;
    while (head < tail) {
      const i = queue[head++];
      const x = i % w;
      const nb = [x > 0 ? i - 1 : -1, x < w - 1 ? i + 1 : -1, i >= w ? i - w : -1, i < w * (h - 1) ? i + w : -1];
      for (const j of nb) {
        if (j >= 0 && mask[j] && !label[j]) {
          label[j] = next;
          queue[tail++] = j;
        }
      }
    }
    if (tail > bestSize) {
      bestSize = tail;
      best = next;
    }
    next++;
  }
  const out = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) out[i] = label[i] === best ? 1 : 0;
  return out;
}
