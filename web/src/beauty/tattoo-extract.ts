// 참고 사진에서 타투 도안 뽑기(순수 계산, 단위 테스트 대상).
//
// 1) 잉크 찾기: 피부 위에서 '주변 피부보다 뚜렷이 어두운' 픽셀(국소 대비). 전체 피부 평균과 비교하면
//    그늘진 피부가 잉크로, 밝은 조명 아래 잉크가 피부로 잘못 잡힌다.
// 2) 잘못 잡힌 것 빼기: 새까맣고 큰 덩어리(검은 끈·머리카락이 피부로 분류된 것). 잉크 선은 피부와 섞여 회색으로 찍힌다.
// 3) 도안 묶기: 나비 여러 마리 + 점처럼 흩어진 도안을 넓혀서 한 무리로 묶고, 잉크가 가장 많은 무리를 고른다.
//    사용자가 영역을 직접 고르면 이 단계를 건너뛴다.
// 4) 도안 만들기: 픽셀마다 잉크 아래의 피부색(주변 피부로 추정)으로 나눈 '투과율'을 구해,
//    내 피부에 곱하면 사진과 같은 비율로 어두워지고 색 잉크는 색이 남는다(RGB = 잉크 색, A = 잉크 양).

export interface InkSource {
  w: number;
  h: number;
  /** RGBA 0~255 */
  d: Uint8ClampedArray;
  /** 피부 확률(0~1, w×h) */
  skin: Float32Array;
  /** 피부가 아닐 확률이 높은 곳(머리카락·옷 등, 0~1, w×h). 없으면 0 */
  avoid?: Float32Array;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface ExtractedDesign {
  /** RGBA: RGB = 피부에 곱할 잉크 색(투과율), A = 잉크 양 */
  rgba: Uint8ClampedArray<ArrayBuffer>;
  w: number;
  h: number;
  /** 원본 사진에서의 영역 */
  rect: Rect;
  /** 원본 크기 잉크 양(0~1, 생성 서버에 참고 물체 마스크로 보낼 때 씀) */
  inkMask: Float32Array;
}

const smooth = (e0: number, e1: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
};

/** 상자 흐림(가로·세로 누적합). 두 번 돌리면 가우시안에 가깝다 */
export function boxBlur(src: Float32Array, w: number, h: number, r: number): Float32Array {
  r = Math.max(1, Math.round(r));
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const k = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    let s = 0;
    for (let x = -r; x <= r; x++) s += src[y * w + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = s / k;
      s += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++) s += tmp[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = s / k;
      s += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

const blur2 = (src: Float32Array, w: number, h: number, r: number): Float32Array => boxBlur(boxBlur(src, w, h, r / 2), w, h, r / 2);

/** 8방향 연결 덩어리 번호(0 = 배경). 반환: 번호 배열과 덩어리 수 */
export function label(mask: Uint8Array, w: number, h: number): { lab: Int32Array; n: number } {
  const lab = new Int32Array(w * h);
  let n = 0;
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (!mask[i] || lab[i]) continue;
    n++;
    lab[i] = n;
    stack.push(i);
    while (stack.length) {
      const k = stack.pop()!;
      const x = k % w;
      const y = (k / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (mask[j] && !lab[j]) {
            lab[j] = n;
            stack.push(j);
          }
        }
      }
    }
  }
  return { lab, n };
}

/** 밝기(감마 공간 0~1) */
function lumaOf(src: InkSource): Float32Array {
  const L = new Float32Array(src.w * src.h);
  for (let i = 0; i < L.length; i++) L[i] = (0.299 * src.d[i * 4] + 0.587 * src.d[i * 4 + 1] + 0.114 * src.d[i * 4 + 2]) / 255;
  return L;
}

/** 잉크 후보(0/1)와 피부 마스크 */
export function findInk(src: InkSource): { ink: Uint8Array; skin: Uint8Array; L: Float32Array } {
  const { w, h } = src;
  const n = w * h;
  const short = Math.min(w, h);
  const L = lumaOf(src);
  const skinRaw = new Float32Array(n);
  for (let i = 0; i < n; i++) skinRaw[i] = src.skin[i] > 0.6 && (src.avoid ? src.avoid[i] < 0.3 : true) ? 1 : 0;
  // 피부 가장자리(옷·머리 경계의 어두운 띠)는 잉크로 오인하기 쉬우므로 조금 깎는다
  const er = boxBlur(skinRaw, w, h, Math.max(1, short * 0.008));
  const skin = new Uint8Array(n);
  for (let i = 0; i < n; i++) skin[i] = er[i] > 0.97 ? 1 : 0;
  const local = blur2(L, w, h, Math.max(4, short * 0.06));
  const ink = new Uint8Array(n);
  for (let i = 0; i < n; i++) ink[i] = skin[i] && local[i] - L[i] > 0.1 ? 1 : 0;
  // 작은 점 잡음, 새까맣고 큰 덩어리(끈·머리카락) 제거
  const { lab, n: nc } = label(ink, w, h);
  const area = new Int32Array(nc + 1);
  for (let i = 0; i < n; i++) area[lab[i]]++;
  const big = new Set<number>();
  for (let c = 1; c <= nc; c++) if (area[c] > n * 0.003) big.add(c);
  const darkOf = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const c = lab[i];
    if (c && big.has(c)) {
      let a = darkOf.get(c);
      if (!a) darkOf.set(c, (a = []));
      a.push(L[i]);
    }
  }
  const drop = new Set<number>();
  for (const [c, ls] of darkOf) {
    ls.sort((a, b) => a - b);
    if (ls[Math.floor(ls.length * 0.2)] < 0.14) drop.add(c);
  }
  for (let i = 0; i < n; i++) {
    const c = lab[i];
    if (c && (area[c] < 4 || drop.has(c))) ink[i] = 0;
  }
  return { ink, skin, L };
}

/** 흩어진 도안을 한 무리로 묶어 잉크가 가장 많은 무리의 영역을 제안한다. 없으면 null */
export function proposeRect(src: InkSource, found = findInk(src)): Rect | null {
  const { w, h } = src;
  const short = Math.min(w, h);
  const { ink } = found;
  let total = 0;
  for (let i = 0; i < ink.length; i++) total += ink[i];
  if (total < short * 0.5) return null;
  const f = new Float32Array(ink.length);
  for (let i = 0; i < ink.length; i++) f[i] = ink[i];
  const grown = boxBlur(f, w, h, Math.max(2, short * 0.02));
  const gm = new Uint8Array(ink.length);
  for (let i = 0; i < ink.length; i++) gm[i] = grown[i] > 0 ? 1 : 0;
  const { lab, n } = label(gm, w, h);
  const amt = new Int32Array(n + 1);
  for (let i = 0; i < ink.length; i++) if (ink[i]) amt[lab[i]]++;
  let best = 0;
  for (let c = 1; c <= n; c++) if (amt[c] > amt[best]) best = c;
  if (!best || amt[best] < 20) return null;
  let x0 = w;
  let y0 = h;
  let x1 = 0;
  let y1 = 0;
  for (let i = 0; i < ink.length; i++) {
    if (!ink[i] || lab[i] !== best) continue;
    const x = i % w;
    const y = (i / w) | 0;
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  const pad = Math.round(Math.max(x1 - x0, y1 - y0) * 0.06) + 2;
  x0 = Math.max(0, x0 - pad);
  y0 = Math.max(0, y0 - pad);
  x1 = Math.min(w - 1, x1 + pad);
  y1 = Math.min(h - 1, y1 + pad);
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

/**
 * 영역 안의 도안을 뽑는다. rect 가 없으면 자동 제안 영역.
 * maxSide: 도안 텍스처 최대 변(GPU 업로드 크기)
 */
export function extractDesign(src: InkSource, rect?: Rect | null, maxSide = 512): ExtractedDesign | null {
  const { w, h, d } = src;
  const n = w * h;
  const short = Math.min(w, h);
  const found = findInk(src);
  const R = rect ?? proposeRect(src, found);
  if (!R || R.w < 4 || R.h < 4) return null;
  // 잉크 아래 피부색 추정: 잉크(조금 넓힘)가 아닌 피부 픽셀만 넓게 흐려 평균
  const inkF = new Float32Array(n);
  for (let i = 0; i < n; i++) inkF[i] = found.ink[i];
  const inkWide = boxBlur(inkF, w, h, Math.max(1, short * 0.006));
  const wgt = new Float32Array(n);
  for (let i = 0; i < n; i++) wgt[i] = src.skin[i] > 0.5 && inkWide[i] === 0 ? 1 : 0;
  const r = Math.max(4, short * 0.05);
  const den = blur2(wgt, w, h, r);
  const ch: Float32Array[] = [];
  let gs = [0, 0, 0];
  let gn = 0;
  for (let c = 0; c < 3; c++) {
    const v = new Float32Array(n);
    for (let i = 0; i < n; i++) v[i] = (d[i * 4 + c] / 255) * wgt[i];
    ch.push(blur2(v, w, h, r));
  }
  for (let i = 0; i < n; i += 3) {
    if (!wgt[i]) continue;
    gs = [gs[0] + d[i * 4] / 255, gs[1] + d[i * 4 + 1] / 255, gs[2] + d[i * 4 + 2] / 255];
    gn++;
  }
  if (!gn) return null;
  const glob = gs.map((v) => v / gn);
  const ow = R.w;
  const oh = R.h;
  const inkMask = new Float32Array(n);
  const full = new Float32Array(ow * oh * 4);
  for (let y = 0; y < oh; y++) {
    for (let x = 0; x < ow; x++) {
      const i = (R.y + y) * w + (R.x + x);
      const o = (y * ow + x) * 4;
      // 주변 피부 정보가 적으면 전체 피부 평균 쪽으로
      const conf = Math.min(1, den[i] / 0.15);
      const bg = [0, 1, 2].map((c) => (den[i] > 1e-4 ? ch[c][i] / den[i] : glob[c]) * conf + glob[c] * (1 - conf));
      const T = [0, 1, 2].map((c) => Math.min(1, (d[i * 4 + c] / 255 + 0.01) / (bg[c] + 0.01)));
      const lumT = 0.299 * T[0] + 0.587 * T[1] + 0.114 * T[2];
      const dark = smooth(0.06, 0.3, 1 - lumT);
      const tint = smooth(0.08, 0.22, Math.max(Math.abs(T[0] - lumT), Math.abs(T[1] - lumT), Math.abs(T[2] - lumT)));
      // 피부가 아닌 곳(옷·머리카락·배경)은 도안이 아니다
      const onSkin = src.avoid ? 1 - smooth(0.3, 0.6, src.avoid[i]) : 1;
      const a = Math.max(dark, tint * 0.8) * onSkin;
      inkMask[i] = a;
      // 다 덮였을 때의 잉크 색: 섞인 비율만큼 되돌린다
      const k = Math.max(a, 0.25);
      for (let c = 0; c < 3; c++) full[o + c] = Math.min(1, Math.max(0, 1 - (1 - T[c]) / k));
      full[o + 3] = a;
    }
  }
  const s = Math.min(1, maxSide / Math.max(ow, oh));
  const tw = Math.max(1, Math.round(ow * s));
  const th = Math.max(1, Math.round(oh * s));
  const rgba = new Uint8ClampedArray(tw * th * 4);
  // 줄일 때는 잉크 양으로 가중 평균(잉크 없는 픽셀의 흰 색이 섞여 선이 옅어지지 않게)
  for (let y = 0; y < th; y++) {
    const sy0 = Math.floor(y / s);
    const sy1 = Math.max(sy0 + 1, Math.floor((y + 1) / s));
    for (let x = 0; x < tw; x++) {
      const sx0 = Math.floor(x / s);
      const sx1 = Math.max(sx0 + 1, Math.floor((x + 1) / s));
      let sa = 0;
      let sr = 0;
      let sg = 0;
      let sb = 0;
      let cnt = 0;
      for (let yy = sy0; yy < Math.min(oh, sy1); yy++) {
        for (let xx = sx0; xx < Math.min(ow, sx1); xx++) {
          const o = (yy * ow + xx) * 4;
          const a = full[o + 3];
          sa += a;
          sr += full[o] * a;
          sg += full[o + 1] * a;
          sb += full[o + 2] * a;
          cnt++;
        }
      }
      const o = (y * tw + x) * 4;
      const a = cnt ? sa / cnt : 0;
      rgba[o] = sa > 1e-4 ? (sr / sa) * 255 : 255;
      rgba[o + 1] = sa > 1e-4 ? (sg / sa) * 255 : 255;
      rgba[o + 2] = sa > 1e-4 ? (sb / sa) * 255 : 255;
      rgba[o + 3] = a * 255;
    }
  }
  return { rgba, w: tw, h: th, rect: R, inkMask };
}
