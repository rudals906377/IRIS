// 상품 자산: 이미지 + 부위 라벨 지도 + 기준점 → 부위별 변형용 격자 메쉬.
// 상품을 바꿀 때 한 번만 계산하고(캐시), 매 프레임에는 메쉬 정점의 화면 좌표만 다시 계산한다.

import type { Vec2 } from './math.ts';

export type Category = 'top' | 'bottom' | 'hat' | 'gloves';

export interface ProductInfo {
  id: string;
  name: string;
  category: Category;
  image: string;
  thumb: string;
  parts: string;
  size: [number, number];
  keypoints: Record<string, [number, number]>;
  sleeve?: 'short' | 'long' | null;
  credit?: string;
}

export interface Catalog {
  version: number;
  products: ProductInfo[];
}

/** 부위 번호(라벨 지도 값 / 80). */
/** 4 = 목 안쪽(뒷깃 안감): 원래 옷이 보이는 곳만 덮고 목(피부)은 가리지 않는다. */
export const PART = { none: 0, torso: 1, sleeveL: 2, sleeveR: 3, neckInner: 4 } as const;

export interface PartMesh {
  partId: number;
  /** 정점의 상품 이미지 픽셀 좌표(x, y 반복). */
  src: Float32Array;
  /** 정점의 화면 픽셀 좌표. 매 프레임 갱신. */
  dst: Float32Array;
  indices: Uint16Array;
  vertexCount: number;
}

export interface GarmentAsset {
  info: ProductInfo;
  image: ImageBitmap | HTMLImageElement;
  labels: Uint8Array;
  width: number;
  height: number;
  kp: Record<string, Vec2>;
  meshes: PartMesh[];
}

const CELL = 18; // 격자 한 칸 크기(상품 이미지 픽셀)

export async function loadCatalog(baseUrl: string): Promise<Catalog> {
  const res = await fetch(new URL('catalog.json', baseUrl));
  if (!res.ok) throw new Error(`상품 목록을 불러오지 못했습니다 (${res.status})`);
  return (await res.json()) as Catalog;
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.decoding = 'async';
  img.src = url;
  await img.decode();
  return img;
}

function readPixels(img: HTMLImageElement, w: number, h: number): Uint8ClampedArray {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

export async function loadGarment(info: ProductInfo, baseUrl: string): Promise<GarmentAsset> {
  const [image, partsImg] = await Promise.all([
    loadImage(new URL(info.image, baseUrl).href),
    loadImage(new URL(info.parts, baseUrl).href),
  ]);
  const [w, h] = info.size;
  const px = readPixels(partsImg, w, h);
  const labels = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) labels[i] = Math.round(px[i * 4] / 80);
  const kp: Record<string, Vec2> = {};
  for (const [k, [x, y]] of Object.entries(info.keypoints)) kp[k] = { x, y };
  return { info, image, labels, width: w, height: h, kp, meshes: buildMeshes(labels, w, h) };
}

/** 부위마다, 그 부위 픽셀이 하나라도 있는 격자 칸만 모아 삼각형 메쉬를 만든다. */
export function buildMeshes(labels: Uint8Array, w: number, h: number): PartMesh[] {
  const cols = Math.ceil(w / CELL);
  const rows = Math.ceil(h / CELL);
  const partIds = new Set<number>();
  for (const l of labels) if (l > 0) partIds.add(l);

  const meshes: PartMesh[] = [];
  for (const partId of [...partIds].sort()) {
    // 칸 점유 여부(가장자리 보간을 위해 이웃 한 칸까지 포함)
    const occ = new Uint8Array(cols * rows);
    for (let y = 0; y < h; y++) {
      const cy = Math.floor(y / CELL);
      for (let x = 0; x < w; x++) {
        if (labels[y * w + x] !== partId) continue;
        const cx = Math.floor(x / CELL);
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx >= 0 && ny >= 0 && nx < cols && ny < rows) occ[ny * cols + nx] = 1;
          }
        }
        x = (cx + 1) * CELL - 1; // 같은 칸의 나머지 픽셀은 건너뛴다
      }
    }
    const vertIndex = new Int32Array((cols + 1) * (rows + 1)).fill(-1);
    const src: number[] = [];
    const idx: number[] = [];
    const vid = (gx: number, gy: number): number => {
      const k = gy * (cols + 1) + gx;
      if (vertIndex[k] < 0) {
        vertIndex[k] = src.length / 2;
        src.push(Math.min(w, gx * CELL), Math.min(h, gy * CELL));
      }
      return vertIndex[k];
    };
    for (let cy = 0; cy < rows; cy++) {
      for (let cx = 0; cx < cols; cx++) {
        if (!occ[cy * cols + cx]) continue;
        const a = vid(cx, cy);
        const b = vid(cx + 1, cy);
        const c = vid(cx, cy + 1);
        const d = vid(cx + 1, cy + 1);
        idx.push(a, b, c, b, d, c);
      }
    }
    if (src.length / 2 > 65535) throw new Error('메쉬 정점 수 초과');
    meshes.push({
      partId,
      src: new Float32Array(src),
      dst: new Float32Array(src.length),
      indices: new Uint16Array(idx),
      vertexCount: src.length / 2,
    });
  }
  return meshes;
}
