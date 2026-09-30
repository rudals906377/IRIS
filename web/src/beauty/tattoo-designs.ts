// 타투 도안: 저작권 문제가 없도록 IRIS가 직접 그린 선화(SVG 경로 → 캔버스).
// 도안 이미지는 RGBA: RGB 잉크 색, A 잉크 양. 사용자가 올린 사진도 같은 형식으로 바꾼다.

export interface TattooDesign {
  id: string;
  name: string;
  /** 가로 / 세로 */
  aspect: number;
  canvas: HTMLCanvasElement;
}

const SIZE = 512;

type Draw = (ctx: CanvasRenderingContext2D) => void;

/** 100 × (100/aspect) 좌표계에서 그린다. */
function make(id: string, name: string, aspect: number, draw: Draw): TattooDesign {
  const canvas = document.createElement('canvas');
  canvas.width = aspect >= 1 ? SIZE : Math.round(SIZE * aspect);
  canvas.height = aspect >= 1 ? Math.round(SIZE / aspect) : SIZE;
  const ctx = canvas.getContext('2d')!;
  const s = canvas.width / 100;
  ctx.scale(s, s);
  ctx.strokeStyle = '#000';
  ctx.fillStyle = '#000';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = 2.2;
  draw(ctx);
  return { id, name, aspect, canvas };
}

const stroke = (ctx: CanvasRenderingContext2D, d: string, w?: number): void => {
  if (w) ctx.lineWidth = w;
  ctx.stroke(new Path2D(d));
};
const fill = (ctx: CanvasRenderingContext2D, d: string): void => ctx.fill(new Path2D(d));

/** 네 갈래 반짝이 별 */
const sparkle = (x: number, y: number, r: number): string =>
  `M${x} ${y - r} Q${x} ${y} ${x + r} ${y} Q${x} ${y} ${x} ${y + r} Q${x} ${y} ${x - r} ${y} Q${x} ${y} ${x} ${y - r} Z`;

export function builtinDesigns(): TattooDesign[] {
  return [
    make('moon', '달과 별', 0.8, (ctx) => {
      // 초승달: 바깥 반원(반지름 40) + 안쪽의 덜 휜 원호(반지름 48). 두 원호 반지름이 같으면 면적이 0이 된다
      fill(ctx, 'M58 22 A40 40 0 0 0 58 102 A48 48 0 0 1 58 22 Z');
      fill(ctx, sparkle(70, 46, 9));
      fill(ctx, sparkle(80, 74, 5));
      fill(ctx, sparkle(66, 96, 3.5));
    }),
    make('butterfly', '나비', 1.2, (ctx) => {
      const wings = [
        'M50 40 C38 8, 8 4, 6 26 C4 42, 30 46, 50 42',
        'M50 44 C32 46, 12 56, 18 72 C24 82, 44 66, 50 46',
        'M50 40 C62 8, 92 4, 94 26 C96 42, 70 46, 50 42',
        'M50 44 C68 46, 88 56, 82 72 C76 82, 56 66, 50 46',
      ];
      for (const w of wings) stroke(ctx, w, 2);
      // 날개 무늬
      stroke(ctx, 'M22 22 C28 26, 34 32, 40 38 M78 22 C72 26, 66 32, 60 38 M28 64 C34 58, 40 52, 46 48 M72 64 C66 58, 60 52, 54 48', 1.2);
      // 몸통·더듬이
      stroke(ctx, 'M50 34 L50 62', 3);
      stroke(ctx, 'M50 34 C47 26, 44 22, 40 20 M50 34 C53 26, 56 22, 60 20', 1.3);
    }),
    make('wave', '파도', 1.5, (ctx) => {
      stroke(ctx, 'M4 58 C14 30, 38 14, 58 22 C72 28, 72 46, 58 48 C48 49, 46 38, 54 36', 2.4);
      stroke(ctx, 'M4 58 C22 52, 34 56, 46 58 C60 60, 76 50, 96 56', 2);
      stroke(ctx, 'M14 50 C22 36, 34 28, 46 28 M22 54 C28 44, 36 38, 44 36', 1.3);
      // 물보라
      for (const [x, y, r] of [[70, 14, 2], [78, 20, 1.6], [84, 12, 1.3], [64, 8, 1.2]]) fill(ctx, `M${x + r} ${y} A${r} ${r} 0 1 1 ${x - r} ${y} A${r} ${r} 0 1 1 ${x + r} ${y} Z`);
    }),
    make('heartbeat', '하트 비트', 2.4, (ctx) => {
      stroke(ctx, 'M2 22 L26 22 L31 12 L37 34 L43 4 L49 30 L53 22 L64 22', 1.8);
      // 작은 하트
      fill(ctx, 'M78 30 C66 22, 64 12, 71 10 C75 9, 77 12, 78 14 C79 12, 81 9, 85 10 C92 12, 90 22, 78 30 Z');
      stroke(ctx, 'M88 22 L98 22', 1.8);
    }),
    make('flower', '들꽃', 0.6, (ctx) => {
      // 꽃잎 5장
      for (let k = 0; k < 5; k++) {
        const a = (k / 5) * Math.PI * 2 - Math.PI / 2;
        ctx.save();
        ctx.translate(50, 34);
        ctx.rotate(a);
        stroke(ctx, 'M0 0 C8 -6, 22 -6, 26 0 C22 6, 8 6, 0 0 Z', 1.8);
        ctx.restore();
      }
      fill(ctx, 'M54 34 A4 4 0 1 1 46 34 A4 4 0 1 1 54 34 Z');
      // 줄기·잎
      stroke(ctx, 'M50 60 C48 90, 54 120, 50 160', 1.8);
      stroke(ctx, 'M51 100 C62 90, 74 92, 78 96 C70 104, 58 104, 51 100 Z', 1.5);
      stroke(ctx, 'M50 126 C40 116, 28 118, 24 122 C32 130, 44 130, 50 126 Z', 1.5);
    }),
    make('mountain', '산과 해', 1, (ctx) => {
      stroke(ctx, 'M50 4 A46 46 0 1 1 49.9 4', 2);
      stroke(ctx, 'M10 72 L34 40 L46 56 L62 30 L90 72', 2);
      stroke(ctx, 'M62 30 L56 46 L64 42 L68 48', 1.3);
      stroke(ctx, 'M66 24 A8 8 0 1 1 65.9 24', 1.6);
      stroke(ctx, 'M20 82 L80 82 M28 88 L72 88', 1.4);
    }),
    make('lettering', '레터링', 3.2, (ctx) => {
      ctx.font = 'italic 20px "Times New Roman", Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('carpe diem', 50, 16);
    }),
  ];
}

/** 사용자가 올린 사진을 도안으로: 투명 배경이면 그대로, 아니면 밝은 바탕을 지우고 어두운 선만 남긴다. */
export async function designFromFile(file: File): Promise<TattooDesign> {
  const bmp = await createImageBitmap(file);
  const scale = Math.min(1, SIZE / Math.max(bmp.width, bmp.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bmp.width * scale));
  canvas.height = Math.max(1, Math.round(bmp.height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = img.data;
  let transparent = 0;
  for (let i = 3; i < d.length; i += 4) if (d[i] < 250) transparent++;
  if (transparent < (d.length / 4) * 0.02) {
    // 흰(밝은) 바탕 → 밝기가 잉크 양이 된다
    for (let i = 0; i < d.length; i += 4) {
      const L = (0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]) / 255;
      d[i + 3] = Math.round(255 * Math.min(1, Math.max(0, (0.92 - L) / 0.6)));
    }
    ctx.putImageData(img, 0, 0);
  }
  return { id: 'upload', name: '내 도안', aspect: canvas.width / canvas.height, canvas };
}
