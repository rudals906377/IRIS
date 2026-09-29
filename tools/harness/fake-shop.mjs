// 나이키·아디다스 상세페이지 구조를 흉내 낸 시험용 쇼핑몰 페이지를 만든다(사진은 testdata 것을 씀, 커밋 안 됨).
// - 나이키형: 대표 이미지(og:image)가 모델 착용 사진, 갤러리 썸네일은 작게, 큰 사진은 지연 로딩(data-src)
// - 아디다스형: JSON-LD Product.image 목록, 파일명에 _laydown/_model, <picture><source srcset>
// - 둘 다 아래쪽에 다른 상품 추천 영역(다른 상품 링크 안의 큰 사진)
// 사용법: node fake-shop.mjs  → web/public/testdata/shop/{nike,adidas,model}.html
import { copyFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const td = resolve(new URL('../../web/public/testdata', import.meta.url).pathname);
const out = resolve(td, 'shop');
mkdirSync(out, { recursive: true });
// 아디다스식 파일명으로 복사(파일명 규칙 시험용)
copyFileSync(resolve(td, 'human/00121_00.jpg'), resolve(out, 'Tee_Black_XX0001_21_model.jpg'));
copyFileSync(resolve(td, 'cloth/tee-navy-logo.jpg'), resolve(out, 'Tee_Black_XX0001_02_laydown.jpg'));
copyFileSync(resolve(td, 'cloth/09163_00.jpg'), resolve(out, 'Tee_Black_XX0001_01_laydown.jpg'));

const recos = ['cloth/09176_00.jpg', 'cloth/09305_00.jpg', 'cloth/tee-red.jpg', 'cloth/04469_00.jpg']
  .map((u, i) => `<a href="/other-product-${i}"><img src="../${u}" width="260" height="340" alt="추천 ${i}"></a>`)
  .join('');
const page = (title, head, body) => `<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>${title}</title>${head}
<style>body{font-family:sans-serif;margin:0}header{height:60px;background:#111;color:#fff}main{display:flex;gap:24px;padding:24px}
.gallery{display:flex;gap:8px}.thumbs{display:flex;flex-direction:column;gap:6px}.thumbs img{width:60px;height:60px;object-fit:cover}
.hero img{width:560px;height:700px;object-fit:contain;background:#f5f5f5}.reco{margin-top:900px;display:flex;gap:12px;padding:24px}</style></head>
<body><header><img src="../cloth/tee-white.jpg" width="40" height="40" alt="로고"></header>${body}<section class="reco"><h2>추천 상품</h2>${recos}</section></body></html>`;

// 나이키형: 첫 사진이 모델 착용, 옷만 찍힌 사진은 3번째(지연 로딩)
writeFileSync(resolve(out, 'nike.html'), page('나이키형 상품', `<meta property="og:image" content="../human/00055_00.jpg">`, `<main><div class="gallery">
<div class="thumbs"><img src="../human/00055_00.jpg"><img src="../human/00034_00.jpg"><img src="../cloth/09163_00.jpg"></div>
<div class="hero"><img src="../human/00055_00.jpg" alt="대표"></div>
<div class="hero"><img data-src="../cloth/09163_00.jpg" alt="상품 컷"></div></div><h1>티셔츠</h1></main>`));

// 아디다스형: JSON-LD 이미지 목록(모델 사진이 먼저), 파일명 규칙, picture/source
const ld = { '@context': 'https://schema.org', '@type': 'Product', name: '티셔츠', image: ['Tee_Black_XX0001_21_model.jpg', 'Tee_Black_XX0001_02_laydown.jpg', 'Tee_Black_XX0001_01_laydown.jpg'].map((f) => new URL(f, 'http://localhost:4173/testdata/shop/').href) };
writeFileSync(resolve(out, 'adidas.html'), page('아디다스형 상품', `<meta property="og:image" content="Tee_Black_XX0001_21_model.jpg"><script type="application/ld+json">${JSON.stringify(ld)}</script>`, `<main><div class="gallery">
<div class="hero"><picture><source srcset="Tee_Black_XX0001_21_model.jpg 600w"><img src="Tee_Black_XX0001_21_model.jpg" alt=""></picture></div>
<div class="hero"><picture><source data-srcset="Tee_Black_XX0001_02_laydown.jpg 600w"><img alt=""></picture></div></div><h1>티셔츠</h1></main>`));

// 모델 착용 사진만 있는 상품
writeFileSync(resolve(out, 'model.html'), page('모델 사진만 있는 상품', `<meta property="og:image" content="../human/00035_00.jpg">`, `<main><div class="hero"><img src="../human/00035_00.jpg"></div><h1>긴팔</h1></main>`));
console.log('시험 쇼핑몰:', out);
