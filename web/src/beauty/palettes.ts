// 메이크업 기본 색상표와 룩(여러 부위 조합). 색은 sRGB 0~1.
// 특정 브랜드 제품색이 아니라 흔히 쓰는 색 계열을 대표하는 값이다.

import type { RGB } from './makeup.ts';

export type PartName = 'base' | 'contour' | 'lip' | 'shadow' | 'blush' | 'liner' | 'brow' | 'hair' | 'nail';

export const PART_LABELS: Record<PartName, string> = {
  base: '피부',
  contour: '윤곽',
  lip: '립',
  shadow: '아이섀도',
  blush: '블러셔',
  liner: '아이라이너',
  brow: '눈썹',
  hair: '헤어',
  nail: '네일',
};

const c = (hex: string): RGB => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255) as RGB;

/** tip: 끝 색(뿌리 → 끝 그라데이션, 헤어만) */
export const PALETTES: Record<PartName, { name: string; color: RGB; tip?: RGB }[]> = {
  // 피부 보정: 파운데이션 호수(한국식 호수 표기는 대략적인 밝기 구분). 보정 세기는 진하기 슬라이더
  // 호수 색은 색감(언더톤)과 ±8% 밝기로만 반영된다. '톤 유지'는 기준 피부색이라 잡티·톤 정리만 한다
  base: [
    { name: '톤 유지', color: c('#d1a38a') },
    { name: '17호 라이트', color: c('#f0d6c2') },
    { name: '21호 내추럴', color: c('#e6c4a8') },
    { name: '23호 베이지', color: c('#d8b394') },
    { name: '25호 웜', color: c('#c79d78') },
    { name: '핑크 톤업', color: c('#f0cfc6') },
    { name: '딥 브라운', color: c('#9c6f50') },
  ],
  // 윤곽: 쉐딩 색(하이라이터는 자동). 쿨 톤 토프가 자연스러운 그림자에 가깝다
  contour: [
    { name: '쿨 토프', color: c('#8c7a70') },
    { name: '소프트 브라운', color: c('#9a7862') },
    { name: '웜 브론즈', color: c('#a06c48') },
    { name: '딥', color: c('#6a5044') },
  ],
  lip: [
    { name: '로지 핑크', color: c('#c7545f') },
    { name: '코랄', color: c('#e0664e') },
    { name: '레드', color: c('#b81d2e') },
    { name: '누드 베이지', color: c('#b77a6a') },
    { name: '말린 장미', color: c('#a5535c') },
    { name: '플럼', color: c('#7e2e4a') },
    { name: '오렌지', color: c('#e2552a') },
    // 참고 사진(아이돌 화보 21장)의 입술 색을 재서 앱의 기준 피부·조명으로 환산한 값
    { name: '아이돌 코랄 핑크', color: c('#d4766a') },
    { name: 'MLBB 로즈', color: c('#c87a71') },
    { name: '누드 로즈', color: c('#a97167') },
    { name: '칠리 레드', color: c('#c4564c') },
    { name: '브릭', color: c('#a05147') },
    { name: '베리', color: c('#ab3f44') },
    { name: '딥 레드', color: c('#882c2d') },
  ],
  shadow: [
    { name: '브라운', color: c('#8a5a44') },
    { name: '로즈 골드', color: c('#c98a7a') },
    { name: '피치', color: c('#e0a07e') },
    { name: '핑크', color: c('#d98fa0') },
    { name: '스모키 그레이', color: c('#4a4550') },
    { name: '카키', color: c('#6e6a45') },
    { name: '샴페인 골드', color: c('#b98a5e') },
    { name: '버건디', color: c('#6e2c38') },
  ],
  blush: [
    { name: '피치', color: c('#f2a08a') },
    { name: '베이비 핑크', color: c('#f08ca0') },
    { name: '코랄', color: c('#f07a60') },
    { name: '로즈', color: c('#d4707f') },
    // 참고 사진의 볼/이마 피부 색 비율을 재서 기본 진하기(0.6)에서 사진처럼 보이게 환산한 값
    { name: '아이돌 코랄', color: c('#cb795e') },
    { name: '쿨 핑크', color: c('#cb7979') },
    { name: '피치 플러시', color: c('#d0835f') },
    { name: '라즈베리', color: c('#c66264') },
    { name: '웜 로즈', color: c('#bc6649') },
  ],
  liner: [
    { name: '블랙', color: c('#1a1414') },
    { name: '브라운', color: c('#3b2a22') },
    { name: '네이비', color: c('#1f2a44') },
  ],
  brow: [
    { name: '다크 브라운', color: c('#3a2a20') },
    { name: '브라운', color: c('#5a4030') },
    { name: '그레이', color: c('#454040') },
  ],
  // 헤어: 머리카락 **평균** 색 기준(각 올은 원래 명암만큼 밝거나 어둡게 바뀐다)
  hair: [
    { name: '초코 브라운', color: c('#4a3024') },
    { name: '애쉬 브라운', color: c('#6b5a4e') },
    { name: '밀크 브라운', color: c('#8a6a50') },
    { name: '블론드', color: c('#c9a26a') },
    { name: '애쉬 그레이', color: c('#8a8a8e') },
    { name: '와인', color: c('#6e1e2a') },
    { name: '체리 레드', color: c('#a0282a') },
    { name: '핑크', color: c('#d88aa0') },
    { name: '라벤더', color: c('#9a86b8') },
    { name: '블루 블랙', color: c('#1c2230') },
    { name: '옴브레 브라운', color: c('#3a2a22'), tip: c('#c9a26a') },
    { name: '옴브레 핑크', color: c('#2a2226'), tip: c('#d88aa0') },
    // 참고 사진(미용실 게시물 32장)의 머리카락 평균색을 분할로 잰 값
    { name: '베이지 블론드', color: c('#ab9077') },
    { name: '퓨어 베이지', color: c('#8e746a') },
    { name: '애쉬 핑크 베이지', color: c('#a69291') },
    { name: '딸기우유 블론드', color: c('#b49895') },
    { name: '라벤더 그레이', color: c('#968280') },
    { name: '실버 그레이', color: c('#858076') },
    { name: '차콜 그레이', color: c('#424143') },
    { name: '카키 블론드', color: c('#877e5b') },
    { name: '옐로 블론드', color: c('#c1a479') },
    { name: '오렌지 레드', color: c('#e1391f') },
    { name: '투톤 오렌지', color: c('#c26f1e'), tip: c('#c9a828') },
  ],
  nail: [
    { name: '레드', color: c('#b3122a') },
    { name: '누드 핑크', color: c('#e3b3ab') },
    { name: '코랄', color: c('#ec6a55') },
    { name: '버건디', color: c('#5c1424') },
    { name: '라벤더', color: c('#b9a3d9') },
    { name: '민트', color: c('#9fdcc6') },
    { name: '네이비', color: c('#1d2a52') },
    { name: '블랙', color: c('#161416') },
    { name: '밀키 화이트', color: c('#f1ece6') },
    // 참고 사진(네일 27장)에서 많이 보인 색
    { name: '젤리 블루', color: c('#40637b') },
    { name: '아이스 블루', color: c('#9fb3cf') },
    { name: '라일락 그레이', color: c('#afa9ba') },
    { name: '실버', color: c('#c3c3bf') },
    { name: '코발트', color: c('#2d4a8e') },
    { name: '핑크 라일락', color: c('#c6b4be') },
    { name: '스모키 모브', color: c('#7f7684') },
  ],
};

export type LookName = 'daily' | 'idol' | 'coral' | 'red' | 'smoky' | 'rose' | 'glam' | 'clear';

type PartValue = { color: RGB; amount: number };

export const LOOKS: Record<LookName, { label: string; parts: Partial<Record<PartName, PartValue>>; gloss?: number; pearl?: number; lipStyle?: 'full' | 'gradient' | 'blur' }> = {
  daily: {
    label: '데일리',
    parts: {
      base: { color: c('#d1a38a'), amount: 0.45 },
      lip: { color: c('#c7545f'), amount: 0.6 },
      shadow: { color: c('#8a5a44'), amount: 0.35 },
      blush: { color: c('#f2a08a'), amount: 0.3 },
      brow: { color: c('#5a4030'), amount: 0.25 },
    },
    gloss: 0.3,
  },
  // 참고 사진에서 가장 많이 보인 조합: 블러 립 + 눈 밑까지 넓은 코랄 핑크 블러셔 + 옅은 로즈 섀도
  idol: {
    label: '아이돌',
    parts: {
      base: { color: c('#f0cfc6'), amount: 0.45 },
      lip: { color: c('#d4766a'), amount: 0.75 },
      shadow: { color: c('#c98a7a'), amount: 0.35 },
      // 참고 사진과 같은 붉어짐(볼/이마 초록·파랑 비율 약 0.87배)이 되도록 측정해 맞춘 진하기
      blush: { color: c('#cb795e'), amount: 0.9 },
      liner: { color: c('#3b2a22'), amount: 0.4 },
      brow: { color: c('#5a4030'), amount: 0.25 },
    },
    gloss: 0.5,
    lipStyle: 'blur',
  },
  coral: {
    label: '코랄',
    parts: {
      lip: { color: c('#e0664e'), amount: 0.7 },
      shadow: { color: c('#e0a07e'), amount: 0.4 },
      blush: { color: c('#f07a60'), amount: 0.35 },
      liner: { color: c('#3b2a22'), amount: 0.6 },
    },
    gloss: 0.45,
  },
  red: {
    label: '레드 립',
    parts: {
      lip: { color: c('#b81d2e'), amount: 0.85 },
      liner: { color: c('#1a1414'), amount: 0.85 },
      brow: { color: c('#3a2a20'), amount: 0.3 },
    },
    gloss: 0.15,
  },
  smoky: {
    label: '스모키',
    parts: {
      lip: { color: c('#b77a6a'), amount: 0.55 },
      shadow: { color: c('#4a4550'), amount: 0.6 },
      liner: { color: c('#1a1414'), amount: 0.9 },
      brow: { color: c('#3a2a20'), amount: 0.35 },
    },
    gloss: 0.2,
  },
  rose: {
    label: '로즈',
    parts: {
      base: { color: c('#f0cfc6'), amount: 0.5 },
      lip: { color: c('#a5535c'), amount: 0.7 },
      shadow: { color: c('#c98a7a'), amount: 0.45 },
      blush: { color: c('#d4707f'), amount: 0.35 },
      liner: { color: c('#3b2a22'), amount: 0.5 },
    },
    gloss: 0.3,
  },
  glam: {
    label: '글램',
    parts: {
      base: { color: c('#d1a38a'), amount: 0.6 },
      contour: { color: c('#8c7a70'), amount: 0.55 },
      lip: { color: c('#9e3a4a'), amount: 0.8 },
      shadow: { color: c('#b98a5e'), amount: 0.6 },
      blush: { color: c('#d4707f'), amount: 0.3 },
      liner: { color: c('#1a1414'), amount: 0.9 },
      brow: { color: c('#3a2a20'), amount: 0.35 },
    },
    gloss: 0.55,
    pearl: 0.8,
  },
  clear: { label: '맨얼굴', parts: {} },
};
