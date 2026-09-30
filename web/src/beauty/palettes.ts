// 메이크업 기본 색상표와 룩(여러 부위 조합). 색은 sRGB 0~1.
// 특정 브랜드 제품색이 아니라 흔히 쓰는 색 계열을 대표하는 값이다.

import type { RGB } from './makeup.ts';

export type PartName = 'lip' | 'shadow' | 'blush' | 'liner' | 'brow' | 'hair' | 'nail';

export const PART_LABELS: Record<PartName, string> = {
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
  lip: [
    { name: '로지 핑크', color: c('#c7545f') },
    { name: '코랄', color: c('#e0664e') },
    { name: '레드', color: c('#b81d2e') },
    { name: '누드 베이지', color: c('#b77a6a') },
    { name: '말린 장미', color: c('#a5535c') },
    { name: '플럼', color: c('#7e2e4a') },
    { name: '오렌지', color: c('#e2552a') },
  ],
  shadow: [
    { name: '브라운', color: c('#8a5a44') },
    { name: '로즈 골드', color: c('#c98a7a') },
    { name: '피치', color: c('#e0a07e') },
    { name: '핑크', color: c('#d98fa0') },
    { name: '스모키 그레이', color: c('#4a4550') },
    { name: '카키', color: c('#6e6a45') },
  ],
  blush: [
    { name: '피치', color: c('#f2a08a') },
    { name: '베이비 핑크', color: c('#f08ca0') },
    { name: '코랄', color: c('#f07a60') },
    { name: '로즈', color: c('#d4707f') },
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
  ],
};

export type LookName = 'daily' | 'coral' | 'red' | 'smoky' | 'rose' | 'clear';

type PartValue = { color: RGB; amount: number };

export const LOOKS: Record<LookName, { label: string; parts: Partial<Record<PartName, PartValue>>; gloss?: number }> = {
  daily: {
    label: '데일리',
    parts: {
      lip: { color: c('#c7545f'), amount: 0.6 },
      shadow: { color: c('#8a5a44'), amount: 0.35 },
      blush: { color: c('#f2a08a'), amount: 0.3 },
      brow: { color: c('#5a4030'), amount: 0.25 },
    },
    gloss: 0.3,
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
      lip: { color: c('#a5535c'), amount: 0.7 },
      shadow: { color: c('#c98a7a'), amount: 0.45 },
      blush: { color: c('#d4707f'), amount: 0.35 },
      liner: { color: c('#3b2a22'), amount: 0.5 },
    },
    gloss: 0.3,
  },
  clear: { label: '맨얼굴', parts: {} },
};
