// 뷰티 스타일 AI(별도 저장소 rudals906377/AI의 분류기)가 알려 주는 속성(한국어 라벨) → 앱 설정 힌트.
// 색은 사진에서 직접 잰 값(photo-style.ts)을 쓰고, 여기서는 "어떤 종류인가"(립 모양, 섀도 종류, 아이라인 유무,
// 네일 무늬·길이, 헤어 기법, 타투 부위·크기)를 정한다. 순수 계산이라 단위 테스트 대상.

import type { NailStyle } from './nail.ts';
import type { TattooPlace } from './tattoo-place.ts';

/** 분류기 결과의 속성 한 줄(JSON 출력 형식과 같다) */
export interface StyleAttr {
  group: string;
  label: string;
  /** 모델에 넣는 영어 설명(생성 모드의 글 설명에 쓴다) */
  label_en?: string;
  score: number;
  level?: 'high' | 'mid' | 'low';
}

export interface StyleAIResult {
  category: 'hair' | 'nail' | 'makeup' | 'tattoo' | string;
  attributes: StyleAttr[];
  secondary?: { category: string; attributes: StyleAttr[] } | null;
  is_beauty?: boolean;
  headline?: string;
  description_ko?: string;
}

export interface MakeupHints {
  lipStyle?: 'full' | 'gradient' | 'blur';
  /** 오버립이면 0.6 */
  overlip?: number;
  gloss?: 'glossy' | 'matte';
  /** 섀도 종류: 음영·스모키는 어두운 색, 펄은 반짝임, 핑크·코랄은 색조 */
  shadow?: 'shade' | 'smoky' | 'pearl' | 'color' | 'pink';
  /** 아이라인이 뚜렷한 스타일(캣아이) */
  liner?: 'cat' | 'soft';
  /** 애교살 강조: 아래 속눈썹 밑을 밝게(펄) */
  underEye?: boolean;
  blush?: 'flush' | 'coral' | 'none';
  contour?: 'shading' | 'highlight';
  /** 피부 표현 */
  base?: 'dewy' | 'semi' | 'matte';
  /** 무드(룩 이름 힌트) */
  mood?: string;
}

export interface HairHints {
  /** 옴브레·발레아쥬 → 끝이 밝은 그라데이션, 브릿지 → 밝은 가닥 */
  tech?: 'ombre' | 'balayage' | 'highlights' | 'inner' | 'twotone' | 'solid';
  tone?: 'cool' | 'warm' | 'neutral';
  colorLabel?: string;
}

export interface NailHints {
  style?: NailStyle;
  /** 0 자연 ~ 1 롱 */
  length?: number;
  shape?: string;
  colorLabel?: string;
}

export interface TattooHints {
  place?: TattooPlace;
  /** 0~1 */
  size?: number;
  /** 컬러 타투면 사진 색 그대로, 블랙이면 검정 잉크 */
  color?: 'black' | 'grey' | 'color';
  styleLabel?: string;
}

export interface StyleHints {
  category: string;
  makeup?: MakeupHints;
  hair?: HairHints;
  nail?: NailHints;
  tattoo?: TattooHints;
  /** 확신이 낮은(0.5 미만) 속성은 힌트에서 뺀 목록(설명용) */
  weak: string[];
}

// 분류기의 보정 확률: 0.5 이상이면 "~로 보입니다" 수준(적중 약 70%). 힌트는 0.4부터 쓰되 색은 늘 측정값을 쓴다
const MIN_SCORE = 0.4;

function pick(attrs: StyleAttr[], group: string): StyleAttr | undefined {
  return attrs.find((a) => a.group === group);
}

/** 분류기 결과를 앱 힌트로 바꾼다. 확신이 낮은 속성은 무시하고 weak에 적는다. */
export function hintsFromStyleAI(r: StyleAIResult): StyleHints {
  const out: StyleHints = { category: r.category, weak: [] };
  const consider = (attrs: StyleAttr[], cat: string): void => {
    const get = (group: string): string | undefined => {
      const a = pick(attrs, group);
      if (!a) return undefined;
      if (a.score < MIN_SCORE) {
        out.weak.push(`${a.group}:${a.label}`);
        return undefined;
      }
      return a.label;
    };
    if (cat === 'makeup') {
      const m: MakeupHints = {};
      const lipT = get('lipTexture');
      if (lipT === '블러립') m.lipStyle = 'blur';
      else if (lipT === '그라데이션립') m.lipStyle = 'gradient';
      else if (lipT === '오버립') {
        m.lipStyle = 'full';
        m.overlip = 0.6;
      } else if (lipT === '글로시') m.gloss = 'glossy';
      else if (lipT === '매트') m.gloss = 'matte';
      const eye = get('eye');
      if (eye === '브라운 음영 섀도') m.shadow = 'shade';
      else if (eye === '스모키') m.shadow = 'smoky';
      else if (eye === '글리터·펄') m.shadow = 'pearl';
      else if (eye === '컬러 섀도') m.shadow = 'color';
      else if (eye === '핑크·코랄 섀도') m.shadow = 'pink';
      else if (eye === '캣아이라인') m.liner = 'cat';
      else if (eye === '강아지 눈매') m.liner = 'soft';
      else if (eye === '애교살 강조') m.underEye = true;
      const cheek = get('cheek');
      if (cheek === '홍조 블러셔') m.blush = 'flush';
      else if (cheek === '코랄 블러셔') m.blush = 'coral';
      else if (cheek === '셰이딩') m.contour = 'shading';
      else if (cheek === '하이라이터') m.contour = 'highlight';
      else if (cheek === '미니멀') m.blush = 'none';
      const base = get('base');
      if (base === '물광' || base === '윤광') m.base = 'dewy';
      else if (base === '세미매트') m.base = 'semi';
      else if (base === '보송 매트') m.base = 'matte';
      const mood = get('mood');
      if (mood) m.mood = mood;
      out.makeup = m;
    } else if (cat === 'hair') {
      const h: HairHints = {};
      const tech = get('colorTech');
      if (tech === '옴브레') h.tech = 'ombre';
      else if (tech === '발레아쥬') h.tech = 'balayage';
      else if (tech === '브릿지') h.tech = 'highlights';
      else if (tech === '이너컬러') h.tech = 'inner';
      else if (tech === '투톤') h.tech = 'twotone';
      else if (tech === '전체 염색') h.tech = 'solid';
      const tone = get('tone');
      if (tone === '쿨톤') h.tone = 'cool';
      else if (tone === '웜톤') h.tone = 'warm';
      else if (tone === '뉴트럴') h.tone = 'neutral';
      const col = get('color');
      if (col) h.colorLabel = col;
      out.hair = h;
    } else if (cat === 'nail') {
      const n: NailHints = {};
      const design = get('design');
      const finish = get('finish');
      const map: Record<string, NailStyle> = {
        원컬러: 'solid',
        프렌치: 'french',
        그라데이션: 'gradient',
        글리터: 'glitter',
        자석: 'cateye',
        오로라: 'aurora',
        크롬: 'chrome',
        글레이즈드: 'chrome',
        시럽: 'jelly',
        치크: 'gradient',
        도트: 'dots',
      };
      if (design && map[design]) n.style = map[design];
      else if (finish === '메탈릭') n.style = 'chrome';
      else if (finish === '투명·쉬어') n.style = 'jelly';
      else if (finish === '펄·쉬머') n.style = 'aurora';
      const len = get('length');
      if (len === '숏네일') n.length = 0;
      else if (len === '미디엄') n.length = 0.35;
      else if (len === '롱네일') n.length = 0.8;
      const shape = get('shape');
      if (shape) n.shape = shape;
      const col = get('color');
      if (col) n.colorLabel = col;
      out.nail = n;
    } else if (cat === 'tattoo') {
      const t: TattooHints = {};
      const place = get('placement');
      const pm: Record<string, TattooPlace> = {
        손목: 'forearmL',
        손가락: 'forearmL',
        팔안쪽: 'forearmL',
        팔뚝: 'upperArmL',
        어깨: 'upperArmL',
        쇄골: 'chest',
        '목·귀뒤': 'neckL',
      };
      if (place && pm[place]) t.place = pm[place];
      const size = get('size');
      if (size === '미니') t.size = 0.15;
      else if (size === '스몰') t.size = 0.3;
      else if (size === '미디엄') t.size = 0.55;
      else if (size === '대형') t.size = 0.9;
      const col = get('color');
      if (col === '블랙') t.color = 'black';
      else if (col === '블랙앤그레이') t.color = 'grey';
      else if (col === '풀컬러' || col === '포인트컬러') t.color = 'color';
      const st = get('style');
      if (st) t.styleLabel = st;
      out.tattoo = t;
    }
  };
  consider(r.attributes, r.category);
  if (r.secondary) consider(r.secondary.attributes, r.secondary.category);
  return out;
}

/** 생성 모드 글 설명에 쓸 속성 그룹(분야별). 무드·톤처럼 그림으로 표현하기 어려운 것은 뺀다 */
const PROMPT_GROUPS: Record<string, string[]> = {
  hair: ['length', 'cut', 'styling', 'perm', 'bangs', 'color', 'colorTech'],
  nail: ['shape', 'length', 'color', 'design', 'finish'],
  tattoo: ['style', 'color', 'subject', 'size'],
};

/**
 * 분류기 속성 → 영어 설명 한 줄(생성 서버의 prompt). 확신 minScore 이상인 속성의 영어 설명을 이어 붙인다.
 * 분류기 결과의 분야가 다르면 빈 문자열.
 */
export function promptFromStyleAI(r: StyleAIResult | undefined, category: 'hair' | 'nail' | 'tattoo', minScore = 0.4): string {
  if (!r) return '';
  const pickFrom = (attrs: StyleAttr[]): string[] =>
    (PROMPT_GROUPS[category] ?? [])
      .map((g) => attrs.find((a) => a.group === g))
      .filter((a): a is StyleAttr => !!a && a.score >= minScore && !!a.label_en)
      .map((a) => a.label_en!);
  let parts: string[] = [];
  if (r.category === category) parts = pickFrom(r.attributes);
  else if (r.secondary && r.secondary.category === category) parts = pickFrom(r.secondary.attributes);
  return parts.join(', ');
}
