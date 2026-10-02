// 사진에서 잰 색 → 앱 설정으로 바꾸는 순수 계산(단위 테스트 대상).
// 메이크업 합성 셰이더(makeup.ts)의 색 계산식을 거꾸로 푼다:
//  - 립(덮는 색): 사진 입술색 = 목표색 × 조명 → 목표색 = 사진 입술색 / 조명(사진 피부로 추정)
//  - 블러셔·아이섀도(비치는 색): 사진 볼색/피부색 비율 → 진하기를 고정하고 그 비율이 나오는 색을 구한다
// 색은 모두 선형 RGB(0~1). sRGB ↔ 선형 변환은 lin/gam.

import type { RGB } from './makeup.ts';

/** 밝은 조명의 중간 밝기 기준 피부(sRGB #d1a38a)의 선형 값 — 셰이더의 REF와 같아야 한다 */
export const REF: RGB = [0.637, 0.366, 0.254];
const WL: RGB = [0.2126, 0.7152, 0.0722];

export const lin = (c: RGB): RGB => c.map((v) => Math.pow(Math.max(0, v), 2.2)) as RGB;
export const gam = (c: RGB): RGB => c.map((v) => Math.pow(Math.max(0, Math.min(1, v)), 1 / 2.2)) as RGB;
export const luma = (c: RGB): number => c[0] * WL[0] + c[1] * WL[1] + c[2] * WL[2];
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
export const hex = (c: RGB): string => '#' + gam(c).map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('');

/** 셰이더와 같은 조명 추정: 피부 평균(선형) → 채널별 조명 배율 */
export function illumination(skin: RGB): RGB {
  const Ls = Math.max(luma(skin), 1e-3);
  const LR = luma(REF);
  const bright = Math.pow(clamp(Ls / LR, 0.1, 2.5), 0.6);
  return skin.map((v, i) => {
    const wb = clamp(1 + 0.7 * ((v / Ls) / (REF[i] / LR) - 1), 0.7, 1.35);
    return wb * bright;
  }) as RGB;
}

/** 사진 입술색(선형)과 사진 피부(선형) → 립 목표색(sRGB) */
export function lipTarget(lip: RGB, skin: RGB): RGB {
  const il = illumination(skin);
  return gam(lip.map((v, i) => clamp(v / il[i], 0, 1)) as RGB);
}

/**
 * 비치는 색 부위(블러셔·아이섀도·눈썹): 사진에서 잰 (부위색 / 피부색) 비율과, 화장 안 한 얼굴의 자연 비율을 비교해
 * 목표색과 진하기를 구한다. amount를 고정하고, 그 진하기에서 실제로 나타나는 정도(keff)를 거꾸로 푼다.
 * @param ratio 사진의 부위색/피부색(선형, 채널별)
 * @param natural 화장 안 한 얼굴의 같은 비율(시험 얼굴로 잰 값)
 * @param keff 진하기 amount일 때 부위 가운데에서 실제로 섞이는 비율(측정값)
 * @returns 색(sRGB)과 진하기. 사진이 자연 비율과 거의 같으면 null(그 부위 화장 없음)
 */
export function tintTarget(ratio: RGB, natural: RGB, amount: number, keff: number, minDiff = 0.05): { color: RGB; amount: number } | null {
  // 화장이 더한 배율 t: 사진 비율 / 자연 비율
  const t = ratio.map((v, i) => clamp(v / natural[i], 0.3, 1.6)) as RGB;
  const diff = Math.max(...t.map((v) => Math.abs(v - 1)));
  if (diff < minDiff) return null;
  // 셰이더: 결과 = 피부 × (1 + keff·(색/REF − 1)) → 색/REF = 1 + (t − 1)/keff
  const k = Math.max(0.05, keff);
  const color = REF.map((r, i) => clamp(r * (1 + (t[i] - 1) / k), 0.02, 1)) as RGB;
  return { color: gam(color), amount };
}

/** 두 색(선형)의 차이(0~1 정도) */
export function colorDiff(a: RGB, b: RGB): number {
  const ga = gam(a);
  const gb = gam(b);
  return Math.hypot(ga[0] - gb[0], ga[1] - gb[1], ga[2] - gb[2]) / Math.sqrt(3);
}

/** 립 광택: 입술 밝기 분포(90% 값 / 중간값)가 클수록 반사광이 많다 */
export function glossFrom(p50: number, p90: number): number {
  return clamp((p90 / Math.max(p50, 1e-3) - 1.15) * 1.6, 0.05, 0.8);
}

/**
 * 헤어: 위(뿌리)·아래(끝) 평균색(선형)이 충분히 다르면 옴브레로 본다.
 * @returns color: 전체 색(sRGB), tip: 끝 색(sRGB) 또는 null
 */
export function hairTarget(all: RGB, top: RGB, bottom: RGB): { color: RGB; tip: RGB | null } {
  const ombre = colorDiff(top, bottom) > 0.09;
  return ombre ? { color: gam(top), tip: gam(bottom) } : { color: gam(all), tip: null };
}

/** k-평균으로 주요 색을 찾는다(작은 표본용). 반환: 비율 내림차순 [색(선형), 비율] */
export function dominantColors(px: RGB[], k = 4, iters = 12): { color: RGB; share: number }[] {
  if (px.length === 0) return [];
  let centers: RGB[] = Array.from({ length: Math.min(k, px.length) }, (_, j) => px[Math.floor((j + 0.5) * (px.length / k))]);
  let labels = new Int32Array(px.length);
  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < px.length; i++) {
      let best = 0;
      let bd = Infinity;
      for (let j = 0; j < centers.length; j++) {
        const c = centers[j];
        const d = (px[i][0] - c[0]) ** 2 + (px[i][1] - c[1]) ** 2 + (px[i][2] - c[2]) ** 2;
        if (d < bd) {
          bd = d;
          best = j;
        }
      }
      labels[i] = best;
    }
    const sums = centers.map(() => [0, 0, 0, 0]);
    for (let i = 0; i < px.length; i++) {
      const s = sums[labels[i]];
      s[0] += px[i][0];
      s[1] += px[i][1];
      s[2] += px[i][2];
      s[3]++;
    }
    centers = centers.map((c, j) => (sums[j][3] ? ([sums[j][0], sums[j][1], sums[j][2]].map((v) => v / sums[j][3]) as RGB) : c));
  }
  const counts = centers.map((_, j) => labels.reduce((n, l) => n + (l === j ? 1 : 0), 0));
  return centers
    .map((color, j) => ({ color, share: counts[j] / px.length }))
    .sort((a, b) => b.share - a.share);
}

/** 피부색 판정(선형 RGB): 주황~분홍 계열의 중간 채도 */
export function isSkin(c: RGB): boolean {
  const g = gam(c);
  const mx = Math.max(g[0], g[1], g[2]);
  const mn = Math.min(g[0], g[1], g[2]);
  if (mx < 0.3) return false;
  const s = (mx - mn) / Math.max(mx, 1e-3);
  if (s < 0.12 || s > 0.62) return false;
  // 빨강이 가장 크고 파랑이 가장 작은 색상(주황~분홍)
  return g[0] >= g[1] && g[1] >= g[2] * 0.9;
}
