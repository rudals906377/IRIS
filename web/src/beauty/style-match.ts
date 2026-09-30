// 사진 따라하기의 되먹임(순수 계산, 단위 테스트 대상).
// 사진에서 잰 비율(참고)과, 화장을 입힌 내 얼굴에서 잰 비율(결과), 화장 전 내 얼굴 비율(자연)을 비교해
// 색과 진하기를 고쳐 결과가 사진에 가까워지게 한다. 한 번에 다 맞추지 않고 두세 번 반복한다.

import type { RGB } from './makeup.ts';
import { REF, gam, lin } from './style-math.ts';

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

export interface TintSetting {
  color: RGB;
  amount: number;
}

/**
 * 비치는 색 부위(블러셔·아이섀도) 맞추기.
 * 사진이 더한 배율 want = 사진 비율 / 자연 비율(일반), 내 결과가 더한 배율 got = 결과 비율 / 내 자연 비율.
 * 셰이더는 결과 ≈ 피부 × (1 + k·(색/REF − 1)) 이므로, 채널마다 (색/REF − 1)을 want/got 만큼 늘린다.
 * 진하기는 초록 채널(붉은 기가 가장 크게 나타나는 채널)로 먼저 맞추고, 넘치면 색으로 보충한다.
 */
export function matchTint(cur: TintSetting, want: RGB, got: RGB, natural: RGB, myNatural: RGB): TintSetting {
  // 밝기 차이(조명·그늘)는 무시하고 색조만 비교
  const nz = (r: RGB): RGB => {
    const l = Math.max(0.2126 * r[0] + 0.7152 * r[1] + 0.0722 * r[2], 1e-4);
    return r.map((v) => v / l) as RGB;
  };
  const [wantC, gotC, natC, myNatC] = [want, got, natural, myNatural].map(nz);
  const w = wantC.map((v, i) => clamp(v / natC[i], 0.3, 1.6)) as RGB;
  const g = gotC.map((v, i) => clamp(v / myNatC[i], 0.3, 1.6)) as RGB;
  // 기준 채널: 사진에서 자연과 가장 많이 다른 채널(붉은 블러셔는 초록, 푸른 섀도는 빨강)
  let d = 1;
  for (let i = 0; i < 3; i++) if (Math.abs(1 - w[i]) > Math.abs(1 - w[d])) d = i;
  const wantDev = 1 - w[d];
  const gotDev = 1 - g[d];
  // 사진에 그 부위 화장이 거의 없으면 끈다(진하기 0 → 호출 쪽에서 색을 지운다)
  if (Math.abs(wantDev) < 0.04) return { ...cur, amount: 0 };
  // 진하기는 넓고 부드럽게 보이도록 높게 두고(가루가 넓게 퍼진 느낌), 색의 편차로 정도를 맞춘다
  const amount = Math.max(cur.amount, 0.85);
  const cl = lin(cur.color);
  // 아직 거의 안 칠해졌으면(측정 잡음 수준) 편차를 크게 키운다
  // 한 번에 너무 크게 바꾸지 않는다(측정 잡음으로 튀는 것을 막고, 반복하며 수렴)
  // 내 결과가 반대 방향이거나 거의 없으면 키우고, 같은 방향이면 비율로
  const boost = Math.abs(gotDev) < 0.015 || Math.sign(gotDev) !== Math.sign(wantDev) ? 1.5 : clamp(wantDev / gotDev, 0.6, 1.6);
  // 색 편차 상한: 블러셔·섀도가 형광처럼 튀지 않게(초록·파랑은 기준의 40% 이상, 빨강은 120% 이하)
  const lo: RGB = [0.7, 0.4, 0.4];
  const hi: RGB = [1.2, 1.3, 1.3];
  const color = cl.map((v, i) => {
    const chanW = 1 - w[i];
    const chanG = 1 - g[i];
    // 채널별 비율(초록 기준 보정에 더해, 채널마다 남은 차이를 맞춘다)
    const rel = Math.abs(chanG) > 0.01 && Math.abs(gotDev) > 0.015 && Math.sign(gotDev) === Math.sign(wantDev) ? (chanW / wantDev) / (chanG / gotDev) : 1;
    const dev = (v / REF[i] - 1) * boost * clamp(rel, 0.6, 1.6);
    return clamp(REF[i] * clamp(1 + dev, lo[i], hi[i]), 0.02, 1);
  }) as RGB;
  return { color: gam(color), amount };
}

/**
 * 덮는 색(립) 맞추기: 결과 입술색(조명 나눈 값)이 사진 입술색(조명 나눈 값)과 같아지도록 색을 채널별 비율로 고친다.
 * 진하기 미만으로 섞이는 만큼(1 − amount)은 원래 입술이 비치므로, 비율 보정을 조금 크게 준다.
 */
export function matchLip(cur: TintSetting, wantLin: RGB, gotLin: RGB): TintSetting {
  const cl = lin(cur.color);
  const color = cl.map((v, i) => {
    const r = clamp(wantLin[i] / Math.max(gotLin[i], 1e-3), 0.5, 2);
    return clamp(v * Math.pow(r, 1.2), 0.005, 1);
  }) as RGB;
  return { color: gam(color), amount: cur.amount };
}

/**
 * 어두운 색으로 덮는 부위(눈썹·아이라인): 밝기만 맞춘다.
 * 사진 부위의 밝기(예상 피부 대비)를 그대로 목표로 삼는다(화장은 어둡게만 할 수 있으므로,
 * 내 원래 부위가 이미 사진보다 어두우면 거의 끈다). naturalL은 참고용으로만 받는다.
 */
export function matchDarkness(cur: TintSetting, wantL: number, gotL: number, naturalL: number, myNaturalL: number): TintSetting {
  void naturalL;
  const targetL = clamp(wantL, 0.02, 1.5);
  if (targetL >= myNaturalL * 0.97) return { ...cur, amount: 0.1 };
  const want = 1 - targetL / myNaturalL;
  const got = 1 - clamp(gotL / myNaturalL, 0.2, 1.2);
  if (got < 0.02) return { ...cur, amount: clamp(cur.amount * 1.5, 0.1, 1) };
  return { ...cur, amount: clamp(cur.amount * (want / got), 0.1, 1) };
}
