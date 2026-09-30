// 상의 착용 리그: 상품 이미지 좌표 → 화면 좌표 변형.
//
// 몸판: 몸통을 타원 기둥으로 보고, 상품 사진의 앞판(옆선~옆선)을 기둥의 앞쪽 반을 감싸게 입힌다.
//       - 매 줄의 옆선은 착용자의 실제 몸통 윤곽(분할에서 측정)에 맞춘다 → 몸에 달라붙는다.
//       - 몸을 돌리면 기둥이 같이 돌아 먼 쪽이 압축되고 옆·뒷면이 드러난다(뒷면은 원단 바탕색).
//       - 기둥의 곡면 방향으로 가장자리를 어둡게 해 입체감을 준다.
//       세로 배율은 몸통 길이에 맞추되 상품 고유의 기장 비율은 살린다.
// 소매: 상품 사진의 소매를 네 꼭짓점(어깨점·겨드랑이·끝단 바깥·끝단 안쪽)으로 된 곡면으로 보고,
//       어깨점·겨드랑이는 몸판 진동 둘레에 그대로 붙이고(틈 없음), 윗선은 팔 바깥쪽,
//       아랫선은 팔 안쪽을 따라가게 한다. 긴팔은 팔꿈치에서 굽는다.

import { LM, type BodyFrame } from './body.ts';
import { PART, type GarmentAsset, type PartMesh } from './garment.ts';
import { add, dist, dot, lerp, mid, norm, perp, scale, smoothstep, sub, type Vec2 } from './math.ts';
import type { TorsoTracker } from './silhouette.ts';

export interface TopFit {
  /** 어깨 봉제선 폭 ÷ 어깨 랜드마크 폭. 봉제선은 관절보다 조금 바깥에 놓인다. */
  seamWidth: number;
  /** 옷 전체를 어깨선 위로 올리는 양(어깨 폭 비율). 관절 중심보다 어깨 표면이 높기 때문. */
  lift: number;
  /** 기장 배율(1 = 상품 비율 그대로). */
  length: number;
}

export const DEFAULT_TOP_FIT: TopFit = { seamWidth: 1.08, lift: 0.06, length: 1 };

/**
 * 기장 기준: 보통 티셔츠(몸길이 ÷ 어깨 봉제선 폭 ≈ 1.55)는 밑단이 어깨→엉덩이 관절 거리의 약 1.12배에 온다.
 * 상품마다 이 비율이 다르면(크롭·롱) 그만큼 밑단 위치를 옮겨 상품의 기장 특성을 살린다.
 */
const STD_LENGTH_RATIO = 1.55;
const STD_HEM_T = 1.12;
/** 몸통 단면 타원의 깊이 ÷ 폭(가슴 두께 대략 가로 폭의 0.65배) */
const TORSO_DEPTH = 0.65;

/** 몸통 반폭 기본값(측정이 없을 때, 어깨 관절 폭 비율) */
function defaultHalf(t: number): number {
  return t < 0.5 ? 0.55 - 0.16 * t : t < 0.85 ? 0.47 : 0.47 + 0.12 * Math.min(1, (t - 0.85) / 0.4);
}

interface SleeveRig {
  mesh: PartMesh;
  lmShoulder: number;
  lmElbow: number;
  lmWrist: number;
  /** 착용자 왼쪽이면 +1(몸 좌표계 u 방향), 오른쪽이면 -1. */
  side: 1 | -1;
  /** 정점별 관(튜브) 좌표: s 뿌리(0) → 끝단(1), v 폭 안의 위치(-1 안쪽 ~ +1 바깥쪽) */
  s: Float32Array;
  v: Float32Array;
  /** 소매 중심선 길이(상품 픽셀) */
  length: number;
  /** 바깥선(어깨점→끝단 바깥)·안쪽선(겨드랑이→끝단 안쪽) 길이(상품 픽셀) */
  topLen: number;
  bottomLen: number;
  /** s 구간별 반폭(상품 픽셀), TUBE_BINS개 */
  halfW: Float32Array;
}

/** 소매 관 좌표의 길이 방향 구간 수(0 ~ 1.2) */
const TUBE_BINS = 12;

export class TopRig {
  readonly asset: GarmentAsset;
  /** 그리는 순서(뒤 → 앞). update 후 갱신된다. */
  order: PartMesh[] = [];
  private readonly torso: PartMesh | undefined;
  /** 몸판 뒷면(앞면 메쉬를 복제, 원단 바탕색으로 그림). 렌더러에 추가 메쉬로 넘긴다. */
  readonly backTorso: PartMesh | undefined;
  /** 상품 사진 몸판의 줄별 좌우 끝(4px 간격) */
  private readonly rowL: Float32Array;
  private readonly rowR: Float32Array;
  private readonly chestHalf: number;
  /** 어깨선 → 겨드랑이 세로 길이(상품 픽셀) */
  private readonly armpitDy: number;
  private readonly neckInner: PartMesh | undefined;
  private readonly sleeves: SleeveRig[] = [];
  private readonly gMid: Vec2;
  private readonly gw: number;
  /** 어깨선 → 밑단 세로 길이(상품 픽셀) */
  private readonly gLen: number;

  constructor(asset: GarmentAsset) {
    this.asset = asset;
    const kp = asset.kp;
    this.gMid = mid(kp.shoulderL, kp.shoulderR);
    this.gw = dist(kp.shoulderL, kp.shoulderR) / (asset.info.widthScale ?? 1);
    const hemY = kp.hemL && kp.hemR ? (kp.hemL.y + kp.hemR.y) / 2 : this.gMid.y + this.gw * STD_LENGTH_RATIO;
    this.gLen = Math.max(1, hemY - this.gMid.y);
    const armpitY = kp.armpitL && kp.armpitR ? (kp.armpitL.y + kp.armpitR.y) / 2 : this.gMid.y + this.gLen * 0.28;
    this.armpitDy = Math.max(1, armpitY - this.gMid.y);
    this.torso = asset.meshes.find((m) => m.partId === PART.torso);
    if (this.torso) {
      this.backTorso = { ...this.torso, dst: new Float32Array(this.torso.dst.length), aux: new Float32Array(this.torso.vertexCount * 2), back: true };
    }
    // 몸판 줄별 좌우 끝(몸판·목 안쪽 라벨)
    {
      const { labels, width: w, height: h } = asset;
      const rows = Math.ceil(h / 4);
      const L = new Float32Array(rows).fill(NaN);
      const R = new Float32Array(rows).fill(NaN);
      for (let r = 0; r < rows; r++) {
        const y = r * 4;
        let a = -1;
        let b = -1;
        for (let x = 0; x < w; x++) {
          const l = labels[y * w + x];
          if (l !== PART.torso && l !== PART.neckInner) continue;
          if (a < 0) a = x;
          b = x;
        }
        if (a >= 0 && b - a > 4) {
          L[r] = a;
          R[r] = b;
        }
      }
      // 빈 줄은 가까운 줄 값으로, 겨드랑이 위는 겨드랑이 줄 폭 이상으로(어깨 경사로 좁아지지 않게)
      const pitRow = Math.round(armpitY / 4);
      const fill = (arr: Float32Array): void => {
        let last = NaN;
        for (let r = 0; r < rows; r++) if (Number.isFinite(arr[r])) last = arr[r]; else arr[r] = last;
        last = NaN;
        for (let r = rows - 1; r >= 0; r--) if (Number.isFinite(arr[r])) last = arr[r]; else arr[r] = last;
      };
      fill(L);
      fill(R);
      const pr = Math.max(0, Math.min(rows - 1, pitRow));
      const pl = kp.armpitR ? kp.armpitR.x : L[pr];
      const prr = kp.armpitL ? kp.armpitL.x : R[pr];
      // 겨드랑이 위는 몸판 라벨이 소매 쪽으로 파여 있으므로 겨드랑이 폭으로 고정
      for (let r = 0; r <= pr; r++) {
        L[r] = pl;
        R[r] = prr;
      }
      this.rowL = L;
      this.rowR = R;
      this.chestHalf = Math.max(4, (prr - pl) / 2);
    }
    this.neckInner = asset.meshes.find((m) => m.partId === PART.neckInner);
    const defs = [
      { partId: PART.sleeveL, sh: 'shoulderL', ap: 'armpitL', out: 'sleeveOuterL', inn: 'sleeveInnerL', side: 1 as const, lms: LM.leftShoulder, lme: LM.leftElbow, lmw: LM.leftWrist },
      { partId: PART.sleeveR, sh: 'shoulderR', ap: 'armpitR', out: 'sleeveOuterR', inn: 'sleeveInnerR', side: -1 as const, lms: LM.rightShoulder, lme: LM.rightElbow, lmw: LM.rightWrist },
    ];
    for (const d of defs) {
      const mesh = asset.meshes.find((m) => m.partId === d.partId);
      if (!mesh || !kp[d.sh] || !kp[d.ap] || !kp[d.out] || !kp[d.inn]) continue;
      const tube = buildTube(asset, d.partId, kp[d.sh], kp[d.ap], kp[d.out], kp[d.inn], mesh);
      this.sleeves.push({
        mesh, lmShoulder: d.lms, lmElbow: d.lme, lmWrist: d.lmw, side: d.side, ...tube,
        topLen: dist(kp[d.sh], kp[d.out]), bottomLen: dist(kp[d.ap], kp[d.inn]),
      });
    }
  }

  update(body: BodyFrame, fit: TopFit = DEFAULT_TOP_FIT, torsoFit?: TorsoTracker): void {
    // 크기 기준: 상품 가슴 폭(겨드랑이~겨드랑이). 보통 옷은 어깨점 폭 ≈ 가슴 폭 × 0.95이므로 기존 어깨 기준과 같은 크기가 된다.
    // 몸을 돌리면 화면상 어깨 폭이 줄므로, 회전 각도로 되돌린 정면 어깨 폭으로 크기를 정한다
    const frontW = body.shoulderW / Math.max(0.6, Math.abs(Math.cos(body.turn)));
    const k = (frontW * fit.seamWidth) / Math.max(this.gw * 0.6, this.chestHalf * 1.9);
    // 세로 배율: 몸통 길이에 맞춰 밑단 위치를 정하되 상품의 기장 비율을 반영한다.
    const hemT = STD_HEM_T * (this.gLen / this.gw / STD_LENGTH_RATIO) * fit.length;
    const kv = (hemT * body.axisLen) / this.gLen;
    const lift = fit.lift * body.shoulderW;
    const sm = body.shoulderMid;
    const axisDir = norm(sub(body.hipMid, sm), body.down);
    const { u, down, hipU } = body;
    const gMid = this.gMid;
    const psi = body.turn;
    const cosP = Math.cos(psi);
    const sinP = Math.sin(psi);
    // 타원 기둥(가로 반지름 1, 깊이 반지름 B)을 psi만큼 돌렸을 때 화면에 보이는 반폭
    const B = TORSO_DEPTH;
    const E = Math.sqrt(cosP * cosP + B * B * sinP * sinP);
    // 가로 여유: 어깨 폭 맞춤 설정(기본 1.08)을 1로 보고 비례
    const easeFit = fit.seamWidth / DEFAULT_TOP_FIT.seamWidth;
    const rows = this.rowL.length;
    const armpitYr = Math.max(1, this.armpitDy * kv - lift);
    // 어깨점(소매 뿌리 바깥 끝): 착용자 팔 위. 상품의 어깨 봉제선이 가슴 폭보다 밖으로 나간 만큼(드롭숄더) 팔을 따라 내려간다.
    const arms = new Map<SleeveRig, { arm: ArmPath; sign: number; cap: Vec2; off0: number; drop: number }>();
    const capLat: Record<1 | -1, number> = { 1: NaN, [-1]: NaN };
    /** 어깨점이 어깨선(어깨 중점 높이)보다 몸통 축 방향으로 얼마나 아래인지(어깨 경사 + 드롭숄더) */
    const capAx: Record<1 | -1, number> = { 1: NaN, [-1]: NaN };
    // 겨드랑이 높이의 몸판 반폭: 측정한 옷 가장자리와 팔 안쪽 가장자리 중 좁은 쪽
    // (팔을 내리면 측정값에 지금 입은 옷의 소매까지 들어오므로 팔 안쪽에서 끊는다)
    const measPit = torsoFit ? torsoFit.at(armpitYr / body.axisLen, body.shoulderW) : { left: defaultHalf(armpitYr / body.axisLen) * body.shoulderW, right: defaultHalf(armpitYr / body.axisLen) * body.shoulderW };
    const pitLat: Record<1 | -1, number> = { 1: measPit.left, [-1]: measPit.right };
    for (const sl of this.sleeves) {
      const arm = new ArmPath(body, sl.lmShoulder, sl.lmElbow, sl.lmWrist, sl.side);
      const sign = dot(arm.normal(0), sub(arm.at(0), sm)) >= 0 ? 1 : -1;
      const w = body.shoulderW;
      let off0 = sl.halfW[0] * k;
      const am0 = torsoFit ? torsoFit.armAt(sl.side, 0.05, w) : NaN;
      off0 = Number.isFinite(am0) ? Math.min(Math.max(off0, am0 * 1.15), am0 * 2.2) : Math.max(off0, w * 0.15);
      const drop = Math.max(0, (this.gw / 2 - this.chestHalf * 0.95) * k);
      // 어깨 관절점은 어깨 윗면보다 아래에 있으므로, 드롭숄더가 아닐수록 어깨점을 위로 올린다
      // 어깨점은 관절 높이 근처(어깨선은 목에서 여기로 기울어 내려온다), 바깥쪽은 소매 뿌리 폭에 맞춘다
      const raise = Math.max(0, w * 0.03 - drop * 0.5) + lift * 0.5;
      const cap = add(add(arm.at(drop), scale(arm.normal(drop), sign * off0 * 0.85)), scale(axisDir, -raise));
      arms.set(sl, { arm, sign, cap, off0, drop });
      capLat[sl.side] = Math.max(0, dot(sub(cap, sm), u) * sl.side);
      capAx[sl.side] = dot(sub(cap, sm), axisDir);
      const inner = add(arm.at(armpitYr), scale(arm.normal(armpitYr), -sign * off0));
      const innerLat = dot(sub(inner, sm), u) * sl.side;
      if (innerLat > body.shoulderW * 0.2) pitLat[sl.side] = Math.min(pitLat[sl.side], innerLat);
    }
    if (!Number.isFinite(capLat[1])) capLat[1] = body.shoulderW * 0.62;
    if (!Number.isFinite(capLat[-1])) capLat[-1] = body.shoulderW * 0.62;
    if (!Number.isFinite(capAx[1])) capAx[1] = body.shoulderW * 0.1;
    if (!Number.isFinite(capAx[-1])) capAx[-1] = body.shoulderW * 0.1;

    const mapInto = (gx: number, gy: number, out: Float32Array, o: number, back = false, aux?: Float32Array): void => {
      // 어깨선 위(목둘레)는 가로와 같은 배율, 아래(몸통)는 몸통 길이 배율
      const dy = gy - gMid.y;
      let yr = (dy < 0 ? dy * k : dy * kv) - lift;
      const r = Math.max(0, Math.min(rows - 1, Math.round(gy / 4)));
      const gl = this.rowL[r];
      const gr = this.rowR[r];
      const halfL = Math.max(4, gr - gMid.x);
      const halfR = Math.max(4, gMid.x - gl);
      // 옆선(-1~1) 기준 가로 위치. 옆선 바깥(소매 쪽으로 번진 몸판)은 조금 더 감긴다.
      // 겨드랑이 위(목둘레·어깨 요크·후드 모자)는 가슴 폭 기준 비율로 재고, 옆선 밖(어깨 봉제선 쪽)은
      // 감지 않고 평평하게 늘린다(어깨점이 몸통 안으로 말려 들어가 소매 뿌리가 뒤집히지 않게).
      const above = dy < this.armpitDy;
      // 겨드랑이 위: 상품 반폭을 어깨 봉제선 반폭(gw/2) → 가슴 반폭으로 보간해 비율을 잰다
      const fracUp = above ? Math.max(0, Math.min(1, (dy + lift / k) / this.armpitDy)) : 1;
      const halfAbove = this.gw / 2 + (this.chestHalf - this.gw / 2) * fracUp;
      let q = above ? (gx - gMid.x) / halfAbove : gx >= gMid.x ? (gx - gMid.x) / halfL : (gx - gMid.x) / halfR;
      // 상품 고유 실루엣(밑단이 퍼진 옷 등)을 일부 살린다
      const rowHalf = above ? this.chestHalf : (halfL + halfR) / 2;
      const ease = easeFit * Math.max(0.9, Math.min(1.35, 1 + 0.35 * (rowHalf / this.chestHalf - 1)));
      // 어깨 경사: 겨드랑이 위 구간은 옆으로 갈수록(|q|→1) 어깨점 높이까지 내려간다(착용자 어깨 기울기 + 드롭숄더)
      if (above) {
        const side = q >= 0 ? 1 : -1;
        const shoulderRowY = -lift;
        yr += Math.min(1, Math.abs(q)) * (capAx[side] - shoulderRowY) * (1 - fracUp);
      }
      let X: number;
      let Z: number;
      let th: number;
      if (above && Math.abs(q) > 1) {
        // 어깨 봉제선 밖(소매 뿌리 쪽으로 번진 몸판)은 조금만 넘어가게 눌러 뾰족한 날개가 생기지 않게 한다
        q = Math.sign(q) * (1 + Math.min(0.15, (Math.abs(q) - 1) * 0.3));
        X = back ? -q : q;
        Z = (back ? -0.3 : 0.3) * B; // 앞면은 확실히 보이고 뒷면은 숨긴다
        th = back ? Math.PI - Math.sign(q) * (Math.PI / 2) : Math.sign(q) * (Math.PI / 2);
      } else {
        q = Math.max(-1.3, Math.min(1.3, q));
        th = (q * Math.PI) / 2;
        if (back) th = Math.PI - th;
        X = Math.sin(th);
        Z = B * Math.cos(th);
      }
      const Xr = X * cosP + Z * sinP;
      const Zr = -X * sinP + Z * cosP;
      const sN = Xr / E;
      const t = Math.max(0, yr) / body.axisLen;
      const meas = torsoFit ? torsoFit.at(t, body.shoulderW) : { left: defaultHalf(t) * body.shoulderW, right: defaultHalf(t) * body.shoulderW };
      // 겨드랑이 위(어깨 경사·진동 둘레)는 상품 비율 그대로(가슴 폭 × 배율), 아래로 갈수록 몸 윤곽에 맞춘다.
      // 어깨 높이에서 잰 윤곽에는 어깨 근육·소매가 포함돼 몸판을 거기 맞추면 어깨가 네모나게 부푼다.
      // 어깨점 가로 위치 → 겨드랑이 반폭 → 그 아래는 측정 폭(단, 겨드랑이에서 갑자기 넓어지지 않게)
      const wFit = smoothstep(0, armpitYr, yr);
      const below = Math.max(0, yr - armpitYr);
      const widen = (m: number, p: number): number => Math.min(m, p + below * 0.25);
      const wdt = {
        left: yr <= armpitYr ? capLat[1] + (pitLat[1] - capLat[1]) * wFit : widen(meas.left, pitLat[1]),
        right: yr <= armpitYr ? capLat[-1] + (pitLat[-1] - capLat[-1]) * wFit : widen(meas.right, pitLat[-1]),
      };
      const xr = (sN >= 0 ? sN * wdt.left : sN * wdt.right) * ease;
      if (aux) {
        // 곡면 법선의 카메라 쪽 성분으로 음영, 뒤로 돌아간 면은 숨김
        const nx = Math.sin(th);
        const nz = Math.cos(th) / B;
        const nl = Math.hypot(nx, nz) || 1;
        const nzr = (-nx * sinP + nz * cosP) / nl;
        aux[o] = 0.7 + 0.3 * Math.max(0, nzr);
        aux[o + 1] = smoothstep(-0.02, 0.1, Zr / B);
      }
      let ox: number;
      let oy: number;
      let lx = u.x;
      let ly = u.y;
      if (yr <= 0) {
        ox = sm.x + down.x * yr;
        oy = sm.y + down.y * yr;
      } else {
        ox = sm.x + axisDir.x * yr;
        oy = sm.y + axisDir.y * yr;
        const w = Math.min(1, yr / body.axisLen);
        const lxx = u.x + (hipU.x - u.x) * w;
        const lyy = u.y + (hipU.y - u.y) * w;
        const l = Math.hypot(lxx, lyy) || 1;
        lx = lxx / l;
        ly = lyy / l;
      }
      out[o] = ox + lx * xr;
      out[o + 1] = oy + ly * xr;
    };

    for (const m of [this.torso, this.neckInner]) {
      if (!m) continue;
      m.aux ??= new Float32Array(m.vertexCount * 2);
      for (let i = 0; i < m.vertexCount; i++) mapInto(m.src[i * 2], m.src[i * 2 + 1], m.dst, i * 2, false, m.aux);
    }
    const bt = this.backTorso;
    if (bt) {
      for (let i = 0; i < bt.vertexCount; i++) mapInto(bt.src[i * 2], bt.src[i * 2 + 1], bt.dst, i * 2, true, bt.aux);
    }

    const sleeveOrder: { mesh: PartMesh; z: number }[] = [];
    const tmpP = new Float32Array(2);
    const mapPoint = (p: Vec2): Vec2 => {
      mapInto(p.x, p.y, tmpP, 0);
      return { x: tmpP[0], y: tmpP[1] };
    };
    for (const sl of this.sleeves) {
      const { arm, sign, cap, off0, drop } = arms.get(sl)!;
      // 겨드랑이: 몸판 옆선 위(몸판과 같은 변형이라 이음새 틈이 없다)
      const pit = mapPoint(this.asset.kp[sl.side === 1 ? 'armpitL' : 'armpitR']);
      const w = body.shoulderW;
      const armR = (a: number): number => w * (0.15 - 0.045 * Math.min(1, Math.max(0, a / arm.length)));
      // 바깥선은 어깨점에서, 안쪽선은 겨드랑이에서 시작해 각자 길이만큼 팔을 따라 내려간다.
      // (상품 사진에서 소매가 옆으로 뻗어 있어도 겨드랑이 쪽이 접히며 무늬가 팔 방향으로 통째로 돌지 않는다)
      // 소매 길이는 팔(어깨→손목) 길이를 넘지 않는다(손을 덮지 않게)
      const lenScale = Math.min(1, (arm.length * 1.03 - drop) / Math.max(1, sl.topLen * k));
      const topLen = sl.topLen * k * lenScale;
      const pitArc = Math.max(0, dot(sub(pit, arm.at(0)), arm.dir(0)));
      // 안쪽선 길이: 상품 값과 "밑단이 팔에 직각이 되는 길이"를 섞는다(팔을 내리면 겨드랑이 쪽이 접히며 밑단이 팔에 직각에 가깝다)
      const perpLen = Math.max(topLen * 0.3, drop + topLen - pitArc);
      const botLen = sl.bottomLen * k * lenScale * 0.4 + perpLen * 0.6;
      const halfAt = (s: number, a: number): number => {
        const x = Math.max(0, Math.min(TUBE_BINS - 1, (Math.max(0, s) / 1.2) * TUBE_BINS - 0.5));
        const b0 = Math.floor(x);
        const b1 = Math.min(TUBE_BINS - 1, b0 + 1);
        let half = (sl.halfW[b0] + (sl.halfW[b1] - sl.halfW[b0]) * (x - b0)) * k;
        // 실제 팔(원래 입은 옷 소매 포함) 두께를 쟀으면: 그보다 조금 넉넉히 덮되, 상품 사진 폭이 과하면 줄인다.
        const am = torsoFit ? torsoFit.armAt(sl.side, Math.min(1, a / arm.length), w) : NaN;
        return Number.isFinite(am) ? Math.min(Math.max(half, am * 1.15), am * 2.2) : Math.max(half, armR(a));
      };
      const outerAt = (a: number, off: number): Vec2 => add(arm.at(a), scale(arm.normal(a), sign * off));
      const innerAt = (a: number, off: number): Vec2 => add(arm.at(a), scale(arm.normal(a), -sign * off));
      const corrTop = sub(cap, outerAt(0, off0));
      const corrBot = sub(pit, innerAt(pitArc, off0));
      const m = sl.mesh;
      for (let i = 0; i < m.vertexCount; i++) {
        // 뿌리 중점보다 위(s<0, 접어 찍은 소매의 어깨 쪽)도 팔 위쪽으로 이어서 놓는다(뿌리에 뭉쳐 비지 않게)
        const s = sl.s[i];
        const v = sl.v[i];
        const aTop = s * topLen;
        const aBot = pitArc + Math.max(0, s) * botLen + Math.min(0, s) * topLen;
        const off = halfAt(s, (aTop + aBot) / 2);
        const fade = 1 - smoothstep(0, 0.35, s);
        const top = add(outerAt(aTop, off), scale(corrTop, fade));
        const bot = add(innerAt(aBot, off), scale(corrBot, fade));
        // v: +1 바깥선, -1 안쪽선(그 밖은 외삽)
        const t = (1 - v) / 2;
        m.dst[i * 2] = top.x + (bot.x - top.x) * t;
        m.dst[i * 2 + 1] = top.y + (bot.y - top.y) * t;
      }
      // 손목이 카메라에 가까울수록 나중에(앞에) 그린다.
      sleeveOrder.push({ mesh: m, z: body.z[sl.lmWrist] + body.z[sl.lmElbow] });
    }
    sleeveOrder.sort((a, b) => b.z - a.z);
    // 몸을 돌리면 먼 쪽 팔의 소매는 몸통 뒤로 간다. 등을 보이면 소매도 바탕색(뒷면)으로.
    const farSide = psi > 0.45 ? 1 : psi < -0.45 ? -1 : 0;
    const backView = Math.abs(psi) > Math.PI / 2;
    const behind: PartMesh[] = [];
    const front: PartMesh[] = [];
    for (const so of sleeveOrder) {
      const rig = this.sleeves.find((x) => x.mesh === so.mesh)!;
      so.mesh.back = backView;
      if (!backView && farSide !== 0 && rig.side === farSide) behind.push(so.mesh);
      else front.push(so.mesh);
    }
    // 소매 뿌리가 진동 둘레에 붙어 있으므로 소매를 몸판 "앞"에 그린다(팔을 내리면 소매가 몸판 옆을 덮음).
    this.order = [
      ...(this.backTorso ? [this.backTorso] : []),
      ...behind,
      ...(this.neckInner ? [this.neckInner] : []),
      ...(this.torso ? [this.torso] : []),
      ...front,
    ];
  }
}

/** 어깨 관절 → 팔꿈치 → 손목 폴리라인. 팔꿈치에서 방향이 부드럽게 바뀐다. */
class ArmPath {
  readonly s: Vec2;
  readonly e: Vec2;
  readonly d1: Vec2;
  readonly d2: Vec2;
  readonly upper: number;
  readonly length: number;
  private readonly joint: number;

  constructor(body: BodyFrame, lmS: number, lmE: number, lmW: number, side: 1 | -1) {
    const { shoulder, elbow, wrist } = armPose(body, lmS, lmE, lmW, side);
    this.s = shoulder;
    this.e = elbow;
    this.d1 = norm(sub(elbow, shoulder), body.down);
    this.d2 = norm(sub(wrist, elbow), this.d1);
    this.upper = Math.max(1, dist(shoulder, elbow));
    this.length = this.upper + dist(elbow, wrist);
    this.joint = 0.1 * this.length;
  }

  at(a: number): Vec2 {
    if (a <= this.upper) return add(this.s, scale(this.d1, a));
    return add(this.e, scale(this.d2, a - this.upper));
  }

  dir(a: number): Vec2 {
    const w = smoothstep(this.upper - this.joint, this.upper + this.joint, a);
    return norm(lerp(this.d1, this.d2, w), this.d1);
  }

  /** 진행 방향의 시계 방향 법선(부호는 호출하는 쪽에서 정한다) */
  normal(a: number): Vec2 {
    return perp(this.dir(a));
  }
}

/** 사각형(p00, p10, p01, p11) 안의 점 p의 쌍선형 좌표 (s, v). 사각형 밖이면 외삽 값. */
export function inverseBilinear(p: Vec2, p00: Vec2, p10: Vec2, p01: Vec2, p11: Vec2): [number, number] {
  const cr = (a: Vec2, b: Vec2): number => a.x * b.y - a.y * b.x;
  const e = sub(p10, p00);
  const f = sub(p01, p00);
  const g = { x: p00.x - p10.x + p11.x - p01.x, y: p00.y - p10.y + p11.y - p01.y };
  const h = sub(p, p00);
  const k2 = cr(g, f);
  const k1 = cr(e, f) + cr(h, g);
  const k0 = cr(h, e);
  let v: number;
  if (Math.abs(k2) < 1e-6 * Math.max(1, Math.abs(k1))) {
    v = -k0 / k1;
  } else {
    const disc = Math.sqrt(Math.max(0, k1 * k1 - 4 * k0 * k2));
    const v1 = (-k1 - disc) / (2 * k2);
    const v2 = (-k1 + disc) / (2 * k2);
    // [0,1]에 더 가까운 해를 고른다
    const dist01 = (x: number): number => (x < 0 ? -x : x > 1 ? x - 1 : 0);
    v = dist01(v1) <= dist01(v2) ? v1 : v2;
  }
  const denX = e.x + g.x * v;
  const denY = e.y + g.y * v;
  const s = Math.abs(denX) > Math.abs(denY) ? (h.x - f.x * v) / denX : (h.y - f.y * v) / denY;
  return [s, v];
}

/** 팔 랜드마크가 잘 보이지 않으면 몸 옆으로 내린 자세로 대신한다. */
export function armPose(body: BodyFrame, lmS: number, lmE: number, lmW: number, side: 1 | -1): { shoulder: Vec2; elbow: Vec2; wrist: Vec2 } {
  const w = body.shoulderW;
  const shoulder = body.p[lmS];
  const hang = norm(add(body.down, scale(body.u, side * 0.18)));
  const elbowDefault = add(shoulder, scale(hang, 0.8 * w));
  const we = smoothstep(0.25, 0.6, body.vis[lmE]);
  const elbow = lerp(elbowDefault, body.p[lmE], we);
  const upperDir = norm(sub(elbow, shoulder), hang);
  const wristDefault = add(elbow, scale(upperDir, 0.7 * w));
  const ww = smoothstep(0.25, 0.6, body.vis[lmW]) * we;
  const wrist = lerp(wristDefault, body.p[lmW], ww);
  return { shoulder, elbow, wrist };
}

/**
 * 소매를 관(튜브)으로 본 좌표: 뿌리(어깨점·겨드랑이 중점) → 끝단 중점을 중심축으로,
 * 각 정점의 축 방향 위치 s(0~1)와, 그 높이의 소매 폭 안에서의 위치 v(-1 안쪽 ~ +1 바깥쪽).
 * 소매가 뻗은 사진·늘어뜨린 사진·모델이 입은 사진 모두 같은 방식으로 다룬다(네 꼭짓점 사각형이 찌그러지지 않음).
 */
function buildTube(asset: GarmentAsset, partId: number, sh: Vec2, pit: Vec2, out: Vec2, inn: Vec2, mesh: PartMesh): { s: Float32Array; v: Float32Array; length: number; halfW: Float32Array } {
  const root = mid(sh, pit);
  const axis = sub(mid(out, inn), root);
  const length = Math.max(1, Math.hypot(axis.x, axis.y));
  const a = scale(axis, 1 / length);
  let n = perp(a);
  if (dot(sub(sh, root), n) < 0) n = scale(n, -1); // 어깨점 쪽 = 바깥
  const lo = new Float32Array(TUBE_BINS).fill(Infinity);
  const hi = new Float32Array(TUBE_BINS).fill(-Infinity);
  const { labels, width: w, height: h } = asset;
  const step = Math.max(1, Math.round(Math.min(w, h) / 300));
  for (let y = 0; y < h; y += step) {
    for (let x = 0; x < w; x += step) {
      if (labels[y * w + x] !== partId) continue;
      const d = { x: x - root.x, y: y - root.y };
      const sv = dot(d, a) / length;
      const bin = Math.floor((sv / 1.2) * TUBE_BINS);
      if (bin < 0 || bin >= TUBE_BINS) continue;
      const u = dot(d, n);
      if (u < lo[bin]) lo[bin] = u;
      if (u > hi[bin]) hi[bin] = u;
    }
  }
  const center = new Float32Array(TUBE_BINS).fill(NaN);
  const halfW = new Float32Array(TUBE_BINS).fill(NaN);
  for (let b = 0; b < TUBE_BINS; b++) {
    if (hi[b] > lo[b]) {
      center[b] = (hi[b] + lo[b]) / 2;
      halfW[b] = Math.max(2, (hi[b] - lo[b]) / 2);
    }
  }
  // 빈 구간은 가까운 구간 값으로
  const fill = (arr: Float32Array, def: number): void => {
    let last = NaN;
    for (let b = 0; b < TUBE_BINS; b++) if (Number.isFinite(arr[b])) last = arr[b]; else arr[b] = last;
    last = NaN;
    for (let b = TUBE_BINS - 1; b >= 0; b--) if (Number.isFinite(arr[b])) last = arr[b]; else arr[b] = last;
    for (let b = 0; b < TUBE_BINS; b++) if (!Number.isFinite(arr[b])) arr[b] = def;
  };
  fill(center, 0);
  fill(halfW, dist(out, inn) / 2);
  // 구간 값을 선형 보간해 소매 가장자리가 계단(톱니)이 되지 않게 한다
  const at = (arr: Float32Array, sv: number): number => {
    const x = Math.max(0, Math.min(TUBE_BINS - 1, (sv / 1.2) * TUBE_BINS - 0.5));
    const i = Math.floor(x);
    const j = Math.min(TUBE_BINS - 1, i + 1);
    return arr[i] + (arr[j] - arr[i]) * (x - i);
  };
  const s = new Float32Array(mesh.vertexCount);
  const v = new Float32Array(mesh.vertexCount);
  for (let i = 0; i < mesh.vertexCount; i++) {
    const d = { x: mesh.src[i * 2] - root.x, y: mesh.src[i * 2 + 1] - root.y };
    const sv = dot(d, a) / length;
    // 격자 여백(라벨 밖) 정점의 외삽은 좁게 제한한다(부풀거나 뒤집히지 않게)
    s[i] = Math.max(-0.15, Math.min(1.25, sv));
    v[i] = Math.max(-1.2, Math.min(1.2, (dot(d, n) - at(center, sv)) / at(halfW, sv)));
  }
  return { s, v, length, halfW };
}
