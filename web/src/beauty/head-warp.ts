// 생성된 머리를 실시간 화면에 붙일 때, 고개 방향(3D)을 따라 휘어지게 하는 계산.
// - 생성에 쓴 캡처의 얼굴 점(x, y, 깊이)과 지금 얼굴 점으로 3D 닮음 변환(크기·회전·이동)을 구한다(Horn 사원수 방법).
// - 머리 사진의 각 격자점에 "머리 모양(타원체) 위의 깊이"를 주고, 그 3D 점을 지금 자세로 돌려 화면에 투영한다.
//   → 고개를 돌리면 가까운 쪽 머리는 그대로, 먼 쪽 머리는 눌려 보인다(평면 스티커처럼 통째로 줄지 않음).
// 화면은 약한 원근(직교 투영)으로 본다: 웹캠 거리에서는 이것으로 충분하다.

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export interface Pose3 {
  s: number;
  /** 행 우선 3×3 회전 */
  R: number[];
  t: Vec3;
}

/** 4×4 대칭 행렬의 가장 큰 고유값의 고유벡터(야코비 회전) */
function maxEigen4(a: number[][]): number[] {
  const n = 4;
  const m = a.map((r) => r.slice());
  const v = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += m[p][q] * m[p][q];
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-30) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p], mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k], mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < n; i++) if (m[i][i] > m[best][best]) best = i;
  return [v[0][best], v[1][best], v[2][best], v[3][best]];
}

/** from → to 의 3D 닮음 변환: to ≈ s·R·from + t (최소제곱) */
export function similarity3(from: Vec3[], to: Vec3[]): Pose3 | null {
  const n = Math.min(from.length, to.length);
  if (n < 3) return null;
  const ca = { x: 0, y: 0, z: 0 }, cb = { x: 0, y: 0, z: 0 };
  for (let i = 0; i < n; i++) {
    ca.x += from[i].x; ca.y += from[i].y; ca.z += from[i].z;
    cb.x += to[i].x; cb.y += to[i].y; cb.z += to[i].z;
  }
  ca.x /= n; ca.y /= n; ca.z /= n; cb.x /= n; cb.y /= n; cb.z /= n;
  let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const ax = from[i].x - ca.x, ay = from[i].y - ca.y, az = from[i].z - ca.z;
    const bx = to[i].x - cb.x, by = to[i].y - cb.y, bz = to[i].z - cb.z;
    Sxx += ax * bx; Sxy += ax * by; Sxz += ax * bz;
    Syx += ay * bx; Syy += ay * by; Syz += ay * bz;
    Szx += az * bx; Szy += az * by; Szz += az * bz;
    den += ax * ax + ay * ay + az * az;
  }
  if (den < 1e-9) return null;
  const N = [
    [Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
    [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
    [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
    [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz],
  ];
  const [w, x, y, z] = maxEigen4(N);
  const R = [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
  let num = 0;
  for (let i = 0; i < n; i++) {
    const ax = from[i].x - ca.x, ay = from[i].y - ca.y, az = from[i].z - ca.z;
    const bx = to[i].x - cb.x, by = to[i].y - cb.y, bz = to[i].z - cb.z;
    num += bx * (R[0] * ax + R[1] * ay + R[2] * az) + by * (R[3] * ax + R[4] * ay + R[5] * az) + bz * (R[6] * ax + R[7] * ay + R[8] * az);
  }
  const s = num / den;
  return {
    s,
    R,
    t: {
      x: cb.x - s * (R[0] * ca.x + R[1] * ca.y + R[2] * ca.z),
      y: cb.y - s * (R[3] * ca.x + R[4] * ca.y + R[5] * ca.z),
      z: cb.z - s * (R[6] * ca.x + R[7] * ca.y + R[8] * ca.z),
    },
  };
}

/** 회전의 전체 각도(도) */
export function rotationDeg(R: number[]): number {
  const c = Math.min(1, Math.max(-1, (R[0] + R[4] + R[8] - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

export function applyPose(P: Pose3, q: Vec3): Vec3 {
  const { s, R, t } = P;
  return {
    x: s * (R[0] * q.x + R[1] * q.y + R[2] * q.z) + t.x,
    y: s * (R[3] * q.x + R[4] * q.y + R[5] * q.z) + t.y,
    z: s * (R[6] * q.x + R[7] * q.y + R[8] * q.z) + t.z,
  };
}

/** 머리 모양 대신 쓰는 타원체(캡처 사진 좌표). 얼굴의 기울기(roll)에 맞춘 축을 쓴다 */
export interface HeadProxy {
  cx: number;
  cy: number;
  cz: number;
  /** 얼굴 가로 방향 단위벡터(얼굴 기준 왼→오) */
  ex: { x: number; y: number };
  /** 반지름: 가로 a, 세로 b, 깊이 c */
  a: number;
  b: number;
  c: number;
}

/**
 * 캡처 얼굴 점(478, 깊이 포함)으로 머리 타원체를 정한다.
 * 가로는 얼굴 폭(234–454)보다 조금 크게, 중심은 귀 높이에서 이마 쪽으로 조금, 앞면이 이마 점(151)을 지나도록 깊이를 맞춘다.
 */
export function headProxy(p: Vec3[]): HeadProxy {
  const L = p[234], Rt = p[454], fore = p[151], chin = p[152];
  const W = Math.hypot(Rt.x - L.x, Rt.y - L.y) || 1;
  const ex = { x: (Rt.x - L.x) / W, y: (Rt.y - L.y) / W };
  const ey = { x: -ex.y, y: ex.x };
  const mx = (L.x + Rt.x) / 2, my = (L.y + Rt.y) / 2;
  // 이마 방향(얼굴 위)으로 얼굴 길이의 12% 올린 곳을 중심으로
  const faceH = Math.abs((chin.x - fore.x) * ey.x + (chin.y - fore.y) * ey.y) || W;
  const cx = mx - ey.x * faceH * 0.12, cy = my - ey.y * faceH * 0.12;
  const a = W * 0.62, b = W * 0.78;
  const cz = (L.z + Rt.z) / 2 + W * 0.08;
  // 앞면이 이마를 지나도록 깊이 반지름을 정한다
  const lx = ((fore.x - cx) * ex.x + (fore.y - cy) * ex.y) / a;
  const ly = ((fore.x - cx) * ey.x + (fore.y - cy) * ey.y) / b;
  const r2 = Math.min(0.95, lx * lx + ly * ly);
  let c = (cz - fore.z) / Math.sqrt(1 - r2);
  if (!Number.isFinite(c)) c = W * 0.6;
  c = Math.min(W * 1.0, Math.max(W * 0.4, c));
  return { cx, cy, cz, ex, a, b, c };
}

/** 사진의 한 점(u, v)에 머리 깊이를 준다: 타원 안이면 앞면, 밖(늘어진 머리)은 머리 중심 깊이의 평면 */
export function proxyDepth(h: HeadProxy, u: number, v: number): number {
  const dx = u - h.cx, dy = v - h.cy;
  const lx = (dx * h.ex.x + dy * h.ex.y) / h.a;
  const ly = (dx * -h.ex.y + dy * h.ex.x) / h.b;
  const r2 = lx * lx + ly * ly;
  return r2 < 1 ? h.cz - h.c * Math.sqrt(1 - r2) : h.cz;
}

export interface WarpMesh {
  /** 격자점의 사진 좌표 0~1 (텍스처 좌표) */
  uv: Float32Array;
  /** 격자점의 3D 위치(캡처 사진 좌표) */
  pos: Vec3[];
  index: Uint16Array;
}

/** 사진의 상자(bbox) 안을 cols×rows 격자로 나누고 각 점에 머리 깊이를 준다 */
export function buildWarpMesh(h: HeadProxy, imgW: number, imgH: number, box: { x0: number; y0: number; x1: number; y1: number }, cols = 32, rows = 32): WarpMesh {
  const uv = new Float32Array((cols + 1) * (rows + 1) * 2);
  const pos: Vec3[] = [];
  let k = 0;
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const u = box.x0 + ((box.x1 - box.x0) * i) / cols;
      const v = box.y0 + ((box.y1 - box.y0) * j) / rows;
      uv[k++] = u / imgW;
      uv[k++] = v / imgH;
      pos.push({ x: u, y: v, z: proxyDepth(h, u, v) });
    }
  }
  const index = new Uint16Array(cols * rows * 6);
  k = 0;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const a = j * (cols + 1) + i, b = a + 1, c = a + cols + 1, d = c + 1;
      index[k++] = a; index[k++] = b; index[k++] = c;
      index[k++] = b; index[k++] = d; index[k++] = c;
    }
  }
  return { uv, pos, index };
}

/** 지금 자세로 격자점을 화면에 투영(x, y) */
export function projectMesh(m: WarpMesh, P: Pose3, out?: Float32Array): Float32Array {
  const o = out && out.length === m.pos.length * 2 ? out : new Float32Array(m.pos.length * 2);
  for (let i = 0; i < m.pos.length; i++) {
    const q = applyPose(P, m.pos[i]);
    o[i * 2] = q.x;
    o[i * 2 + 1] = q.y;
  }
  return o;
}

/** 고개를 돌렸을 때 먼 쪽 머리가 얼굴을 덮지 않게 지울 자리: 얼굴 윤곽(눈썹 아래). 시계 방향 점 번호 */
export const FACE_CUT = [356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300];

/**
 * 지울 자리의 삼각형(바깥 고리 투명 0 → 안쪽 고리 1, 안은 1): 경계가 부드럽게 지워진다.
 * 반환: xy(점 좌표), a(지우는 세기), index
 */
export function cutMesh(pts: { x: number; y: number }[], inner = 0.8): { xy: Float32Array; a: Float32Array; index: Uint16Array } {
  const n = pts.length;
  let cx = 0, cy = 0;
  for (const q of pts) { cx += q.x; cy += q.y; }
  cx /= n; cy /= n;
  const xy = new Float32Array((2 * n + 1) * 2);
  const a = new Float32Array(2 * n + 1);
  for (let i = 0; i < n; i++) {
    xy[i * 2] = pts[i].x; xy[i * 2 + 1] = pts[i].y; a[i] = 0;
    xy[(n + i) * 2] = cx + (pts[i].x - cx) * inner; xy[(n + i) * 2 + 1] = cy + (pts[i].y - cy) * inner; a[n + i] = 1;
  }
  xy[4 * n] = cx; xy[4 * n + 1] = cy; a[2 * n] = 1;
  const index = new Uint16Array(n * 9);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const o0 = i, o1 = (i + 1) % n, i0 = n + i, i1 = n + ((i + 1) % n);
    index[k++] = o0; index[k++] = o1; index[k++] = i0;
    index[k++] = o1; index[k++] = i1; index[k++] = i0;
    index[k++] = i0; index[k++] = i1; index[k++] = 2 * n;
  }
  return { xy, a, index };
}
