// 실시간 메이크업 렌더러: 얼굴 영역 다각형 → 마스크(반 해상도) → 부위별 경계 흐림 → 피부 질감을 살린 색 합성.
// 카메라를 그린 직후, 같은 WebGL 문맥에서 화면에 덧그린다(영상은 기기 밖으로 나가지 않는다).

import type { Vec2 } from '../engine/math.ts';
import { triangulate, type FaceRegions, type LipStyle, type MVert } from './face-regions.ts';

export type RGB = [number, number, number];

export interface MakeupLook {
  /** over: 입술 라인(-1 안쪽만 ~ 0 윤곽 그대로 ~ 1 오버립) */
  lip?: { color: RGB; amount: number; gloss: number; over?: number; style?: LipStyle };
  /** pearl: 펄(반짝이·윤기) 0~1 */
  shadow?: { color: RGB; amount: number; pearl?: number };
  blush?: { color: RGB; amount: number };
  liner?: { color: RGB; amount: number };
  brow?: { color: RGB; amount: number };
  /** 윤곽: color 쉐딩 색(하이라이터는 밝은 샴페인으로 자동), amount 세기 */
  contour?: { color: RGB; amount: number };
  /** 피부 보정: color 파운데이션 호수 색, amount 보정 세기(잡티·결 줄이기, 톤 정리) */
  base?: { color: RGB; amount: number };
}

const FULL_VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// 값이 있는 삼각형 망 채우기: 정점은 영상 픽셀 좌표, aVal 진하기, aST 부위 안 좌표(얼굴 폭 단위)
const FILL_VS = `#version 300 es
in vec2 aPos;
in float aVal;
in vec2 aST;
uniform vec2 uSize;
out float vVal;
out vec2 vST;
void main() {
  vVal = aVal;
  vST = aST;
  gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0);
}`;
const FILL_FS = `#version 300 es
precision highp float;
uniform vec4 uColor;
uniform float uSparkle;  // 1이면 B 채널에 펄 반짝이 무늬
uniform vec2 uPose;      // 얼굴 돌림·끄덕임: 움직일 때 반짝이가 켜졌다 꺼졌다 하게
in float vVal;
in vec2 vST;
out vec4 o;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec4 c = uColor * vVal;
  if (uSparkle > 0.5) {
    // 얼굴 폭의 0.9% 크기 칸마다 반짝이 하나를 둘지 정한다(피부에 붙어 함께 움직임)
    vec2 cell = floor(vST / 0.009);
    float h = hash(cell);
    float phase = hash(cell + 17.3) * 6.2832;
    float tw = 0.5 + 0.5 * sin(phase + uPose.x * 22.0 + uPose.y * 15.0);
    float on = step(0.76, h) * pow(tw, 3.0);
    c = vec4(0.0, 0.0, on * smoothstep(0.1, 0.6, vVal), 0.0);
  }
  o = c;
}`;

// 채널별 폭이 다른 가우시안 흐림(가로·세로 두 번).
// 표본 간격을 가장 넓은 채널에 맞추면 좁은 채널(립·아이라인)은 거의 안 흐려져 계단이 보이므로,
// 좁은 채널 무리와 넓은 채널 무리를 각자 간격으로 따로 모은다.
const BLUR_FS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uDir;      // (1,0) 또는 (0,1)
uniform vec4 uSigma;    // 채널별 표준편차(px)
uniform vec4 uWide;     // 1이면 넓은 무리
uniform vec2 uSize;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  vec4 sF = mix(uSigma, vec4(0.0), uWide);
  vec4 sW = mix(vec4(0.0), uSigma, uWide);
  float stepF = max(0.75, max(max(sF.x, sF.y), max(sF.z, sF.w)) * 3.0 / 7.0);
  float stepW = max(0.75, max(max(sW.x, sW.y), max(sW.z, sW.w)) * 3.0 / 7.0);
  vec4 s2 = 2.0 * uSigma * uSigma + 1e-4;
  vec4 acc = vec4(0.0);
  vec4 wsum = vec4(0.0);
  for (int i = -7; i <= 7; i++) {
    float xf = float(i) * stepF;
    float xw = float(i) * stepW;
    vec4 x = mix(vec4(xf), vec4(xw), uWide);
    vec4 w = exp(-x * x / s2);
    vec4 tf = texture(uTex, uv + uDir * xf / uSize);
    vec4 tw = texture(uTex, uv + uDir * xw / uSize);
    acc += mix(tf, tw, uWide) * w;
    wsum += w;
  }
  o = acc / wsum;
}`;

// 피부 보정용 카메라 흐림(반 해상도, 가로·세로 두 번). 첫 번은 카메라에서, 둘째는 중간 결과에서 읽는다.
const CAMBLUR_FS = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uDir;
uniform float uSigma;   // 반 해상도 px
uniform float uFromCam;
uniform vec2 uSize;
in vec2 vUv;
out vec4 o;
void main() {
  vec2 uv = uFromCam > 0.5 ? vUv : gl_FragCoord.xy / uSize;
  float lod = uFromCam > 0.5 ? 1.0 : 0.0;
  float st = max(1.0, uSigma * 3.0 / 7.0);
  vec3 acc = vec3(0.0);
  float ws = 0.0;
  for (int i = -7; i <= 7; i++) {
    float x = float(i) * st;
    float w = exp(-x * x / (2.0 * uSigma * uSigma + 1e-4));
    acc += textureLod(uSrc, uv + uDir * x / uSize, lod).rgb * w;
    ws += w;
  }
  o = vec4(acc / ws, 1.0);
}`;

// 합성의 정점 셰이더: 화면 전체 삼각형 3개 정점에서 조명(볼 피부 평균색)과 입술 평균 밝기를 한 번만 잰다.
const COMPOSITE_VS = `#version 300 es
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform vec2 uSkinPts[4];
uniform vec2 uLipPts[6];
uniform float uSkinLod;
uniform float uLipLod;
out vec2 vUv;
flat out vec3 vSkin;
flat out float vLipL;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
  // 볼 피부: 머리카락·손이 가린 표본은 얼굴 피부 확률(분할 B)로 덜 믿는다
  vec3 acc = vec3(0.0);
  float ws = 0.0;
  for (int i = 0; i < 4; i++) {
    float w = 0.05 + textureLod(uSeg, uSkinPts[i], 0.0).b;
    acc += pow(textureLod(uCam, uSkinPts[i], uSkinLod).rgb, vec3(2.2)) * w;
    ws += w;
  }
  vSkin = acc / ws;
  float l = 0.0;
  for (int i = 0; i < 6; i++) l += dot(pow(textureLod(uCam, uLipPts[i], uLipLod).rgb, vec3(2.2)), vec3(0.2126, 0.7152, 0.0722));
  vLipL = l / 6.0;
}`;

// 색 계산은 선형 공간에서 한다.
//  - 립·아이라이너(덮는 색소): 목표색 × 조명(볼 피부로 추정한 밝기·색온도) × 그 자리의 명암(입술 평균 대비)
//  - 블러셔·아이섀도·눈썹(얇게 비치는 색): 원래 피부 × (목표색 / 기준 피부색) — 조명과 피부 결이 그대로 남는다
const COMPOSITE_FS = `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uM1;   // R 립, G 아이섀도, B 블러셔, A 눈썹
uniform sampler2D uM2;   // R 아이라이너, G 눈(흰자·눈동자), B 펄 반짝이, A 이목구비(피부 보정 제외)
uniform sampler2D uM3;   // R 쉐딩, G 하이라이터
uniform sampler2D uCamBlur;
uniform vec3 uBase; uniform float uBaseAmt;
uniform vec3 uContour; uniform float uContourAmt;
uniform vec3 uLip; uniform float uLipAmt; uniform float uGloss;
uniform vec3 uShadow; uniform float uShadowAmt; uniform float uPearl;
uniform vec3 uBlush; uniform float uBlushAmt;
uniform vec3 uLiner; uniform float uLinerAmt;
uniform vec3 uBrow; uniform float uBrowAmt;
in vec2 vUv;
flat in vec3 vSkin;
flat in float vLipL;
out vec4 o;
const vec3 WL = vec3(0.2126, 0.7152, 0.0722);
const vec3 WG = vec3(0.299, 0.587, 0.114);
// 기준 피부색(밝은 조명의 중간 밝기 피부, sRGB #d1a38a)을 선형으로
const vec3 REF = vec3(0.637, 0.366, 0.254);
vec3 lin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 gam(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
void main() {
  vec2 mUv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 m1 = texture(uM1, mUv);
  vec4 m2 = texture(uM2, mUv);
  // 손(몸 피부)이 얼굴 앞에 오면 그 위에는 칠하지 않는다
  float hand = smoothstep(0.5, 0.8, texture(uSeg, vUv).g);
  m1 *= 1.0 - hand;
  m2 *= 1.0 - hand;
  // 눈 안쪽에는 아이섀도·눈썹이 번지지 않게
  float eye = smoothstep(0.35, 0.75, m2.g);
  float shadowM = smoothstep(0.0, 0.85, m1.g) * (1.0 - eye);
  // 피부 보정 범위: 얼굴 피부(분할 B)에서 이목구비를 뺀 곳
  float baseM = uBaseAmt > 0.0 ? uBaseAmt * smoothstep(0.3, 0.8, texture(uSeg, vUv).b) * (1.0 - hand) * (1.0 - smoothstep(0.1, 0.6, m2.a)) : 0.0;
  // 윤곽은 얼굴 피부에만(앞머리·배경 제외)
  vec4 m3 = uContourAmt > 0.0 ? texture(uM3, mUv) * smoothstep(0.3, 0.8, texture(uSeg, vUv).b) * (1.0 - hand) : vec4(0.0);
  float a = max(max(max(m1.r * uLipAmt, shadowM * uShadowAmt), max(m1.b * uBlushAmt, m1.a * uBrowAmt)), max(max(m2.r * uLinerAmt, baseM), max(m3.r, m3.g) * uContourAmt));
  if (a < 0.002) discard;

  vec3 c = textureLod(uCam, vUv, 0.0).rgb;
  vec3 cl = lin(c);
  float Lp = dot(cl, WL);
  // 조명 추정: 볼 피부 평균의 밝기·색온도를 기준 피부와 비교(피부 톤 차이 때문에 절반 정도만 따른다)
  float Ls = max(dot(vSkin, WL), 1e-3);
  vec3 wb = clamp(mix(vec3(1.0), (vSkin / Ls) / (REF / dot(REF, WL)), 0.7), 0.7, 1.35);
  vec3 illum = wb * pow(clamp(Ls / dot(REF, WL), 0.1, 2.5), 0.6);
  // 영상의 채도(흐린 영상·저조도 카메라는 색이 옅다)를 볼 피부 채도로 가늠해 덮는 색에도 반영
  vec3 sg = gam(vSkin);
  float sat = (max(sg.r, max(sg.g, sg.b)) - min(sg.r, min(sg.g, sg.b))) / max(max(sg.r, max(sg.g, sg.b)), 1e-3);
  float satK = mix(1.0, clamp(sat / 0.34, 0.45, 1.15), 0.6);
  // 그 자리의 명암(피부 평균 대비): 그늘·주름
  float shade = clamp(Lp / Ls, 0.0, 2.5);
  // 주변보다 어두운 점(눈썹 올)
  float Lloc = dot(lin(textureLod(uCam, vUv, 2.5).rgb), WL);

  vec3 outL = cl;
  if (uContourAmt > 0.0) {
    // 쉐딩: 비치는 색(피부 × 색 비율), 하이라이터: 빛을 받는 곳일수록 더 밝게(샴페인 빛)
    outL *= mix(vec3(1.0), clamp(lin(uContour) / REF, 0.2, 1.0), min(1.0, m3.r * uContourAmt * 1.5));
    float hl = min(1.0, m3.g * uContourAmt * 1.3) * (0.6 + 0.4 * smoothstep(0.7, 1.3, shade));
    outL = mix(outL, outL * 1.4 + vec3(0.03, 0.025, 0.015) * illum, hl * 0.8);
  }
  if (baseM > 0.002) {
    // ① 잡티·모공: 흐린 사진과의 차이(결) 중 작은 것만 줄이고, 눈가·콧방울 같은 큰 경계는 남긴다
    vec3 bl = lin(texture(uCamBlur, mUv).rgb);
    vec3 d = cl - bl;
    float amp = length(d) / max(dot(bl, WL), 0.02);
    vec3 sm = bl + d * mix(0.3, 1.0, smoothstep(0.08, 0.3, amp));
    // ② 톤 정리: 붉은기·얼룩을 얼굴 평균 피부색 쪽으로(밝기는 유지)
    float Lsm = dot(sm, WL);
    sm = mix(sm, (vSkin / Ls) * Lsm, 0.3);
    // ③ 파운데이션 호수: 기준 피부 대비 색 비율을 얇게
    sm *= mix(vec3(1.0), clamp(lin(uBase) / REF, 0.6, 1.6), 0.35);
    outL = mix(cl, sm, baseM);
  }
  // 눈썹: 올은 진하게, 올 사이 피부는 옅게 채운다
  float hairy = smoothstep(0.0, 0.3, 1.0 - Lp / max(Lloc, 1e-3));
  vec3 browT = clamp(lin(uBrow) / REF, 0.0, 1.0);
  outL *= mix(vec3(1.0), browT, clamp(m1.a * uBrowAmt * mix(0.9, 1.8, hairy), 0.0, 1.0));
  // 블러셔: 비치는 색(피부 × 비율) 70% + 가루 색 30%
  vec3 bl = lin(uBlush);
  vec3 blushC = mix(outL * clamp(bl / REF, 0.0, 1.4), bl * illum * shade, 0.3);
  outL = mix(outL, blushC, min(1.0, m1.b * uBlushAmt * 1.3));
  // 아이섀도: 비치는 색 65% + 색소 35%, 펄은 빛 받는 곳에 윤기와 반짝이
  vec3 sh = lin(uShadow);
  vec3 shP = mix(vec3(dot(sh, WL)), sh, satK);
  vec3 shadowC = mix(outL * clamp(sh / REF, 0.0, 1.4), shP * illum * shade, 0.35);
  shadowC += (sh * 0.7 + 0.3) * illum * uPearl * (0.3 * smoothstep(0.2, 0.9, shade) + 0.9 * m2.b);
  outL = mix(outL, shadowC, min(1.0, shadowM * uShadowAmt * 1.3));
  vec3 outG = gam(outL);
  // 립: 덮는 색 × 조명 × 입술 명암(입술 평균 대비), 광택은 주변보다 밝은 결을 살려 더한다
  float lipShade = clamp(Lp / max(vLipL, 1e-3), 0.15, 2.2);
  vec3 lp = lin(uLip);
  vec3 lipC = gam(mix(vec3(dot(lp, WL)), lp, satK) * illum * lipShade);
  float L = dot(c, WG);
  float Lblur = dot(textureLod(uCam, vUv, 3.0).rgb, WG);
  lipC += vec3(max(0.0, L - Lblur) * uGloss * 2.5 + smoothstep(0.55, 0.85, L) * uGloss * 0.25);
  outG = mix(outG, lipC, m1.r * uLipAmt);
  // 아이라이너
  outG = mix(outG, gam(lin(uLiner) * illum), m2.r * uLinerAmt);
  o = vec4(outG, 1.0);
}`;

interface Prog {
  prog: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null>;
}

export class MakeupRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly fill: Prog;
  private readonly blur: Prog;
  private readonly comp: Prog;
  private readonly camBlur: Prog;
  private readonly vao: WebGLVertexArrayObject;
  private readonly buf: WebGLBuffer;
  private readonly empty: WebGLVertexArrayObject;
  /** m1, m2 원본 마스크 / t1, t2 가로 흐림 / b1, b2 최종 / 피부 보정용 카메라 흐림(중간, 최종) / 윤곽 m3, t3, b3 */
  private tex: WebGLTexture[] = [];
  private fbo: WebGLFramebuffer[] = [];
  private w = 0;
  private h = 0;
  private scratch = new Float32Array(4096);

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.fill = this.program(FILL_VS, FILL_FS, ['uSize', 'uColor', 'uSparkle', 'uPose']);
    this.blur = this.program(FULL_VS, BLUR_FS, ['uTex', 'uDir', 'uSigma', 'uWide', 'uSize']);
    this.comp = this.program(COMPOSITE_VS, COMPOSITE_FS, [
      'uCam', 'uSeg', 'uM1', 'uM2', 'uLip', 'uLipAmt', 'uGloss', 'uShadow', 'uShadowAmt', 'uPearl', 'uBlush', 'uBlushAmt', 'uLiner', 'uLinerAmt', 'uBrow', 'uBrowAmt',
      'uSkinPts', 'uLipPts', 'uSkinLod', 'uLipLod', 'uCamBlur', 'uBase', 'uBaseAmt', 'uM3', 'uContour', 'uContourAmt',
    ]);
    this.camBlur = this.program(FULL_VS, CAMBLUR_FS, ['uSrc', 'uDir', 'uSigma', 'uFromCam', 'uSize']);
    this.vao = gl.createVertexArray()!;
    this.buf = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    // 정점: x, y, 진하기, s, t
    const attr = (name: string, size: number, offset: number): void => {
      const loc = gl.getAttribLocation(this.fill.prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 20, offset);
    };
    attr('aPos', 2, 0);
    attr('aVal', 1, 8);
    attr('aST', 2, 12);
    gl.bindVertexArray(null);
    this.empty = gl.createVertexArray()!;
    for (let i = 0; i < 11; i++) {
      const t = gl.createTexture()!;
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      const f = gl.createFramebuffer()!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      this.tex.push(t);
      this.fbo.push(f);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private resize(W: number, H: number): void {
    const w = Math.max(1, Math.round(W / 2));
    const h = Math.max(1, Math.round(H / 2));
    if (w === this.w && h === this.h) return;
    const gl = this.gl;
    this.w = w;
    this.h = h;
    for (const t of this.tex) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    }
  }

  /** 삼각형 망을 현재 틀(framebuffer)에 채운다. color × 정점 진하기, blend 방식 eq */
  private mesh(verts: MVert[], color: [number, number, number, number], eq: number, sparkle = false): void {
    if (verts.length < 3) return;
    const gl = this.gl;
    const need = verts.length * 5;
    if (this.scratch.length < need) this.scratch = new Float32Array(need * 2);
    const d = this.scratch;
    verts.forEach((q, j) => {
      d[j * 5] = q.x;
      d[j * 5 + 1] = q.y;
      d[j * 5 + 2] = q.v;
      d[j * 5 + 3] = q.s;
      d[j * 5 + 4] = q.t;
    });
    gl.blendEquation(eq);
    gl.uniform4f(this.fill.u.uColor, ...color);
    gl.uniform1f(this.fill.u.uSparkle, sparkle ? 1 : 0);
    gl.bufferData(gl.ARRAY_BUFFER, d.subarray(0, need), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, verts.length);
  }

  /**
   * 카메라를 그린 화면 위에 메이크업을 덧그린다.
   * @param size 영상 크기(px). regions 좌표와 같은 기준
   */
  draw(regions: FaceRegions, look: MakeupLook, cam: WebGLTexture, seg: WebGLTexture, size: { width: number; height: number }, alpha = 1): void {
    const gl = this.gl;
    const { width: W, height: H } = size;
    this.resize(W, H);
    const fw = regions.faceW;

    // ① 마스크 그리기
    gl.useProgram(this.fill.prog);
    gl.uniform2f(this.fill.u.uSize, W, H);
    gl.uniform2f(this.fill.u.uPose, regions.yaw, regions.pitch);
    gl.bindVertexArray(this.vao);
    // ARRAY_BUFFER 연결은 VAO에 저장되지 않는다: 다른 효과(타투·네일)가 바꿔 놓았을 수 있으므로 매번 다시 연결
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const poly = (pts: Vec2[]): MVert[] => triangulate(pts).map((k) => ({ x: pts[k].x, y: pts[k].y, v: 1, s: 0, t: 0 }));

    gl.viewport(0, 0, this.w, this.h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[0]);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    // 립: 안쪽 → 바깥 윤곽 띠(입 벌린 안쪽은 원래 비어 있다)
    if (look.lip) this.mesh(regions.lip, [1, 0, 0, 0], gl.MAX);
    if (look.shadow) this.mesh(regions.shadow, [0, 1, 0, 0], gl.MAX);
    if (look.blush) this.mesh(regions.blush, [0, 0, 1, 0], gl.MAX);
    if (look.brow) this.mesh(regions.brow, [0, 0, 0, 1], gl.MAX);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[1]);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (look.liner) this.mesh(regions.liner, [1, 0, 0, 0], gl.MAX);
    if (look.shadow) {
      this.mesh([...poly(regions.eyeR), ...poly(regions.eyeL)], [0, 1, 0, 0], gl.MAX);
      if ((look.shadow.pearl ?? 0) > 0) this.mesh(regions.shadow, [0, 0, 1, 0], gl.MAX, true);
    }
    if (look.base) this.mesh(regions.features.flatMap(poly), [0, 0, 0, 1], gl.MAX);
    if (look.contour) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[8]);
      gl.clear(gl.COLOR_BUFFER_BIT);
      this.mesh(regions.contour, [1, 0, 0, 0], gl.MAX);
      this.mesh(regions.highlight, [0, 1, 0, 0], gl.MAX);
    }
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);

    // ② 흐림: 부위별 폭(얼굴 폭 비율, 반 해상도 px). 섀도·블러셔는 망 자체가 옅어지므로 조금만 흐린다
    const s = fw / 2;
    // 작은 얼굴(흐린 영상)에서도 경계가 오려 붙인 듯 날카롭지 않게 최소 1px
    this.blurPass(0, 2, 4, [Math.max(0.8, s * 0.005), s * 0.015, s * 0.06, Math.max(1, s * 0.01)], [0, 1, 1, 0]);
    this.blurPass(1, 3, 5, [Math.max(0.6, s * 0.004), Math.max(0.6, s * 0.004), 0.5, s * 0.012], [0, 0, 0, 0]);
    if (look.base) this.blurCamera(cam, regions, s * 0.014);
    if (look.contour) this.blurPass(8, 9, 10, [s * 0.035, s * 0.022, 0.5, 0.5], [1, 1, 0, 0]);

    // ③ 합성(화면)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.useProgram(this.comp.prog);
    const bind = (unit: number, t: WebGLTexture, name: string): void => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.uniform1i(this.comp.u[name], unit);
    };
    bind(0, cam, 'uCam');
    bind(1, seg, 'uSeg');
    bind(2, this.tex[4], 'uM1');
    bind(3, this.tex[5], 'uM2');
    bind(4, this.tex[7], 'uCamBlur');
    bind(5, this.tex[10], 'uM3');
    const u = this.comp.u;
    const set = (c: string, a: string, part?: { color: RGB; amount: number }): void => {
      gl.uniform3f(u[c], ...(part?.color ?? [0, 0, 0]));
      gl.uniform1f(u[a], (part?.amount ?? 0) * alpha);
    };
    set('uLip', 'uLipAmt', look.lip);
    gl.uniform1f(u.uGloss, look.lip?.gloss ?? 0);
    set('uShadow', 'uShadowAmt', look.shadow);
    gl.uniform1f(u.uPearl, look.shadow?.pearl ?? 0);
    set('uBlush', 'uBlushAmt', look.blush);
    set('uLiner', 'uLinerAmt', look.liner);
    set('uBrow', 'uBrowAmt', look.brow);
    set('uBase', 'uBaseAmt', look.base);
    set('uContour', 'uContourAmt', look.contour);
    // 조명·입술 밝기 표본(영상 좌표 → 텍스처 좌표)과 평균 낼 크기(밉맵 단계)
    gl.uniform2fv(u.uSkinPts, regions.skinPts.flatMap((q) => [q.x / W, q.y / H]));
    gl.uniform2fv(u.uLipPts, regions.lipBody.flatMap((q) => [q.x / W, q.y / H]));
    gl.uniform1f(u.uSkinLod, Math.max(0, Math.log2(fw * 0.07)));
    gl.uniform1f(u.uLipLod, Math.max(0, Math.log2(fw * 0.02)));
    gl.bindVertexArray(this.empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  /** 피부 보정용: 얼굴 둘레만(scissor) 카메라를 반 해상도로 흐린다. sigma: 반 해상도 px */
  private blurCamera(cam: WebGLTexture, regions: FaceRegions, sigma: number): void {
    const gl = this.gl;
    const b = regions.bbox;
    const pad = regions.faceW * 0.1;
    // 틀(framebuffer)의 0행은 영상 맨 아래
    const x0 = Math.max(0, Math.floor((b.x0 - pad) / 2));
    const x1 = Math.min(this.w, Math.ceil((b.x1 + pad) / 2));
    const y0 = Math.max(0, Math.floor(this.h - (b.y1 + pad) / 2));
    const y1 = Math.min(this.h, Math.ceil(this.h - (b.y0 - pad) / 2));
    if (x1 <= x0 || y1 <= y0) return;
    gl.useProgram(this.camBlur.prog);
    gl.uniform2f(this.camBlur.u.uSize, this.w, this.h);
    gl.uniform1f(this.camBlur.u.uSigma, Math.max(1, sigma));
    gl.bindVertexArray(this.empty);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(this.camBlur.u.uSrc, 0);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x0, y0, x1 - x0, y1 - y0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[6]);
    gl.bindTexture(gl.TEXTURE_2D, cam);
    gl.uniform1f(this.camBlur.u.uFromCam, 1);
    gl.uniform2f(this.camBlur.u.uDir, 1, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[7]);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[6]);
    gl.uniform1f(this.camBlur.u.uFromCam, 0);
    gl.uniform2f(this.camBlur.u.uDir, 0, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private blurPass(src: number, tmp: number, dst: number, sigma: number[], wide: number[]): void {
    const gl = this.gl;
    gl.useProgram(this.blur.prog);
    gl.uniform2f(this.blur.u.uSize, this.w, this.h);
    gl.uniform4f(this.blur.u.uWide, wide[0], wide[1], wide[2], wide[3]);
    gl.uniform4f(this.blur.u.uSigma, Math.max(0.5, sigma[0]), Math.max(0.5, sigma[1]), Math.max(0.5, sigma[2]), Math.max(0.5, sigma[3]));
    gl.bindVertexArray(this.empty);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(this.blur.u.uTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[tmp]);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[src]);
    gl.uniform2f(this.blur.u.uDir, 1, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[dst]);
    gl.bindTexture(gl.TEXTURE_2D, this.tex[tmp]);
    gl.uniform2f(this.blur.u.uDir, 0, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private program(vs: string, fs: string, uniforms: string[]): Prog {
    const gl = this.gl;
    const compile = (type: number, src: string): WebGLShader => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(`메이크업 셰이더 컴파일 실패: ${gl.getShaderInfoLog(sh)}`);
      return sh;
    };
    const prog = gl.createProgram()!;
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(`메이크업 셰이더 링크 실패: ${gl.getProgramInfoLog(prog)}`);
    const u: Prog['u'] = {};
    for (const n of uniforms) u[n] = gl.getUniformLocation(prog, n);
    return { prog, u };
  }
}
