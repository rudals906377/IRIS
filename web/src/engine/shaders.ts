// WebGL2 셰이더. 좌표 규칙: 화면 픽셀(x 오른쪽, y 아래)을 정점에서 클립 좌표로 바꾼다.
// 카메라·분할 텍스처는 0행이 영상 맨 위이므로 uv = (x/W, y/H)로 바로 샘플링한다.

export const FULLSCREEN_VS = /* glsl */ `#version 300 es
out vec2 vUv;
void main() {
  // 정점 3개로 화면 전체를 덮는 삼각형
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export const CAMERA_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uOcc;
uniform sampler2D uGF;
uniform float uUseGF;
uniform float uDebugSeg;
uniform float uDebugOcc;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c = textureLod(uCam, vUv, 0.0).rgb;
  if (uDebugSeg > 0.5) {
    vec4 s = texture(uSeg, vUv);
    if (uUseGF > 0.5) {
      vec4 gf = textureLod(uGF, vec2(vUv.x, 1.0 - vUv.y), 0.0);
      float I = dot(c, vec3(0.299, 0.587, 0.114));
      s.r = clamp(gf.x * I + gf.y, 0.0, 1.0);
      s.g = clamp(gf.z * I + gf.w, 0.0, 1.0);
    }
    c = mix(c, vec3(0.95, 0.35, 0.8), s.r * 0.45);   // 머리카락·얼굴
    c = mix(c, vec3(1.0, 0.7, 0.2), s.g * 0.45);     // 몸 피부
    c = mix(c, vec3(0.2, 0.8, 1.0), s.b * 0.35);     // 옷
  }
  if (uDebugOcc > 0.5) {
    vec4 oc = texture(uOcc, vec2(vUv.x, 1.0 - vUv.y));
    c = mix(c, vec3(1.0, 0.2, 0.2), oc.r * 0.35);
    c = mix(c, vec3(0.2, 1.0, 0.3), oc.b * 0.5);
  }
  o = vec4(c, 1.0);
}`;

export const OCC_VS = /* glsl */ `#version 300 es
in vec2 aPos;
in vec2 aA;
in vec2 aB;
in float aR;
in vec3 aCh;
uniform vec2 uSize;
out vec2 vPix;
flat out vec2 vA;
flat out vec2 vB;
flat out float vR;
flat out vec3 vCh;
void main() {
  vPix = aPos; vA = aA; vB = aB; vR = aR; vCh = aCh;
  gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0);
}`;

export const OCC_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vPix;
flat in vec2 vA;
flat in vec2 vB;
flat in float vR;
flat in vec3 vCh;
uniform float uFeather;
out vec4 o;
void main() {
  vec2 ab = vB - vA;
  float t = clamp(dot(vPix - vA, ab) / max(dot(ab, ab), 1e-4), 0.0, 1.0);
  float d = length(vPix - (vA + ab * t));
  float a = 1.0 - smoothstep(vR - uFeather, vR + uFeather, d);
  o = vec4(vCh * a, 1.0);
}`;

export const GARMENT_VS = /* glsl */ `#version 300 es
in vec2 aSrc;
in vec2 aDst;
uniform vec2 uSize;
uniform vec2 uTexSize;
out vec2 vUv;
void main() {
  vUv = aSrc / uTexSize;
  gl_Position = vec4(aDst.x / uSize.x * 2.0 - 1.0, 1.0 - aDst.y / uSize.y * 2.0, 0.0, 1.0);
}`;

export const GARMENT_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uTex;    // 상품(premultiplied RGBA)
uniform sampler2D uLabel;  // R: 부위 번호, G: 몸판 확장 플래그(봉제선 틈 방지)
uniform sampler2D uCam;    // 카메라(밉맵 포함)
uniform sampler2D uSeg;    // R 머리카락·얼굴, G 몸 피부, B 옷, A 사람
uniform sampler2D uOcc;    // R 몸판 가림 팔, G 피부 허용, B 손
uniform sampler2D uGF;     // 가이디드 필터 계수 (a1, b1, a2, b2)
uniform float uUseGF;
uniform vec2 uSize;
uniform float uPart;
uniform float uIsTorso;
uniform float uIsInner;
uniform float uAlpha;
uniform float uShade;
uniform float uUseSeg;
uniform float uDebug;
in vec2 vUv;
out vec4 o;
void main() {
  // 암시적 LOD 샘플링은 discard 전에(분기 밖에서) 해야 밉맵 선택이 정의된다.
  vec4 c = texture(uTex, vUv);
  vec2 lab = textureLod(uLabel, vUv, 0.0).rg;
  vec2 fc = gl_FragCoord.xy / uSize;
  vec2 camUv = vec2(fc.x, 1.0 - fc.y);
  vec4 occ = textureLod(uOcc, fc, 0.0);
  vec4 seg = textureLod(uSeg, camUv, 0.0) * uUseSeg;
  if (uUseGF > 0.5) {
    // 원본 해상도 밝기 I로 머리카락·피부 확률을 다시 계산(머리카락 올 단위 경계)
    vec4 gf = textureLod(uGF, fc, 0.0);
    float I = dot(textureLod(uCam, camUv, 0.0).rgb, vec3(0.299, 0.587, 0.114));
    seg.r = clamp(gf.x * I + gf.y, 0.0, 1.0);
    seg.g = clamp(gf.z * I + gf.w, 0.0, 1.0);
  }

  float part = floor(lab.r * 255.0 + 0.5);
  bool mine = abs(part - uPart) < 0.5 || (uIsTorso > 0.5 && lab.g > 0.5);
  if (uDebug > 0.5 && uDebug < 1.5) { o = vec4(part / 3.0, 0.0, 1.0 - part / 3.0, 1.0); return; }
  if (!mine || c.a < 0.004) discard;
  if (uDebug > 1.5 && uDebug < 2.5) { o = vec4(1.0, 0.0, 0.0, 1.0); return; }
  if (uDebug > 2.5) { o = vec4(occ.rgb, 1.0); return; }
  // 확률을 경계 폭만 남기고 선명하게: 0.35 이하는 0, 0.65 이상은 1
  vec4 sg = smoothstep(vec4(0.35), vec4(0.65), seg);
  float hide = max(occ.b, sg.r);
  hide = max(hide, sg.g * occ.g);
  hide = max(hide, occ.r * uIsTorso);
  float a = uAlpha * (1.0 - clamp(hide, 0.0, 1.0));
  // 목 안쪽은 원래 옷(분할의 '옷') 위에만 그린다. 분할이 없으면 피부를 덮을 위험이 있어 그리지 않는다.
  if (uIsInner > 0.5) a *= sg.b;

  // 셰이딩 전이: 원래 옷의 주름·그림자(밝기의 고주파 성분)를 새 옷에 곱한다.
  const vec3 W = vec3(0.299, 0.587, 0.114);
  float hi = dot(textureLod(uCam, camUv, 1.5).rgb, W);
  float lo = dot(textureLod(uCam, camUv, 5.0).rgb, W);
  float ratio = clamp(hi / max(lo, 0.04), 0.78, 1.22);
  float gate = mix(1.0, sg.b, uUseSeg);
  float sh = mix(1.0, ratio, uShade * gate);
  o = vec4(c.rgb * sh, c.a) * a;
}`;

// ---- 가이디드 필터(He et al.): 저해상도 분할 확률을 카메라 밝기 경계에 맞춰 원본 해상도로 복원 ----
// 1단계: 창 안의 평균 I, I², p, I·p (p = 머리카락·얼굴 확률, 몸 피부 확률)
export const GF_STATS_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform vec2 uLow;      // 저해상도 크기(px)
uniform float uLod;     // 저해상도에 맞는 카메라 밉맵 단계
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
const int R = 4;
const vec3 W = vec3(0.299, 0.587, 0.114);
void main() {
  vec2 fc = gl_FragCoord.xy / uLow;
  vec2 uv0 = vec2(fc.x, 1.0 - fc.y);
  vec2 texel = 1.0 / uLow;
  vec4 s0 = vec4(0.0);
  vec2 s1 = vec2(0.0);
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      vec2 uv = uv0 + vec2(float(dx), float(dy)) * texel;
      float I = dot(textureLod(uCam, uv, uLod).rgb, W);
      vec2 p = textureLod(uSeg, uv, 0.0).rg;
      s0 += vec4(I, I * I, p.x, I * p.x);
      s1 += vec2(p.y, I * p.y);
    }
  }
  float n = float((2 * R + 1) * (2 * R + 1));
  o0 = s0 / n;
  o1 = vec4(s1 / n, 0.0, 1.0);
}`;

// 2단계: 창마다 선형 계수 a, b를 구하고 그 평균을 낸다. 결과 (a1, b1, a2, b2)
export const GF_MEAN_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uS0;
uniform sampler2D uS1;
uniform vec2 uLow;
uniform float uEps;
out vec4 o;
const int R = 4;
void main() {
  vec2 fc = gl_FragCoord.xy / uLow;
  vec2 texel = 1.0 / uLow;
  vec4 acc = vec4(0.0);
  for (int dy = -R; dy <= R; dy++) {
    for (int dx = -R; dx <= R; dx++) {
      vec2 uv = fc + vec2(float(dx), float(dy)) * texel;
      vec4 a = textureLod(uS0, uv, 0.0);
      vec2 b = textureLod(uS1, uv, 0.0).rg;
      float varI = max(a.y - a.x * a.x, 0.0);
      float a1 = (a.w - a.x * a.z) / (varI + uEps);
      float a2 = (b.y - a.x * b.x) / (varI + uEps);
      acc += vec4(a1, a.z - a1 * a.x, a2, b.x - a2 * a.x);
    }
  }
  o = acc / float((2 * R + 1) * (2 * R + 1));
}`;
