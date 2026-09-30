// WebGL2 셰이더(카메라 화면·분할 보기·가이디드 필터). 효과별 셰이더는 각 효과 모듈(beauty/)에 있다.
// 좌표 규칙: 카메라·분할 텍스처는 0행이 영상 맨 위이므로 uv = (x/W, y/H)로 바로 샘플링한다.

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
uniform sampler2D uGF;
uniform float uUseGF;
uniform float uDebugSeg;
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
    c = mix(c, vec3(0.95, 0.35, 0.8), s.r * 0.45);   // 머리카락
    c = mix(c, vec3(1.0, 0.7, 0.2), s.g * 0.45);     // 몸 피부(손·팔·목)
    c = mix(c, vec3(0.2, 0.8, 1.0), s.b * 0.35);     // 얼굴 피부
  }
  o = vec4(c, 1.0);
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
