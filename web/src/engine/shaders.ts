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
// 원래 입은 옷 지우기
uniform float uRemove;       // 0 끔, 1 켬, 2 지울 곳 표시(디버그)
uniform sampler2D uCov;      // 새 옷이 덮는 정도(R)
uniform sampler2D uBg;       // 기억해 둔 배경(rgb) + 신뢰도(a)
uniform sampler2D uFill;     // 사람 뺀 배경(premultiplied, 밉맵) → 주변 배경으로 구멍 메우기
uniform sampler2D uSkin;     // 몸 피부(premultiplied, 밉맵) → 평균 피부색
uniform vec2 uSize;
uniform vec2 uSm;            // 어깨 중점(px)
uniform vec2 uLat;           // 몸 가로 방향(착용자 오른쪽→왼쪽)
uniform vec2 uAxis;          // 몸통 축 방향(아래)
uniform float uAxisLen;      // 몸통 길이(px)
uniform float uTorsoHalf;    // 몸통 반폭(px)
uniform float uPitAx;        // 어깨선 → 겨드랑이 거리(px)
uniform float uSw;           // 어깨 폭(px)
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c = textureLod(uCam, vUv, 0.0).rgb;
  if (uRemove > 0.5) {
    vec2 sUv = vec2(vUv.x, 1.0 - vUv.y);
    vec4 sg = smoothstep(vec4(0.35), vec4(0.65), texture(uSeg, vUv));
    float cov = texture(uCov, sUv).r;
    vec4 occ = texture(uOcc, sUv);
    vec2 px = vUv * uSize;
    vec2 d = px - uSm;
    float lat = abs(dot(d, uLat));
    float ax = dot(d, uAxis);
    // 지우는 곳: 원래 옷('옷' 분할) 중 새 옷이 덮지 않는 곳.
    // 바지(엉덩이 아래)와 몸통 안쪽(겨드랑이 아래 몸통 폭 안)은 무엇으로 채울지 모르므로 건드리지 않는다.
    float below = smoothstep(uAxisLen * 0.85, uAxisLen * 1.0, ax);
    float core = (1.0 - smoothstep(uTorsoHalf * 0.8, uTorsoHalf * 0.95, lat)) * smoothstep(uPitAx * 0.8, uPitAx * 1.1, ax);
    float r = sg.b * (1.0 - smoothstep(0.2, 0.6, cov)) * (1.0 - below) * (1.0 - core) * (1.0 - sg.r);
    if (uRemove > 2.5) r = 1.0; // 디버그: 채움색을 화면 전체에 표시
    if (r > 0.001) {
      // 팔 둘레: 맨팔(평균 피부색 × 원래 명암), 그 밖(모자·어깨 부풀림): 배경
      float arm = max(occ.r, occ.g);
      // 목을 덮던 원래 옷(목폴라·후드 끈 등)을 지운 자리는 피부색으로
      float neck = (1.0 - smoothstep(uSw * 0.1, uSw * 0.3, ax)) * (1.0 - smoothstep(uSw * 0.25, uSw * 0.4, lat));
      arm = max(arm, neck);
      vec4 sk = textureLod(uSkin, vec2(0.5), 10.0);
      vec3 skin = sk.a > 0.002 ? sk.rgb / sk.a : vec3(0.78, 0.6, 0.5);
      const vec3 W = vec3(0.299, 0.587, 0.114);
      float shade = clamp(dot(textureLod(uCam, vUv, 1.0).rgb, W) / max(dot(textureLod(uCam, vUv, 4.0).rgb, W), 0.04), 0.8, 1.15);
      vec3 skinC = skin * shade;
      vec4 bg = texture(uBg, sUv);
      vec4 fill = vec4(0.0);
      for (float l = 3.0; l <= 7.0; l += 1.0) {
        vec4 f = textureLod(uFill, sUv, l);
        if (f.a > 0.02) { fill = f / f.a; break; }
      }
      vec3 bgC = mix(fill.a > 0.0 ? fill.rgb : c, bg.rgb, smoothstep(0.3, 0.7, bg.a));
      vec3 rep = mix(bgC, skinC, arm);
      c = uRemove > 2.5 ? rep : uRemove > 1.5 ? mix(c, vec3(1.0, 0.0, 1.0), r) : mix(c, rep, r);
    }
  }
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
in vec2 aAux;   // x: 몸통 곡면 음영(1 = 정면), y: 보이는 정도(몸 뒤로 돌아간 면은 0)
uniform vec2 uSize;
uniform vec2 uTexSize;
out vec2 vUv;
out vec2 vAux;
void main() {
  vUv = aSrc / uTexSize;
  vAux = aAux;
  gl_Position = vec4(aDst.x / uSize.x * 2.0 - 1.0, 1.0 - aDst.y / uSize.y * 2.0, 0.0, 1.0);
}`;

export const GARMENT_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uTex;    // 상품(premultiplied RGBA)
uniform sampler2D uLabel;  // R: 부위 번호, G: 몸판 확장 플래그(봉제선 틈 방지)
uniform sampler2D uCam;    // 카메라(밉맵 포함)
uniform sampler2D uSeg;    // R 머리카락·얼굴, G 몸 피부, B 옷, A 사람
uniform sampler2D uOcc;    // R 팔 중심(몸판 가림), G 팔 주변(몸판 위 피부 판정 영역), B 손
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
uniform float uIsBack;
uniform float uCovOnly;
uniform vec2 uSm;        // 어깨 중점(px)
uniform vec2 uAxis;      // 몸통 축(아래 방향)
uniform vec2 uLat;       // 몸 가로 방향
uniform float uSw;       // 어깨 폭(px)
in vec2 vUv;
in vec2 vAux;
out vec4 o;
void main() {
  // 암시적 LOD 샘플링은 discard 전에(분기 밖에서) 해야 밉맵 선택이 정의된다.
  vec4 c = texture(uTex, vUv);
  if (uIsBack > 0.5) {
    // 뒷면: 상품 사진에 뒷면이 없으므로 원단의 바탕색(아주 흐린 밉맵)으로 칠한다(앞면 인쇄는 비치지 않게).
    vec4 base = textureLod(uTex, vUv, 6.0);
    c = vec4(base.rgb / max(base.a, 0.02) * c.a * 0.9, c.a);
  }
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
  if (uDebug > 2.5 && uDebug < 3.5) { o = vec4(occ.rgb, 1.0); return; }
  // 확률을 경계 폭만 남기고 선명하게: 0.35 이하는 0, 0.65 이상은 1
  vec4 sg = smoothstep(vec4(0.35), vec4(0.65), seg);
  // 손(B)과 머리카락·얼굴은 모든 부위를 가린다.
  // 몸판만: 팔 중심(R) + 팔 주변(G) 안의 피부. 소매는 자기 팔 피부로 구멍 나면 안 되므로 제외.
  float hide = max(occ.b, sg.r);
  hide = max(hide, uIsTorso * max(occ.r, sg.g * occ.g));
  // 목 보존: 어깨선 근처·가운데의 피부(목)는 어떤 부위도 덮지 않는다(깃이 높은 옷이 목을 가리지 않게)
  {
    vec2 d = gl_FragCoord.xy * vec2(1.0, -1.0) + vec2(0.0, uSize.y) - uSm;
    float ax = dot(d, uAxis);
    float lat = abs(dot(d, uLat));
    // 어깨선(쇄골 높이) 위의 피부만 목으로 본다. 그 아래 가슴 피부는 새 옷이 덮어야 한다.
    float neck = (1.0 - smoothstep(0.0, uSw * 0.14, ax)) * (1.0 - smoothstep(uSw * 0.26, uSw * 0.4, lat));
    hide = max(hide, sg.g * neck);
  }
  float a = uAlpha * (1.0 - clamp(hide, 0.0, 1.0));
  // 디버그 4: 가림 원인 색 표시 (빨강 손, 초록 피부×아래팔, 파랑 머리카락·얼굴, 흰색 몸판 팔 구멍)
  if (uDebug > 3.5) { o = vec4(occ.b, sg.g * occ.g, sg.r, 1.0) + vec4(occ.r * uIsTorso * 0.5); return; }
  // 목 안쪽은 원래 옷(분할의 '옷') 위에만 그린다. 분할이 없으면 피부를 덮을 위험이 있어 그리지 않는다.
  if (uIsInner > 0.5) a *= sg.b;

  // 셰이딩 전이: 원래 옷의 주름·그림자(밝기의 완만한 변화)를 새 옷에 곱한다.
  // 원래 옷의 인쇄 무늬가 비치지 않도록, 대비가 크거나 색이 바뀌는 곳은 전이하지 않는다.
  const vec3 W = vec3(0.299, 0.587, 0.114);
  vec3 cHi = textureLod(uCam, camUv, 2.0).rgb;
  vec3 cLo = textureLod(uCam, camUv, 5.0).rgb;
  float hi = dot(cHi, W);
  float lo = max(dot(cLo, W), 0.04);
  float ratio = hi / lo;
  float chromaShift = length(cHi / max(hi, 0.04) - cLo / lo);
  float wrinkle = (1.0 - smoothstep(0.16, 0.32, abs(ratio - 1.0))) * (1.0 - smoothstep(0.06, 0.16, chromaShift));
  ratio = clamp(ratio, 0.82, 1.18);
  float gate = mix(1.0, sg.b, uUseSeg) * wrinkle;
  float sh = mix(1.0, ratio, uShade * gate);
  o = vec4(c.rgb * sh * vAux.x, c.a) * a * vAux.y;
  if (uCovOnly > 0.5) o = vec4(o.a);
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

// 가중치를 곱한 색(premultiplied): 밉맵으로 줄이면 가중 평균(마스크 영역만의 평균색)이 된다.
export const MASKED_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform vec4 uPick;     // 분할 채널 선택(가중치 = dot(seg, uPick)), uInvert이면 1-가중치
uniform float uInvert;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c = textureLod(uCam, vUv, 1.0).rgb;
  float w = clamp(dot(texture(uSeg, vUv), uPick), 0.0, 1.0);
  if (uInvert > 0.5) w = 1.0 - smoothstep(0.2, 0.5, w);
  else w = smoothstep(0.5, 0.8, w);
  o = vec4(c * w, w);
}`;

// 배경 기억: 사람이 없는 곳은 조금씩 현재 영상으로 갱신하고 신뢰도를 올린다.
export const BG_UPDATE_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uPrev;
uniform float uRate;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c = textureLod(uCam, vUv, 1.0).rgb;
  float person = texture(uSeg, vUv).a;
  float free = 1.0 - smoothstep(0.1, 0.4, person);
  vec4 prev = texture(uPrev, vec2(vUv.x, 1.0 - vUv.y));
  float k = free * (prev.a < 0.05 ? 1.0 : uRate);
  o = vec4(mix(prev.rgb, c, k), min(1.0, prev.a + free * 0.08));
}`;
