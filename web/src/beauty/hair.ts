// 헤어 컬러(염색): 머리카락 분할(+가이디드 필터 올 단위 경계) → 원래 명암을 살린 색 바꾸기.
// 어두운 머리를 밝은 색으로 바꿀 때는 머리카락 평균 밝기 대비 각 올의 밝기(상대 밝기)를 목표색에 곱해
// 결·윤기가 그대로 남게 한다. 평균 밝기는 GPU에서 작은 텍스처의 밉맵으로 구한다(CPU로 읽어 오지 않음).

import { FULLSCREEN_VS } from '../engine/shaders.ts';
import { compileProgram, type FrameTextures } from '../engine/renderer.ts';
import type { RGB } from './makeup.ts';

export interface HairLook {
  color: RGB;
  /** 0~1 염색 농도 */
  amount: number;
  /** 뿌리 → 끝 그라데이션용 끝 색(없으면 한 색) */
  tip?: RGB;
}

const W = 'const vec3 W = vec3(0.299, 0.587, 0.114);';

/** 머리카락 확률: 가이디드 필터가 있으면 원본 해상도로 정밀화한 값 */
const HAIR_PROB = /* glsl */ `
float hairProb(vec2 uv, vec3 c) {
  float p = texture(uSeg, uv).r;
  if (uUseGF > 0.5) {
    vec4 gf = textureLod(uGF, vec2(uv.x, 1.0 - uv.y), 0.0);
    p = clamp(gf.x * dot(c, W) + gf.y, 0.0, 1.0);
  }
  return p;
}`;

// ① 통계: (밝기·확률, 확률, 세로 위치·확률) → 64×64 텍스처, 밉맵 맨 위가 전체 평균
const STATS_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uGF;
uniform float uUseGF;
in vec2 vUv;
layout(location = 0) out vec4 o;
layout(location = 1) out vec4 o1;
${W}
${HAIR_PROB}
void main() {
  vec3 c = textureLod(uCam, vUv, 2.0).rgb;
  float p = hairProb(vUv, c);
  p = smoothstep(0.5, 0.9, p);
  float L = dot(c, W);
  o = vec4(L * p, p, vUv.y * p, L * L * p);
  o1 = vec4(c * p, p);
}`;

// ② 합성
const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uGF;
uniform sampler2D uStats;
uniform sampler2D uStatsC;  // 머리카락 가중 색 합(rgb·p, p)
uniform float uUseGF;
uniform vec3 uColor;
uniform vec3 uTip;
uniform float uTipOn;
uniform float uAmount;
uniform vec2 uSpan;   // 머리카락 세로 범위(uv): 뿌리, 끝
in vec2 vUv;
out vec4 o;
${W}
${HAIR_PROB}
void main() {
  vec3 c = textureLod(uCam, vUv, 0.0).rgb;
  float p = hairProb(vUv, c);
  vec4 st = textureLod(uStats, vec2(0.5), 6.0);
  float mean = st.r / max(st.g, 1e-4);
  // 올 단위 경계: 확률이 어중간한 띠(머리카락과 피부·배경이 섞인 곳)에서는 픽셀 밝기로 올을 가려낸다.
  // 머리카락이 주변보다 어두우면 주변 평균보다 어두운 픽셀이 올, 밝은 머리면 반대.
  // 확률이 아주 낮은 곳(배경)은 건드리지 않는다: 배경 무늬가 올로 오인되지 않게
  float band = smoothstep(0.15, 0.4, p) * smoothstep(0.95, 0.7, p);
  if (band > 0.001) {
    float Lp = dot(c, W);
    float Lb = dot(textureLod(uCam, vUv, 2.5).rgb, W);
    float dirn = mean < Lb ? -1.0 : 1.0;
    float t = (Lp - Lb) * dirn / max(abs(mean - Lb), 0.04);
    float strand = smoothstep(-0.15, 0.35, t);
    p = clamp(p + (strand - 0.5) * band * 0.9, 0.0, 1.0);
  }
  // 얼굴 피부·손 위에는 칠하지 않는다(분할이 이마 잔머리 쪽으로 번지는 것 방지)
  vec4 s = texture(uSeg, vUv);
  p *= 1.0 - smoothstep(0.5, 0.9, max(s.b, s.g));
  // 안쪽(확실한 머리카락)용 세기와 경계용 덮임 비율. 경계는 확률을 거의 그대로 덮임 비율로 쓴다:
  // 반쯤 섞인 픽셀을 반만 염색해야 어두운 테두리가 남지 않는다
  // 단, 확률이 아주 낮은 곳(분할이 이마·배경으로 살짝 번진 곳)까지 칠하면 밝은 색일 때 계단 모양 밝은 띠가 생기므로
  // 0.12 아래는 손대지 않고 0.12~0.65 사이에서 비율을 올린다
  float m = smoothstep(0.25, 0.9, p) * uAmount;
  float mo = smoothstep(0.12, 0.65, p) * uAmount;
  if (mo < 0.003) discard;

  float L = dot(c, W);
  float edge = smoothstep(0.45, 0.95, p);
  // 명암(결)은 비율로 잰다: 어두운 머리도 밝은 머리도 올의 밝기 '비율'은 비슷하므로,
  //  - 올 결: 픽셀 / 바로 주변(약 3px) 평균 → 가는 올 하나하나
  //  - 큰 명암: 주변(약 11px) / 머리 전체 평균 → 정수리·옆머리의 밝고 어두움(압축해서 과하지 않게)
  // 경계(확률 낮은 곳)는 배경이 섞여 있으므로 1 쪽으로 눌러 번쩍이는 테두리를 막는다
  const float e = 0.02;
  float Lsm = dot(textureLod(uCam, vUv, 1.5).rgb, W);
  float Lrg = dot(textureLod(uCam, vUv, 3.5).rgb, W);
  float strandR = clamp((L + e) / (Lsm + e), 0.45, 2.2);
  float regionR = pow(clamp((Lrg + e) / (mean + e), 0.25, 4.0), 0.6);
  float shade = mix(1.0, regionR * mix(1.0, strandR, 0.85), edge);

  vec3 target = uColor;
  if (uTipOn > 0.5) {
    float t = smoothstep(uSpan.x, uSpan.y, vUv.y);
    target = mix(uColor, uTip, t);
  }
  vec3 dyed = target * shade;
  // 실제 머리카락처럼: 밝은 결은 채도가 빠져 흰빛에 가까워지고(클리핑 대신), 그늘은 원래 색이 조금 비친다
  float dl = dot(dyed, W);
  dyed = mix(dyed, vec3(dl), smoothstep(1.15, 2.2, shade) * 0.6);
  dyed = mix(dyed, c * (dot(target, W) / max(mean, 0.03)), (1.0 - smoothstep(0.45, 0.9, shade)) * 0.35);
  dyed = clamp(dyed, 0.0, 1.0);
  // 윤기(주변보다 밝은 결)는 흰 빛으로 조금 더해 광택을 유지(경계는 제외)
  float Lblur = dot(textureLod(uCam, vUv, 3.0).rgb, W);
  dyed += vec3(max(0.0, L - Lblur) * 0.5 * edge);
  vec3 inner = mix(c, clamp(dyed, 0.0, 1.0), m);
  // 경계: 픽셀은 머리카락과 배경이 섞인 값이라, 통째로 섞으면 원래 머리색이 테두리로 남는다(밝게 염색할 때 어두운 테두리).
  // 확률을 섞인 비율로 보고, 주변 머리카락의 평균색을 빼고 그 염색색을 더한다(배경 몫은 그대로).
  vec4 hc = textureLod(uStatsC, vec2(vUv.x, 1.0 - vUv.y), 2.0);
  vec3 Hloc = hc.a > 1e-3 ? hc.rgb / hc.a : vec3(mean);
  float shadeH = clamp(pow((dot(Hloc, W) + e) / (mean + e), 0.35), 0.7, 1.4);
  vec3 outer = c + (target * shadeH - Hloc) * mo;
  o = vec4(mix(clamp(outer, 0.0, 1.0), inner, edge), 1.0);
}`;

export class HairColorRenderer {
  private readonly stats;
  private readonly comp;
  private readonly tex: WebGLTexture;
  private readonly texC: WebGLTexture;
  private readonly fbo: WebGLFramebuffer;
  private readonly vao: WebGLVertexArrayObject;
  private readonly gl: WebGL2RenderingContext;
  private static readonly N = 64;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.stats = compileProgram(gl, FULLSCREEN_VS, STATS_FS, ['uCam', 'uSeg', 'uGF', 'uUseGF']);
    this.comp = compileProgram(gl, FULLSCREEN_VS, COMPOSITE_FS, [
      'uCam', 'uSeg', 'uGF', 'uStats', 'uStatsC', 'uUseGF', 'uColor', 'uTip', 'uTipOn', 'uAmount', 'uSpan',
    ]);
    const N = HairColorRenderer.N;
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    // 반정밀도 실수가 되면 평균이 정확하지만, 안 되는 기기에서도 8비트로 충분히 동작한다
    const float = !!gl.getExtension('EXT_color_buffer_float');
    const levels = Math.log2(N) + 1;
    gl.texStorage2D(gl.TEXTURE_2D, levels, float ? gl.RGBA16F : gl.RGBA8, N, N);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.texC = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.texC);
    gl.texStorage2D(gl.TEXTURE_2D, levels, float ? gl.RGBA16F : gl.RGBA8, N, N);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.texC, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.vao = gl.createVertexArray()!;
  }

  /**
   * @param span 머리카락 세로 범위(uv, 뿌리·끝). 그라데이션에 쓴다. 얼굴 점에서 구한다.
   */
  draw(look: HairLook, t: FrameTextures, span: [number, number], alpha = 1): void {
    if (!t.hasSeg) return;
    const gl = this.gl;
    const useGF = t.gf ? 1 : 0;
    const bind = (unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null): void => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    gl.bindVertexArray(this.vao);
    gl.disable(gl.BLEND);

    // ① 통계
    const N = HairColorRenderer.N;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, N, N);
    gl.useProgram(this.stats.prog);
    bind(0, t.cam, this.stats.u.uCam);
    bind(1, t.seg, this.stats.u.uSeg);
    if (t.gf) bind(2, t.gf, this.stats.u.uGF);
    gl.uniform1f(this.stats.u.uUseGF, useGF);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, this.texC);
    gl.generateMipmap(gl.TEXTURE_2D);

    // ② 합성
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, t.width, t.height);
    const u = this.comp.u;
    gl.useProgram(this.comp.prog);
    bind(0, t.cam, u.uCam);
    bind(1, t.seg, u.uSeg);
    if (t.gf) bind(2, t.gf, u.uGF);
    bind(3, this.tex, u.uStats);
    bind(4, this.texC, u.uStatsC);
    gl.uniform1f(u.uUseGF, useGF);
    gl.uniform3f(u.uColor, ...look.color);
    gl.uniform3f(u.uTip, ...(look.tip ?? look.color));
    gl.uniform1f(u.uTipOn, look.tip ? 1 : 0);
    gl.uniform1f(u.uAmount, look.amount * alpha);
    gl.uniform2f(u.uSpan, span[0], span[1]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}
