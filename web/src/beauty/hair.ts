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
out vec4 o;
${W}
${HAIR_PROB}
void main() {
  vec3 c = textureLod(uCam, vUv, 2.0).rgb;
  float p = hairProb(vUv, c);
  p = smoothstep(0.5, 0.9, p);
  float L = dot(c, W);
  o = vec4(L * p, p, vUv.y * p, L * L * p);
}`;

// ② 합성
const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uGF;
uniform sampler2D uStats;
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
  // 얼굴 피부·손 위에는 칠하지 않는다(분할이 이마 잔머리 쪽으로 번지는 것 방지)
  vec4 s = texture(uSeg, vUv);
  p *= 1.0 - smoothstep(0.5, 0.9, max(s.b, s.g));
  float m = smoothstep(0.25, 0.9, p) * uAmount;
  if (m < 0.003) discard;

  vec4 st = textureLod(uStats, vec2(0.5), 6.0);
  float mean = st.r / max(st.g, 1e-4);
  float var = max(st.a / max(st.g, 1e-4) - mean * mean, 1e-5);
  float L = dot(c, W);
  // 상대 밝기: 평균이면 1. 표준편차로 나눠 대비를 일정하게 맞춘 뒤 목표 대비(0.35)로 다시 편다.
  // 경계(확률 낮은 곳)는 배경 밝기가 섞여 있으므로 상대 밝기를 1 쪽으로 눌러 번쩍이는 테두리를 막는다
  float z = (L - mean) / sqrt(var);
  float edge = smoothstep(0.45, 0.95, p);
  float rel = mix(1.0, clamp(1.0 + z * 0.35, 0.2, 1.9), edge);

  vec3 target = uColor;
  if (uTipOn > 0.5) {
    float t = smoothstep(uSpan.x, uSpan.y, vUv.y);
    target = mix(uColor, uTip, t);
  }
  vec3 dyed = target * rel;
  // 실제 머리카락처럼: 밝은 결은 채도가 빠지고, 그늘은 원래 색이 조금 비친다
  float dl = dot(dyed, W);
  dyed = mix(dyed, vec3(dl), smoothstep(1.1, 1.8, rel) * 0.45);
  dyed = mix(dyed, c * (dot(target, W) / max(mean, 0.03)), (1.0 - smoothstep(0.4, 0.9, rel)) * 0.35);
  // 윤기(주변보다 밝은 결)는 흰 빛으로 조금 더해 광택을 유지(경계는 제외)
  float Lblur = dot(textureLod(uCam, vUv, 3.0).rgb, W);
  dyed += vec3(max(0.0, L - Lblur) * 0.5 * edge);
  o = vec4(mix(c, clamp(dyed, 0.0, 1.0), m), 1.0);
}`;

export class HairColorRenderer {
  private readonly stats;
  private readonly comp;
  private readonly tex: WebGLTexture;
  private readonly fbo: WebGLFramebuffer;
  private readonly vao: WebGLVertexArrayObject;
  private readonly gl: WebGL2RenderingContext;
  private static readonly N = 64;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.stats = compileProgram(gl, FULLSCREEN_VS, STATS_FS, ['uCam', 'uSeg', 'uGF', 'uUseGF']);
    this.comp = compileProgram(gl, FULLSCREEN_VS, COMPOSITE_FS, [
      'uCam', 'uSeg', 'uGF', 'uStats', 'uUseGF', 'uColor', 'uTip', 'uTipOn', 'uAmount', 'uSpan',
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
    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tex, 0);
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

    // ② 합성
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, t.width, t.height);
    const u = this.comp.u;
    gl.useProgram(this.comp.prog);
    bind(0, t.cam, u.uCam);
    bind(1, t.seg, u.uSeg);
    if (t.gf) bind(2, t.gf, u.uGF);
    bind(3, this.tex, u.uStats);
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
