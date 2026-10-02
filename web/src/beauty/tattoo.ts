// 타투 합성: 격자(tattoo-place.ts)에 도안을 입혀 그리되, 몸 피부(분할 G, 가이디드 필터 정밀화)에만 남긴다.
// 옷·머리카락이 덮은 곳은 자연히 가려진다. 잉크는 피부 아래 색소처럼 곱하기로 섞어 피부 결·명암이 비친다.

import type { FrameTextures } from '../engine/renderer.ts';
import { compileProgram } from '../engine/renderer.ts';
import type { TattooMesh } from './tattoo-place.ts';

const VS = /* glsl */ `#version 300 es
in vec2 aPos;
in vec2 aUv;
in float aVis;
uniform vec2 uSize;
out vec2 vUv;
out float vVis;
void main() {
  vUv = aUv;
  vVis = aVis;
  gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0);
}`;

const FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uDesign;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uGF;
uniform float uUseGF;
uniform vec2 uSize;
uniform float uAmount;
uniform vec3 uInk;      // 도안이 검은색일 때 쓰는 잉크 색
in vec2 vUv;
in float vVis;
out vec4 o;
const vec3 W = vec3(0.299, 0.587, 0.114);
void main() {
  vec2 camUv = vec2(gl_FragCoord.x / uSize.x, 1.0 - gl_FragCoord.y / uSize.y);
  vec3 c = textureLod(uCam, camUv, 0.0).rgb;
  vec4 seg = texture(uSeg, camUv);
  float skin = seg.g;
  if (uUseGF > 0.5) {
    vec4 gf = textureLod(uGF, vec2(camUv.x, 1.0 - camUv.y), 0.0);
    skin = clamp(gf.z * dot(c, W) + gf.w, 0.0, 1.0);
  }
  // 가까이 찍힌 목은 분할이 '얼굴 피부'로 판단하기도 한다. 위치는 턱 아래로 정해지므로 얼굴 피부도 피부로 본다
  skin = smoothstep(0.35, 0.8, max(skin, seg.b));
  // 잉크가 살짝 번진 느낌: 밉맵 한 단계 섞기
  vec4 d = mix(texture(uDesign, vUv), textureLod(uDesign, vUv, 1.5), 0.35);
  float ink = d.a * smoothstep(0.05, 0.45, vVis) * skin * uAmount;
  if (ink < 0.004) discard;
  // 검은 도안은 지정 잉크색, 색 도안은 그 색(너무 밝은 색은 피부에서 잘 안 보이므로 조금 진하게)
  vec3 col = d.a > 0.0 ? d.rgb / max(d.a, 1e-3) : vec3(0.0);
  vec3 inkCol = mix(uInk, col * 0.85, step(0.08, max(max(col.r, col.g), col.b)));
  // 피부 아래 색소: 곱하기(명암·결 유지) + 아주 조금 바로 덮기
  vec3 tinted = mix(c * inkCol * 1.15, inkCol, 0.15);
  o = vec4(tinted, ink);
}`;

export class TattooRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog;
  private readonly vao: WebGLVertexArrayObject;
  private readonly vbo: WebGLBuffer;
  private readonly ibo: WebGLBuffer;
  private readonly tex: WebGLTexture;
  private designSource: TexImageSource | null = null;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = compileProgram(gl, VS, FS, ['uDesign', 'uCam', 'uSeg', 'uGF', 'uUseGF', 'uSize', 'uAmount', 'uInk']);
    this.vao = gl.createVertexArray()!;
    this.vbo = gl.createBuffer()!;
    this.ibo = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    const attr = (name: string, size: number, offset: number): void => {
      const loc = gl.getAttribLocation(this.prog.prog, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 20, offset);
    };
    attr('aPos', 2, 0);
    attr('aUv', 2, 8);
    attr('aVis', 1, 16);
    gl.bindVertexArray(null);
    this.tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  /** 도안 바꾸기(바뀔 때만 GPU로 올린다). 캔버스는 알파 미리곱(premultiplied) 상태로 올린다. */
  setDesign(src: TexImageSource): void {
    if (src === this.designSource) return;
    this.designSource = src;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  draw(mesh: TattooMesh, t: FrameTextures, amount: number, ink: [number, number, number]): void {
    if (!this.designSource) return;
    const gl = this.gl;
    const u = this.prog.u;
    gl.useProgram(this.prog.prog);
    const bind = (unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null): void => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(0, this.tex, u.uDesign);
    bind(1, t.cam, u.uCam);
    bind(2, t.seg, u.uSeg);
    if (t.gf) bind(3, t.gf, u.uGF);
    gl.uniform1f(u.uUseGF, t.gf ? 1 : 0);
    gl.uniform2f(u.uSize, t.width, t.height);
    // 분할이 없으면 피부 판정을 할 수 없으므로 그대로 그린다(분할 텍스처가 비어 있으면 안 보임)
    gl.uniform1f(u.uAmount, amount * mesh.confidence);
    gl.uniform3f(u.uInk, ...ink);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.data, gl.DYNAMIC_DRAW);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawElements(gl.TRIANGLES, mesh.index.length, gl.UNSIGNED_SHORT, 0);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }
}
