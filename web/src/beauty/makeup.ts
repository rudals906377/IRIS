// 실시간 메이크업 렌더러: 얼굴 영역 다각형 → 마스크(반 해상도) → 부위별 경계 흐림 → 피부 질감을 살린 색 합성.
// 카메라를 그린 직후, 같은 WebGL 문맥에서 화면에 덧그린다(영상은 기기 밖으로 나가지 않는다).

import type { Vec2 } from '../engine/math.ts';
import { strokeStrip, triangulate, type FaceRegions } from './face-regions.ts';

export type RGB = [number, number, number];

export interface MakeupLook {
  lip?: { color: RGB; amount: number; gloss: number };
  shadow?: { color: RGB; amount: number };
  blush?: { color: RGB; amount: number };
  liner?: { color: RGB; amount: number };
  brow?: { color: RGB; amount: number };
}

const FULL_VS = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// 다각형 채우기: 정점은 영상 픽셀 좌표
const FILL_VS = `#version 300 es
in vec2 aPos;
uniform vec2 uSize;
void main() { gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0); }`;
const FILL_FS = `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 o;
void main() { o = uColor; }`;

// 채널별 폭이 다른 가우시안 흐림(가로·세로 두 번)
const BLUR_FS = `#version 300 es
precision highp float;
uniform sampler2D uTex;
uniform vec2 uDir;      // (1,0) 또는 (0,1)
uniform vec4 uSigma;    // 채널별 표준편차(px)
uniform vec2 uSize;
out vec4 o;
void main() {
  vec2 uv = gl_FragCoord.xy / uSize;
  float maxS = max(max(uSigma.x, uSigma.y), max(uSigma.z, uSigma.w));
  float step = max(1.0, maxS * 3.0 / 7.0);
  vec4 acc = vec4(0.0);
  vec4 wsum = vec4(0.0);
  for (int i = -7; i <= 7; i++) {
    float x = float(i) * step;
    vec4 w = exp(-x * x / (2.0 * uSigma * uSigma + 1e-4));
    acc += texture(uTex, uv + uDir * x / uSize) * w;
    wsum += w;
  }
  o = acc / wsum;
}`;

const COMPOSITE_FS = `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uM1;   // R 립, G 아이섀도, B 블러셔, A 눈썹
uniform sampler2D uM2;   // R 아이라이너, G 눈(흰자·눈동자)
uniform vec3 uLip; uniform float uLipAmt; uniform float uGloss;
uniform vec3 uShadow; uniform float uShadowAmt;
uniform vec3 uBlush; uniform float uBlushAmt;
uniform vec3 uLiner; uniform float uLinerAmt;
uniform vec3 uBrow; uniform float uBrowAmt;
in vec2 vUv;
out vec4 o;
const vec3 W = vec3(0.299, 0.587, 0.114);
// 원래 밝기(주름·그늘)를 살린 채 색만 바꾼다: 밝기 0.5에서 목표색이 되도록
vec3 tint(vec3 color, float L) { return color * (0.35 + 1.3 * L); }
void main() {
  vec3 c = textureLod(uCam, vUv, 0.0).rgb;
  vec2 mUv = vec2(vUv.x, 1.0 - vUv.y);
  vec4 m1 = texture(uM1, mUv);
  vec4 m2 = texture(uM2, mUv);
  // 손(몸 피부)이 얼굴 앞에 오면 그 위에는 칠하지 않는다
  float hand = smoothstep(0.5, 0.8, texture(uSeg, vUv).g);
  m1 *= 1.0 - hand;
  m2 *= 1.0 - hand;
  float L = dot(c, W);
  vec3 out3 = c;
  // 눈 안쪽에는 아이섀도·눈썹이 번지지 않게
  float eye = smoothstep(0.35, 0.75, m2.g);
  // 넓게 흐린 영역은 가운데 농도가 떨어지므로 다시 끌어올린다
  float shadowM = smoothstep(0.0, 0.6, m1.g) * (1.0 - eye);
  // 눈썹: 곱하기로 진하게만
  out3 = mix(out3, out3 * clamp(uBrow * 2.2, 0.0, 1.0), m1.a * uBrowAmt);
  // 블러셔·아이섀도: 피부 밝기를 살린 색조
  out3 = mix(out3, tint(uBlush, L), m1.b * uBlushAmt);
  // 아이섀도: 색조 절반 + 색소처럼 곱하기 절반(어두운 색은 확실히 어둡게)
  vec3 shadowC = mix(tint(uShadow, L), out3 * uShadow * 1.6, 0.5);
  out3 = mix(out3, shadowC, shadowM * uShadowAmt);
  // 립: 색 + 광택(주변보다 밝은 결을 살려 더함)
  float Lblur = dot(textureLod(uCam, vUv, 3.0).rgb, W);
  vec3 lip = tint(uLip, L) + vec3(max(0.0, L - Lblur) * uGloss * 2.5 + smoothstep(0.55, 0.85, L) * uGloss * 0.25);
  out3 = mix(out3, lip, m1.r * uLipAmt);
  // 아이라이너
  out3 = mix(out3, uLiner, m2.r * uLinerAmt);
  float a = max(max(max(m1.r * uLipAmt, shadowM * uShadowAmt), max(m1.b * uBlushAmt, m1.a * uBrowAmt)), m2.r * uLinerAmt);
  o = vec4(out3, 1.0) * step(0.002, a);
  if (a < 0.002) discard;
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
  private readonly vao: WebGLVertexArrayObject;
  private readonly buf: WebGLBuffer;
  private readonly empty: WebGLVertexArrayObject;
  /** m1, m2 원본 마스크 / t1, t2 가로 흐림 / b1, b2 최종 */
  private tex: WebGLTexture[] = [];
  private fbo: WebGLFramebuffer[] = [];
  private w = 0;
  private h = 0;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.fill = this.program(FILL_VS, FILL_FS, ['uSize', 'uColor']);
    this.blur = this.program(FULL_VS, BLUR_FS, ['uTex', 'uDir', 'uSigma', 'uSize']);
    this.comp = this.program(FULL_VS, COMPOSITE_FS, [
      'uCam', 'uSeg', 'uM1', 'uM2', 'uLip', 'uLipAmt', 'uGloss', 'uShadow', 'uShadowAmt', 'uBlush', 'uBlushAmt', 'uLiner', 'uLinerAmt', 'uBrow', 'uBrowAmt',
    ]);
    this.vao = gl.createVertexArray()!;
    this.buf = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    const loc = gl.getAttribLocation(this.fill.prog, 'aPos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.empty = gl.createVertexArray()!;
    for (let i = 0; i < 6; i++) {
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
    gl.bindVertexArray(this.vao);
    // ARRAY_BUFFER 연결은 VAO에 저장되지 않는다: 다른 효과(타투·네일)가 바꿔 놓았을 수 있으므로 매번 다시 연결
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    const polys = (list: { pts: Vec2[]; tris: number[] }[], color: [number, number, number, number], eq: number): void => {
      gl.blendEquation(eq);
      gl.uniform4f(this.fill.u.uColor, ...color);
      for (const { pts, tris } of list) {
        if (tris.length < 3) continue;
        const d = new Float32Array(tris.length * 2);
        tris.forEach((k, j) => {
          d[j * 2] = pts[k].x;
          d[j * 2 + 1] = pts[k].y;
        });
        gl.bufferData(gl.ARRAY_BUFFER, d, gl.DYNAMIC_DRAW);
        gl.drawArrays(gl.TRIANGLES, 0, tris.length);
      }
    };
    const poly = (pts: Vec2[]): { pts: Vec2[]; tris: number[] } => ({ pts, tris: triangulate(pts) });
    const strip = (pts: Vec2[]): { pts: Vec2[]; tris: number[] } => {
      const tris: number[] = [];
      for (let i = 0; i + 3 < pts.length + 1 && i + 2 < pts.length; i += 2) tris.push(i, i + 1, i + 2, i + 1, i + 3, i + 2);
      return { pts, tris: tris.filter((_, k) => tris[k] < pts.length) };
    };
    const ellipse = (e: FaceRegions['blushR']): Vec2[] => {
      const out: Vec2[] = [];
      const nx = -e.dir.y;
      const ny = e.dir.x;
      for (let k = 0; k < 28; k++) {
        const a = (k / 28) * Math.PI * 2;
        const u = Math.cos(a) * e.rx;
        const v = Math.sin(a) * e.ry;
        out.push({ x: e.c.x + e.dir.x * u + nx * v, y: e.c.y + e.dir.y * u + ny * v });
      }
      return out;
    };

    gl.viewport(0, 0, this.w, this.h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[0]);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (look.lip) {
      polys([poly(regions.lipsOuter)], [1, 0, 0, 0], gl.MAX);
      // 입 벌린 안쪽은 비운다(R만 0으로)
      polys([poly(regions.lipsInner)], [0, 1, 1, 1], gl.MIN);
    }
    if (look.shadow) polys([poly(regions.shadowR), poly(regions.shadowL)], [0, 1, 0, 0], gl.MAX);
    if (look.blush) polys([poly(ellipse(regions.blushR)), poly(ellipse(regions.blushL))], [0, 0, 1, 0], gl.MAX);
    if (look.brow) polys([poly(regions.browR), poly(regions.browL)], [0, 0, 0, 1], gl.MAX);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo[1]);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (look.liner) {
      const lw = fw * 0.012;
      polys([strip(strokeStrip(regions.linerR, lw)), strip(strokeStrip(regions.linerL, lw))], [1, 0, 0, 0], gl.MAX);
    }
    if (look.shadow) polys([poly(regions.eyeR), poly(regions.eyeL)], [0, 1, 0, 0], gl.MAX);
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);

    // ② 흐림: 부위별 폭(얼굴 폭 비율, 반 해상도 px)
    const s = fw / 2;
    this.blurPass(0, 2, 4, [s * 0.008, s * 0.05, s * 0.09, s * 0.012]);
    this.blurPass(1, 3, 5, [s * 0.004, s * 0.004, 0.5, 0.5]);

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
    const u = this.comp.u;
    const set = (c: string, a: string, part?: { color: RGB; amount: number }): void => {
      gl.uniform3f(u[c], ...(part?.color ?? [0, 0, 0]));
      gl.uniform1f(u[a], (part?.amount ?? 0) * alpha);
    };
    set('uLip', 'uLipAmt', look.lip);
    gl.uniform1f(u.uGloss, look.lip?.gloss ?? 0);
    set('uShadow', 'uShadowAmt', look.shadow);
    set('uBlush', 'uBlushAmt', look.blush);
    set('uLiner', 'uLinerAmt', look.liner);
    set('uBrow', 'uBrowAmt', look.brow);
    gl.bindVertexArray(this.empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  private blurPass(src: number, tmp: number, dst: number, sigma: number[]): void {
    const gl = this.gl;
    gl.useProgram(this.blur.prog);
    gl.uniform2f(this.blur.u.uSize, this.w, this.h);
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
