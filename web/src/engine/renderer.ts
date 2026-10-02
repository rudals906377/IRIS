// WebGL2 합성 기반: 카메라 화면, 분할 텍스처, 가이디드 필터(머리카락 올 단위 경계).
// 각 효과(메이크업·헤어 등)는 같은 WebGL 문맥에서 카메라를 그린 뒤 덧그린다(Effect.draw).

import { CAMERA_FS, FULLSCREEN_VS, GF_MEAN_FS, GF_STATS_FS } from './shaders.ts';

type Uniforms = Record<string, WebGLUniformLocation | null>;
interface Program {
  prog: WebGLProgram;
  u: Uniforms;
}

/** 가이디드 필터 저해상도 배율(1/2)과 그에 맞는 카메라 밉맵 단계. 1/4에서는 앞머리 올이 뭉개져 1/2로 올림 */
const GF_DOWN = 2;
const GF_LOD = 1;

interface GuidedFilter {
  statsProg: Program;
  meanProg: Program;
  s0: WebGLTexture;
  s1: WebGLTexture;
  mean: WebGLTexture;
  statsFbo: WebGLFramebuffer;
  meanFbo: WebGLFramebuffer;
  w: number;
  h: number;
}

/** 효과가 쓰는 공용 GPU 자원 */
export interface FrameTextures {
  gl: WebGL2RenderingContext;
  cam: WebGLTexture;
  seg: WebGLTexture;
  /** 가이디드 필터 계수(a1,b1,a2,b2): 머리카락(R)·몸 피부(G) 확률 = a·밝기 + b. 없으면 null */
  gf: WebGLTexture | null;
  hasSeg: boolean;
  width: number;
  height: number;
}

export interface DrawOptions {
  useSeg: boolean;
  debugSeg: boolean;
  /** 머리카락·피부 경계를 원본 해상도로 정밀화(가이디드 필터) */
  refine?: boolean;
  /** 카메라를 그린 뒤 화면에 덧그리는 효과들 */
  effects?: ((t: FrameTextures) => void)[];
}

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private width = 0;
  private height = 0;
  private readonly camProg: Program;
  private readonly camTex: WebGLTexture;
  private readonly segTex: WebGLTexture;
  private readonly emptyVao: WebGLVertexArrayObject;
  private hasSeg = false;
  private readonly gf: GuidedFilter | null;

  constructor(canvas: HTMLCanvasElement, lowLatency = true) {
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      premultipliedAlpha: true,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
      // 합성기를 거치지 않는 저지연 캔버스 힌트(지원 브라우저에서만 효과)
      desynchronized: lowLatency,
    });
    if (!gl) throw new Error('이 브라우저는 WebGL2를 지원하지 않습니다.');
    this.gl = gl;
    this.camProg = this.program(FULLSCREEN_VS, CAMERA_FS, ['uCam', 'uSeg', 'uGF', 'uUseGF', 'uDebugSeg']);
    this.camTex = this.texture(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR);
    this.segTex = this.texture(gl.LINEAR, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, this.segTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    this.emptyVao = gl.createVertexArray()!;
    this.gf = this.createGuidedFilter();
  }

  /** 부동소수점 렌더 타깃이 없으면(일부 구형 기기) 정밀화를 끈다. */
  private createGuidedFilter(): GuidedFilter | null {
    const gl = this.gl;
    if (!gl.getExtension('EXT_color_buffer_float')) return null;
    const statsProg = this.program(FULLSCREEN_VS, GF_STATS_FS, ['uCam', 'uSeg', 'uLow', 'uLod']);
    const meanProg = this.program(FULLSCREEN_VS, GF_MEAN_FS, ['uS0', 'uS1', 'uLow', 'uEps']);
    return {
      statsProg,
      meanProg,
      s0: this.texture(gl.LINEAR, gl.LINEAR),
      s1: this.texture(gl.LINEAR, gl.LINEAR),
      mean: this.texture(gl.LINEAR, gl.LINEAR),
      statsFbo: gl.createFramebuffer()!,
      meanFbo: gl.createFramebuffer()!,
      w: 0,
      h: 0,
    };
  }

  get refineAvailable(): boolean {
    return this.gf !== null;
  }

  get size(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height) return;
    const gl = this.gl;
    this.width = width;
    this.height = height;
    (gl.canvas as HTMLCanvasElement).width = width;
    (gl.canvas as HTMLCanvasElement).height = height;
    if (this.gf) {
      const g = this.gf;
      g.w = Math.max(1, Math.ceil(width / GF_DOWN));
      g.h = Math.max(1, Math.ceil(height / GF_DOWN));
      for (const t of [g.s0, g.s1, g.mean]) {
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, g.w, g.h, 0, gl.RGBA, gl.HALF_FLOAT, null);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, g.statsFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, g.s0, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, g.s1, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, g.meanFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, g.mean, 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private runGuidedFilter(): void {
    const gl = this.gl;
    const g = this.gf!;
    gl.disable(gl.BLEND);
    gl.bindVertexArray(this.emptyVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, g.statsFbo);
    gl.viewport(0, 0, g.w, g.h);
    gl.useProgram(g.statsProg.prog);
    this.bindTex(0, this.camTex, g.statsProg.u.uCam);
    this.bindTex(1, this.segTex, g.statsProg.u.uSeg);
    gl.uniform2f(g.statsProg.u.uLow, g.w, g.h);
    gl.uniform1f(g.statsProg.u.uLod, GF_LOD);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, g.meanFbo);
    gl.useProgram(g.meanProg.prog);
    this.bindTex(0, g.s0, g.meanProg.u.uS0);
    this.bindTex(1, g.s1, g.meanProg.u.uS1);
    gl.uniform2f(g.meanProg.u.uLow, g.w, g.h);
    // 작을수록 밝기 경계에 더 민감(머리카락 올이 살아남), 너무 작으면 잡음이 마스크에 섞인다.
    gl.uniform1f(g.meanProg.u.uEps, 0.0008);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  uploadCamera(source: TexImageSource): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.camTex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  /** flags: RGBA8 확률 — R 머리카락, G 몸 피부, B 얼굴 피부, A 사람. */
  uploadSeg(flags: Uint8ClampedArray | null, w: number, h: number): void {
    const gl = this.gl;
    this.hasSeg = flags !== null;
    if (!flags) return;
    gl.bindTexture(gl.TEXTURE_2D, this.segTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, flags);
  }

  draw(opts: DrawOptions): void {
    const gl = this.gl;
    const W = this.width;
    const H = this.height;
    const useGF = this.gf !== null && (opts.refine ?? true) && opts.useSeg && this.hasSeg;
    if (useGF) this.runGuidedFilter();

    // 카메라
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    const cp = this.camProg;
    gl.useProgram(cp.prog);
    this.bindTex(0, this.camTex, cp.u.uCam);
    this.bindTex(1, this.segTex, cp.u.uSeg);
    gl.uniform1f(cp.u.uDebugSeg, opts.debugSeg && this.hasSeg ? 1 : 0);
    gl.uniform1f(cp.u.uUseGF, useGF ? 1 : 0);
    if (this.gf) this.bindTex(5, this.gf.mean, cp.u.uGF);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    // 효과
    const tex: FrameTextures = {
      gl,
      cam: this.camTex,
      seg: this.segTex,
      gf: useGF ? this.gf!.mean : null,
      hasSeg: this.hasSeg && opts.useSeg,
      width: W,
      height: H,
    };
    for (const fx of opts.effects ?? []) {
      fx(tex);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, W, H);
      gl.disable(gl.BLEND);
    }
  }

  /** 방금 그린 화면을 읽는다(같은 프레임 안에서만 유효). 반환: 위가 0행인 RGBA */
  readFrame(): ImageData {
    const gl = this.gl;
    const W = this.width;
    const H = this.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const buf = new Uint8ClampedArray(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    // WebGL은 아래가 0행이므로 뒤집는다
    const out = new Uint8ClampedArray(W * H * 4);
    const row = W * 4;
    for (let y = 0; y < H; y++) out.set(buf.subarray(y * row, (y + 1) * row), (H - 1 - y) * row);
    return new ImageData(out, W, H);
  }

  /** 거울 루프백 측정용 단색 화면. */
  drawSolid(white: boolean): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    const v = white ? 1 : 0;
    gl.clearColor(v, v, v, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private bindTex(unit: number, tex: WebGLTexture, loc: WebGLUniformLocation | null): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  }

  private texture(minFilter: number, magFilter: number): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, minFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, magFilter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private program(vs: string, fs: string, uniforms: string[]): Program {
    return compileProgram(this.gl, vs, fs, uniforms);
  }
}

/** 셰이더 프로그램을 만들고 유니폼 위치를 모은다(효과 모듈도 함께 쓴다). */
export function compileProgram(gl: WebGL2RenderingContext, vs: string, fs: string, uniforms: string[]): Program {
  const compile = (type: number, src: string): WebGLShader => {
    const sh = gl.createShader(type)!;
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      throw new Error(`셰이더 컴파일 실패: ${gl.getShaderInfoLog(sh)}`);
    }
    return sh;
  };
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`셰이더 링크 실패: ${gl.getProgramInfoLog(prog)}`);
  }
  const u: Uniforms = {};
  for (const name of uniforms) u[name] = gl.getUniformLocation(prog, name);
  return { prog, u };
}
