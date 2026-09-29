// WebGL2 합성기.
// 매 프레임: ① 카메라 영상 → ② 가림 버퍼(팔·손 캡슐) → ③ 옷 부위를 뒤에서 앞 순서로 알파 합성.
// 옷 픽셀은 가림 버퍼와 분할 마스크(머리카락·얼굴·피부)에 따라 지워지고, 원래 옷의 음영을 입는다.

import type { Capsule } from './occluders.ts';
import { PART, type GarmentAsset, type PartMesh } from './garment.ts';
import { CAMERA_FS, FULLSCREEN_VS, GARMENT_FS, GARMENT_VS, GF_MEAN_FS, GF_STATS_FS, OCC_FS, OCC_VS } from './shaders.ts';

type Uniforms = Record<string, WebGLUniformLocation | null>;

interface Program {
  prog: WebGLProgram;
  u: Uniforms;
}

interface PartGPU {
  mesh: PartMesh;
  vao: WebGLVertexArrayObject;
  dstBuf: WebGLBuffer;
  count: number;
}

export interface GarmentGPU {
  asset: GarmentAsset;
  tex: WebGLTexture;
  label: WebGLTexture;
  parts: Map<PartMesh, PartGPU>;
}

export interface DrawLayer {
  gpu: GarmentGPU;
  order: PartMesh[];
  alpha: number;
}

export interface DrawOptions {
  layers: DrawLayer[];
  capsules: Capsule[];
  shade: number;
  useSeg: boolean;
  debugSeg: boolean;
  debugOcc: boolean;
  /** 머리카락·피부 경계를 원본 해상도로 정밀화(가이디드 필터) */
  refine?: boolean;
  /** 0 정상, 1 라벨 색, 2 통과 픽셀 빨강, 3 가림 버퍼 */
  debugGarment?: number;
}

const OCC_SCALE = 0.5;
/** 가이디드 필터 저해상도 배율(1/4)과 그에 맞는 카메라 밉맵 단계 */
const GF_DOWN = 4;
const GF_LOD = 2;

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

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private width = 0;
  private height = 0;
  private readonly camProg: Program;
  private readonly occProg: Program;
  private readonly garmentProg: Program;
  private readonly camTex: WebGLTexture;
  private readonly segTex: WebGLTexture;
  private readonly occTex: WebGLTexture;
  private readonly occFbo: WebGLFramebuffer;
  private occW = 1;
  private occH = 1;
  private readonly emptyVao: WebGLVertexArrayObject;
  private readonly occVao: WebGLVertexArrayObject;
  private readonly occBuf: WebGLBuffer;
  private occData = new Float32Array(0);
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
    this.camProg = this.program(FULLSCREEN_VS, CAMERA_FS, ['uCam', 'uSeg', 'uOcc', 'uGF', 'uUseGF', 'uDebugSeg', 'uDebugOcc']);
    this.occProg = this.program(OCC_VS, OCC_FS, ['uSize', 'uFeather']);
    this.garmentProg = this.program(GARMENT_VS, GARMENT_FS, [
      'uSize', 'uTexSize', 'uTex', 'uLabel', 'uCam', 'uSeg', 'uOcc', 'uPart', 'uIsTorso', 'uIsInner', 'uAlpha', 'uShade', 'uUseSeg', 'uDebug', 'uGF', 'uUseGF',
    ]);
    this.camTex = this.texture(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR);
    this.segTex = this.texture(gl.LINEAR, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, this.segTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
    this.occTex = this.texture(gl.LINEAR, gl.LINEAR);
    this.occFbo = gl.createFramebuffer()!;
    this.emptyVao = gl.createVertexArray()!;
    this.occVao = gl.createVertexArray()!;
    this.occBuf = gl.createBuffer()!;
    this.setupOccVao();
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
    this.occW = Math.max(1, Math.round(width * OCC_SCALE));
    this.occH = Math.max(1, Math.round(height * OCC_SCALE));
    gl.bindTexture(gl.TEXTURE_2D, this.occTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.occW, this.occH, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.occFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.occTex, 0);
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

  /** flags: RGBA8 확률 — R 머리카락·얼굴, G 몸 피부, B 옷, A 사람. */
  uploadSeg(flags: Uint8ClampedArray | null, w: number, h: number): void {
    const gl = this.gl;
    this.hasSeg = flags !== null;
    if (!flags) return;
    gl.bindTexture(gl.TEXTURE_2D, this.segTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, flags);
  }

  createGarment(asset: GarmentAsset): GarmentGPU {
    const gl = this.gl;
    const tex = this.texture(gl.LINEAR_MIPMAP_LINEAR, gl.LINEAR);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, asset.image);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

    // 라벨: R = 부위 번호, G = 몸판을 2px 넓힌 영역(소매와의 경계 틈 방지)
    const { width: w, height: h, labels } = asset;
    const rg = new Uint8Array(w * h * 2);
    const R = 2;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        rg[i * 2] = labels[i];
        if (labels[i] === PART.none) continue;
        let near = false;
        for (let dy = -R; dy <= R && !near; dy++) {
          const yy = y + dy;
          if (yy < 0 || yy >= h) continue;
          for (let dx = -R; dx <= R; dx++) {
            const xx = x + dx;
            if (xx >= 0 && xx < w && labels[yy * w + xx] === PART.torso) {
              near = true;
              break;
            }
          }
        }
        rg[i * 2 + 1] = near ? 255 : 0;
      }
    }
    const label = this.texture(gl.NEAREST, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, w, h, 0, gl.RG, gl.UNSIGNED_BYTE, rg);

    const parts = new Map<PartMesh, PartGPU>();
    const prog = this.garmentProg.prog;
    const locSrc = gl.getAttribLocation(prog, 'aSrc');
    const locDst = gl.getAttribLocation(prog, 'aDst');
    for (const mesh of asset.meshes) {
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      const srcBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, srcBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.src, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(locSrc);
      gl.vertexAttribPointer(locSrc, 2, gl.FLOAT, false, 0, 0);
      const dstBuf = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, dstBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.dst.byteLength, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(locDst);
      gl.vertexAttribPointer(locDst, 2, gl.FLOAT, false, 0, 0);
      const ib = gl.createBuffer()!;
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      parts.set(mesh, { mesh, vao, dstBuf, count: mesh.indices.length });
    }
    return { asset, tex, label, parts };
  }

  deleteGarment(g: GarmentGPU): void {
    const gl = this.gl;
    gl.deleteTexture(g.tex);
    gl.deleteTexture(g.label);
    for (const p of g.parts.values()) {
      gl.deleteVertexArray(p.vao);
      gl.deleteBuffer(p.dstBuf);
    }
  }

  draw(opts: DrawOptions): void {
    const gl = this.gl;
    const W = this.width;
    const H = this.height;

    // ② 가림 버퍼
    this.drawOccluders(opts.capsules);

    // 머리카락·피부 경계 정밀화
    const useGF = this.gf !== null && (opts.refine ?? true) && opts.useSeg && this.hasSeg;
    if (useGF) this.runGuidedFilter();

    // ① 카메라
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, W, H);
    gl.disable(gl.BLEND);
    gl.useProgram(this.camProg.prog);
    this.bindTex(0, this.camTex, this.camProg.u.uCam);
    this.bindTex(1, this.segTex, this.camProg.u.uSeg);
    this.bindTex(2, this.occTex, this.camProg.u.uOcc);
    gl.uniform1f(this.camProg.u.uDebugSeg, opts.debugSeg && this.hasSeg ? 1 : 0);
    gl.uniform1f(this.camProg.u.uDebugOcc, opts.debugOcc ? 1 : 0);
    gl.uniform1f(this.camProg.u.uUseGF, useGF ? 1 : 0);
    if (this.gf) this.bindTex(5, this.gf.mean, this.camProg.u.uGF);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ③ 옷
    if (opts.layers.length === 0) return;
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const p = this.garmentProg;
    gl.useProgram(p.prog);
    gl.uniform2f(p.u.uSize, W, H);
    gl.uniform1f(p.u.uShade, opts.shade);
    gl.uniform1f(p.u.uUseSeg, opts.useSeg && this.hasSeg ? 1 : 0);
    gl.uniform1f(p.u.uDebug, opts.debugGarment ?? 0);
    this.bindTex(2, this.camTex, p.u.uCam);
    this.bindTex(3, this.segTex, p.u.uSeg);
    this.bindTex(4, this.occTex, p.u.uOcc);
    gl.uniform1f(p.u.uUseGF, useGF ? 1 : 0);
    if (this.gf) this.bindTex(5, this.gf.mean, p.u.uGF);
    for (const layer of opts.layers) {
      const g = layer.gpu;
      gl.uniform2f(p.u.uTexSize, g.asset.width, g.asset.height);
      gl.uniform1f(p.u.uAlpha, layer.alpha);
      this.bindTex(0, g.tex, p.u.uTex);
      this.bindTex(1, g.label, p.u.uLabel);
      for (const mesh of layer.order) {
        const part = g.parts.get(mesh);
        if (!part) continue;
        gl.uniform1f(p.u.uPart, mesh.partId);
        gl.uniform1f(p.u.uIsTorso, mesh.partId === PART.torso ? 1 : 0);
        gl.uniform1f(p.u.uIsInner, mesh.partId === PART.neckInner ? 1 : 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, part.dstBuf);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, mesh.dst);
        gl.bindVertexArray(part.vao);
        gl.drawElements(gl.TRIANGLES, part.count, gl.UNSIGNED_SHORT, 0);
      }
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  /** 거울 루프백 측정용 단색 화면. */
  drawSolid(white: boolean): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    const c = white ? 1 : 0;
    gl.clearColor(c, c, c, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  private drawOccluders(capsules: Capsule[]): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.occFbo);
    gl.viewport(0, 0, this.occW, this.occH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (capsules.length === 0) return;

    // 캡슐마다 사각형(삼각형 2개) = 정점 6개 × [pos2, a2, b2, r1, ch3] = 10 float
    const FLOATS = 10;
    const need = capsules.length * 6 * FLOATS;
    if (this.occData.length < need) this.occData = new Float32Array(need * 2);
    const d = this.occData;
    let o = 0;
    const pad = 3;
    for (const c of capsules) {
      const r = c.r + pad;
      let dx = c.b.x - c.a.x;
      let dy = c.b.y - c.a.y;
      const l = Math.hypot(dx, dy);
      if (l < 1e-3) {
        dx = 1;
        dy = 0;
      } else {
        dx /= l;
        dy /= l;
      }
      const nx = -dy;
      const ny = dx;
      const ax = c.a.x - dx * r;
      const ay = c.a.y - dy * r;
      const bx = c.b.x + dx * r;
      const by = c.b.y + dy * r;
      const corners = [
        [ax + nx * r, ay + ny * r],
        [bx + nx * r, by + ny * r],
        [bx - nx * r, by - ny * r],
        [ax - nx * r, ay - ny * r],
      ];
      for (const k of [0, 1, 2, 0, 2, 3]) {
        d[o++] = corners[k][0];
        d[o++] = corners[k][1];
        d[o++] = c.a.x;
        d[o++] = c.a.y;
        d[o++] = c.b.x;
        d[o++] = c.b.y;
        d[o++] = c.r;
        d[o++] = c.ch[0];
        d[o++] = c.ch[1];
        d[o++] = c.ch[2];
      }
    }
    gl.useProgram(this.occProg.prog);
    gl.uniform2f(this.occProg.u.uSize, this.width, this.height);
    gl.uniform1f(this.occProg.u.uFeather, Math.max(2, this.width / 400));
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.MAX);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(this.occVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.occBuf);
    gl.bufferData(gl.ARRAY_BUFFER, d.subarray(0, o), gl.DYNAMIC_DRAW);
    gl.drawArrays(gl.TRIANGLES, 0, o / FLOATS);
    gl.bindVertexArray(null);
    gl.blendEquation(gl.FUNC_ADD);
    gl.disable(gl.BLEND);
  }

  private setupOccVao(): void {
    const gl = this.gl;
    const prog = this.occProg.prog;
    gl.bindVertexArray(this.occVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.occBuf);
    const stride = 10 * 4;
    const attrs: [string, number, number][] = [
      ['aPos', 2, 0],
      ['aA', 2, 2],
      ['aB', 2, 4],
      ['aR', 1, 6],
      ['aCh', 3, 7],
    ];
    for (const [name, size, off] of attrs) {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) continue;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, off * 4);
    }
    gl.bindVertexArray(null);
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
    const gl = this.gl;
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
}
