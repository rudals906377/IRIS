// 생성된 머리 스티커를 격자로 휘어 그리는 작은 WebGL2 그리기 도구.
// 결과 캔버스를 2D 덧그림 캔버스에 drawImage 로 옮겨 쓴다(본 화면 WebGL 과 상태를 섞지 않게 따로 둔다).

const VS = `#version 300 es
in vec2 aPos; in vec2 aUv; in float aA;
uniform vec2 uSize;
out vec2 vUv; out float vA;
void main(){ vUv = aUv; vA = aA; gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0); }`;

const FS = `#version 300 es
precision mediump float;
in vec2 vUv; in float vA;
uniform sampler2D uTex; uniform float uAlpha; uniform int uMode;
out vec4 o;
void main(){
  if (uMode == 0) { o = texture(uTex, vUv) * uAlpha; }
  else { o = vec4(0.0, 0.0, 0.0, vA * uAlpha); }
}`;

export class StickerGL {
  readonly canvas: HTMLCanvasElement;
  /** 텍스처가 올라가 있는지(새 스티커면 false 로 두어 다시 올리게) */
  ready = false;
  private readonly gl: WebGL2RenderingContext;
  private readonly prog: WebGLProgram;
  private readonly tex: WebGLTexture;
  private readonly posBuf: WebGLBuffer;
  private readonly uvBuf: WebGLBuffer;
  private readonly aBuf: WebGLBuffer;
  private readonly idxBuf: WebGLBuffer;
  private readonly loc: { pos: number; uv: number; a: number; size: WebGLUniformLocation; alpha: WebGLUniformLocation; mode: WebGLUniformLocation };

  constructor() {
    this.canvas = document.createElement('canvas');
    const gl = this.canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: true });
    if (!gl) throw new Error('WebGL2 를 쓸 수 없어요');
    this.gl = gl;
    const sh = (type: number, src: string): WebGLShader => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'link');
    this.prog = p;
    this.loc = {
      pos: gl.getAttribLocation(p, 'aPos'),
      uv: gl.getAttribLocation(p, 'aUv'),
      a: gl.getAttribLocation(p, 'aA'),
      size: gl.getUniformLocation(p, 'uSize')!,
      alpha: gl.getUniformLocation(p, 'uAlpha')!,
      mode: gl.getUniformLocation(p, 'uMode')!,
    };
    this.tex = gl.createTexture()!;
    this.posBuf = gl.createBuffer()!;
    this.uvBuf = gl.createBuffer()!;
    this.aBuf = gl.createBuffer()!;
    this.idxBuf = gl.createBuffer()!;
  }

  setImage(img: ImageBitmap | HTMLCanvasElement): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.ready = true;
  }

  private attr(buf: WebGLBuffer, loc: number, data: Float32Array, size: number): void {
    const gl = this.gl;
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  }

  /**
   * w×h 캔버스에 휜 스티커를 그리고, cut 이 있으면 그 자리를 부드럽게 지운다.
   * xy: 격자점 화면 좌표, uv: 텍스처 좌표, alpha: 전체 불투명도, cutAlpha: 지우는 세기
   */
  draw(w: number, h: number, mesh: { xy: Float32Array; uv: Float32Array; index: Uint16Array }, alpha: number, cut: { xy: Float32Array; a: Float32Array; index: Uint16Array } | null, cutAlpha: number): HTMLCanvasElement {
    const gl = this.gl;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog);
    gl.uniform2f(this.loc.size, w, h);
    gl.enable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    // 1) 스티커(미리 곱한 알파)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform1i(this.loc.mode, 0);
    gl.uniform1f(this.loc.alpha, alpha);
    this.attr(this.posBuf, this.loc.pos, mesh.xy, 2);
    this.attr(this.uvBuf, this.loc.uv, mesh.uv, 2);
    if (this.loc.a >= 0) {
      gl.disableVertexAttribArray(this.loc.a);
      gl.vertexAttrib1f(this.loc.a, 1);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.index, gl.DYNAMIC_DRAW);
    gl.drawElements(gl.TRIANGLES, mesh.index.length, gl.UNSIGNED_SHORT, 0);
    // 2) 얼굴 자리 지우기: 결과 = 기존 × (1 − a)
    if (cut && cutAlpha > 0.001) {
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform1i(this.loc.mode, 1);
      gl.uniform1f(this.loc.alpha, cutAlpha);
      this.attr(this.posBuf, this.loc.pos, cut.xy, 2);
      if (this.loc.uv >= 0) {
        gl.disableVertexAttribArray(this.loc.uv);
        gl.vertexAttrib2f(this.loc.uv, 0, 0);
      }
      this.attr(this.aBuf, this.loc.a, cut.a, 1);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, cut.index, gl.DYNAMIC_DRAW);
      gl.drawElements(gl.TRIANGLES, cut.index.length, gl.UNSIGNED_SHORT, 0);
    }
    gl.disable(gl.BLEND);
    return this.canvas;
  }
}
