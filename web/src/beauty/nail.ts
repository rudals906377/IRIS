// 네일 합성: 손톱마다 작은 사각형을 그리고, 조각 셰이더에서 손톱 모양(끝이 둥근 사각형)을 만든 뒤
// 원래 손톱의 밝기·굴곡을 살린 색 + 광택 줄 + 무늬(프렌치·그라데이션·글리터·도트)와 마감(크롬·젤리·캣아이·오로라)을 입힌다.
// 길이를 늘리면 손가락 끝을 넘어 아몬드 모양으로 연장한다(참고 사진에서 가장 많던 모양).

import type { FrameTextures } from '../engine/renderer.ts';
import { compileProgram } from '../engine/renderer.ts';
import type { RGB } from './makeup.ts';
import type { NailQuad } from './nail-place.ts';

export type NailStyle = 'solid' | 'french' | 'gradient' | 'glitter' | 'dots' | 'chrome' | 'jelly' | 'cateye' | 'aurora';
const STYLE_ID: Record<NailStyle, number> = { solid: 0, french: 1, gradient: 2, glitter: 3, dots: 4, chrome: 5, jelly: 6, cateye: 7, aurora: 8 };

export interface NailLook {
  color: RGB;
  /** 무늬에 쓰는 둘째 색(프렌치 끝, 그라데이션 끝, 도트) */
  color2: RGB;
  style: NailStyle;
  /** 0~1 */
  amount: number;
  /** 0 자연 길이 ~ 1 손톱 길이의 1.2배만큼 아몬드 모양으로 연장 */
  length: number;
}

const VS = /* glsl */ `#version 300 es
in vec2 aPos;
in vec2 aLocal;   // x: 폭 방향 −1~1, y: 길이 방향 0(뿌리)~1(끝) — 가장자리 여유 포함
in float aVis;
in float aSeed;
in float aAspect;  // 손톱 폭 / 길이
in vec2 aRef;      // 손가락 피부 기준점(카메라 uv)
uniform vec2 uSize;
out vec2 vLocal;
out float vVis;
out float vSeed;
out float vAspect;
out vec2 vRef;
void main() {
  vLocal = aLocal;
  vAspect = aAspect;
  vRef = aRef;
  vVis = aVis;
  vSeed = aSeed;
  gl_Position = vec4(aPos.x / uSize.x * 2.0 - 1.0, 1.0 - aPos.y / uSize.y * 2.0, 0.0, 1.0);
}`;

const FS = /* glsl */ `#version 300 es
precision highp float;
uniform sampler2D uCam;
uniform sampler2D uSeg;
uniform sampler2D uGF;
uniform float uUseGF;
uniform vec2 uSize;
uniform vec3 uColor;
uniform vec3 uColor2;
uniform float uStyle;
uniform float uAmount;
uniform float uExt;     // 연장 길이(손톱 길이 단위)
in vec2 vLocal;
in float vVis;
in float vSeed;
in float vAspect;
in vec2 vRef;
out vec4 o;
const vec3 W = vec3(0.299, 0.587, 0.114);
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  vec2 camUv = vec2(gl_FragCoord.x / uSize.x, 1.0 - gl_FragCoord.y / uSize.y);
  vec3 c = textureLod(uCam, camUv, 0.0).rgb;
  // 손톱 모양: 길이 1 기준 좌표에서 둥근 사각형. 뿌리(큐티클) 쪽은 폭 전체로 둥근 U자, 끝은 덜 둥글게
  float x = vLocal.x;
  float y = vLocal.y;
  float hx = vAspect * 0.5;
  vec2 q = vec2(abs(x) * hx, y - 0.5);
  float r = y < 0.5 ? hx * 0.95 : hx * 0.55;
  vec2 bq = vec2(hx, 0.5) - r;
  vec2 dq = q - bq;
  float sd = length(max(dq, 0.0)) + min(max(dq.x, dq.y), 0.0) - r;   // 음수 = 안쪽
  // 연장: 가운데부터 끝까지 폭이 줄어드는 아몬드 윤곽
  float Ltot = 1.0 + uExt;
  if (uExt > 0.01 && y >= 0.5) {
    float t = clamp((y - 0.5) / (Ltot - 0.5), 0.0, 1.0);
    float hwy = hx * sqrt(max(0.0, 1.0 - t * t)) * mix(1.0, 0.8, t);
    sd = max(abs(x) * hx - hwy, y - Ltot);
  }
  float aa = fwidth(sd) * 1.2 + 1e-4;
  // 손가락 끝을 넘어간 연장 부분(아래는 손가락이 없으므로 피부·밝기 판단을 쓰지 않는다)
  float ext = uExt > 0.01 ? smoothstep(0.85, 1.05, y) : 0.0;
  // 영상으로 경계 다듬기: 손톱은 같은 손가락 피부보다 밝다(기준점은 마지막 마디 관절 뒤 피부).
  // 추정 손톱의 안쪽은 늘 칠하고, 가장자리 근처(안팎 0.35·반폭)는 밝기로 손톱인지 판단해 넓히거나 줄인다.
  float Lref = dot(textureLod(uCam, vRef, 2.0).rgb, W);
  float Ls = dot(textureLod(uCam, camUv, 1.0).rgb, W);
  float bright = smoothstep(0.0, 0.07, Ls - Lref);
  float core = smoothstep(0.0, -0.3 * hx, sd);
  float reach = smoothstep(0.35 * hx + aa, 0.35 * hx - aa, sd);
  float shape = reach * max(core, bright * smoothstep(0.35 * hx, 0.0, sd));
  shape = mix(shape, smoothstep(aa, -aa, sd), ext);
  // 실제로 피부(손)가 보이는 곳에만: 물건 뒤로 숨은 손가락 끝은 칠하지 않는다
  float skin = texture(uSeg, camUv).g;
  if (uUseGF > 0.5) {
    vec4 gf = textureLod(uGF, vec2(camUv.x, 1.0 - camUv.y), 0.0);
    skin = clamp(gf.z * dot(c, W) + gf.w, 0.0, 1.0);
  }
  float m = shape * vVis * uAmount * mix(smoothstep(0.15, 0.5, skin), 1.0, ext);
  if (m < 0.004) discard;

  // 무늬별 색
  vec3 col = uColor;
  int st = int(uStyle + 0.5);
  if (st == 1) {
    // 프렌치: 끝부분 초승달 모양
    float line = 0.72 + 0.12 * x * x;
    col = mix(uColor, uColor2, smoothstep(line - 0.02, line + 0.02, y));
  } else if (st == 2) {
    col = mix(uColor, uColor2, smoothstep(0.15, 0.95, y));
  } else if (st == 4) {
    vec2 g = vec2(x * 1.6, y * 3.2);
    vec2 cell = fract(g) - 0.5;
    col = mix(uColor, uColor2, smoothstep(0.22, 0.17, length(cell)));
  }

  // 원래 밝기를 살린 색: 주변 평균 밝기로 조명 세기를, 평균 대비 밝기로 손톱 결·반사를 살린다
  float L = dot(c, W);
  // 연장 부분은 뒤가 배경이므로 손가락 피부 밝기를 조명으로 쓴다
  float Lavg = mix(dot(textureLod(uCam, camUv, 3.0).rgb, W), Lref * 1.1, ext);
  float rel = mix(clamp(L / max(Lavg, 0.04), 0.6, 1.4), 1.0, ext);
  vec3 painted = col * (0.3 + 0.95 * Lavg) * rel;
  // 광택: 손톱 가운데를 따라가는 부드러운 빛 줄 + 원래 반사광
  float streak = exp(-pow((x + 0.35) / 0.18, 2.0)) * smoothstep(0.15, 0.5, y) * smoothstep(0.98, 0.7, y);
  painted += vec3(0.22 * streak);
  painted += vec3(smoothstep(0.7, 0.95, L) * 0.25);
  if (st == 3) {
    // 글리터: 작은 반짝이 점(손톱마다 다른 배치)
    vec2 q = floor(vec2(x * 9.0, y * 14.0) + vSeed * 17.0);
    float h = hash(q);
    painted = mix(painted, uColor2 * 1.2 + 0.25, step(0.82, h) * 0.85);
  } else if (st == 5) {
    // 크롬: 주변을 비추는 금속. 넓은 명암 띠 + 날카로운 반사
    // (띠 대비를 낮추고 원래 손톱 명암을 섞어 줄무늬처럼 보이지 않게)
    float env = 0.5 + 0.5 * cos((x * 0.8 + (y - 0.5) * 0.5) * 2.6);
    float spec = exp(-pow((x - 0.35) / 0.14, 2.0)) * smoothstep(0.1, 0.4, y) * smoothstep(1.1, 0.7, y);
    painted = col * (0.45 + 0.75 * env) * (0.45 + 0.75 * Lavg) * mix(1.0, rel, 0.5) + vec3(0.6) * spec * 0.7;
  } else if (st == 6) {
    // 젤리: 반투명(원래 손톱·피부가 비침) + 더 강한 광택
    painted = mix(c, painted, 0.55) + vec3(0.3 * streak);
  } else if (st == 7) {
    // 캣아이: 어두운 바탕에 비스듬한 빛 띠와 자잘한 반짝임
    float d = (y - 0.55) - 0.35 * x;
    float band = exp(-pow(d / 0.13, 2.0));
    float sp = step(0.8, hash(floor(vec2(x * 14.0, y * 20.0) + vSeed * 9.0))) * band;
    painted = col * 0.35 * (0.3 + 0.95 * Lavg) * rel + uColor2 * band * 0.9 + vec3(sp * 0.4);
  } else if (st == 8) {
    // 오로라: 보는 자리에 따라 무지개빛이 도는 막
    float tt = x * 0.25 + y * 0.45 + vSeed * 0.13;
    vec3 ir = 0.5 + 0.5 * cos(6.2832 * (tt + vec3(0.0, 0.33, 0.67)));
    painted = mix(painted, painted * 0.55 + ir * 0.45 * (0.4 + 0.8 * Lavg), 0.65) + vec3(0.3 * streak);
  }
  o = vec4(clamp(painted, 0.0, 1.0), m);
}`;

export class NailRenderer {
  private readonly gl: WebGL2RenderingContext;
  private readonly prog;
  private readonly vao: WebGLVertexArrayObject;
  private readonly vbo: WebGLBuffer;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.prog = compileProgram(gl, VS, FS, ['uCam', 'uSeg', 'uGF', 'uUseGF', 'uSize', 'uColor', 'uColor2', 'uStyle', 'uAmount', 'uExt']);
    this.vao = gl.createVertexArray()!;
    this.vbo = gl.createBuffer()!;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    const attr = (name: string, size: number, offset: number): void => {
      const loc = gl.getAttribLocation(this.prog.prog, name);
      if (loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 36, offset);
    };
    attr('aPos', 2, 0);
    attr('aLocal', 2, 8);
    attr('aVis', 1, 16);
    attr('aSeed', 1, 20);
    attr('aAspect', 1, 24);
    attr('aRef', 2, 28);
    gl.bindVertexArray(null);
  }

  draw(nails: NailQuad[], look: NailLook, t: FrameTextures): void {
    if (nails.length === 0) return;
    const gl = this.gl;
    // 손톱마다 두 삼각형(6정점). 가장자리 흐림 여유로 10% 크게
    const data = new Float32Array(nails.length * 6 * 9);
    const ext = Math.max(0, Math.min(1, look.length)) * 1.2;
    let k = 0;
    nails.forEach((n, idx) => {
      const px = -n.dir.y;
      const py = n.dir.x;
      // 경계 다듬기용 여유: 폭 1.4배, 길이 앞뒤로 더
      const hw = (n.width / 2) * 1.4;
      const corner = (sx: number, sy: number): number[] => {
        // sy: 0 뿌리 ~ 1 끝, 여유 포함 −0.2 ~ 1.2(+연장)
        const along = (sy - 0.5) * n.len;
        return [n.c.x + n.dir.x * along + px * sx * hw, n.c.y + n.dir.y * along + py * sx * hw, sx * 1.4, sy, n.vis, idx + 1, n.width / n.len, n.ref.x / t.width, n.ref.y / t.height];
      };
      const a = corner(-1, -0.2);
      const b = corner(1, -0.2);
      const c = corner(-1, 1.2 + ext);
      const d = corner(1, 1.2 + ext);
      for (const v of [a, b, c, b, d, c]) {
        data.set(v, k);
        k += 9;
      }
    });
    const u = this.prog.u;
    gl.useProgram(this.prog.prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, t.cam);
    gl.uniform1i(u.uCam, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, t.seg);
    gl.uniform1i(u.uSeg, 1);
    if (t.gf) {
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, t.gf);
      gl.uniform1i(u.uGF, 2);
    }
    gl.uniform1f(u.uUseGF, t.gf ? 1 : 0);
    gl.uniform2f(u.uSize, t.width, t.height);
    gl.uniform3f(u.uColor, ...look.color);
    gl.uniform3f(u.uColor2, ...look.color2);
    gl.uniform1f(u.uStyle, STYLE_ID[look.style]);
    gl.uniform1f(u.uAmount, look.amount);
    gl.uniform1f(u.uExt, ext);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, nails.length * 6);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(null);
  }
}
