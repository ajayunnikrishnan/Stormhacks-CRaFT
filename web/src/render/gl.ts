/** Thin WebGL2 helpers: context, shader programs, data textures with row wrapping, FBOs. */

export class GLError extends Error {}

export function createContext(canvas: HTMLCanvasElement): WebGL2RenderingContext {
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
  if (!gl) throw new GLError("WebGL2 is required");
  // Needed to render into RGBA16F/RGBA32F (SV pre-pass + offscreen walk target).
  if (!gl.getExtension("EXT_color_buffer_float")) throw new GLError("EXT_color_buffer_float is required");
  return gl;
}

export function compileShader(gl: WebGL2RenderingContext, type: number, src: string, label: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    const numbered = src.split("\n").map((l, i) => `${String(i + 1).padStart(4)}: ${l}`).join("\n");
    throw new GLError(`shader ${label} failed:\n${log}\n${numbered}`);
  }
  return sh;
}

export class Program {
  readonly prog: WebGLProgram;
  private uniforms = new Map<string, WebGLUniformLocation | null>();
  constructor(readonly gl: WebGL2RenderingContext, vs: string, fs: string, label = "program") {
    const p = gl.createProgram()!;
    gl.attachShader(p, compileShader(gl, gl.VERTEX_SHADER, vs, label + ".vert"));
    gl.attachShader(p, compileShader(gl, gl.FRAGMENT_SHADER, fs, label + ".frag"));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new GLError(`link ${label}: ${gl.getProgramInfoLog(p)}`);
    this.prog = p;
  }
  use() { this.gl.useProgram(this.prog); }
  loc(name: string): WebGLUniformLocation | null {
    if (!this.uniforms.has(name)) this.uniforms.set(name, this.gl.getUniformLocation(this.prog, name));
    return this.uniforms.get(name)!;
  }
  u1i(n: string, v: number) { this.gl.uniform1i(this.loc(n), v); }
  u1f(n: string, v: number) { this.gl.uniform1f(this.loc(n), v); }
  u2f(n: string, a: number, b: number) { this.gl.uniform2f(this.loc(n), a, b); }
  u3f(n: string, a: number, b: number, c: number) { this.gl.uniform3f(this.loc(n), a, b, c); }
  u3fv(n: string, v: Float32Array | number[]) { this.gl.uniform3fv(this.loc(n), v); }
  u4fv(n: string, v: Float32Array | number[]) { this.gl.uniform4fv(this.loc(n), v); }
  umat4(n: string, v: Float32Array | number[]) { this.gl.uniformMatrix4fv(this.loc(n), false, v); }
  /** Bind texture to unit and set sampler uniform. */
  tex(n: string, unit: number, tex: WebGLTexture) {
    this.gl.activeTexture(this.gl.TEXTURE0 + unit);
    this.gl.bindTexture(this.gl.TEXTURE_2D, tex);
    this.gl.uniform1i(this.loc(n), unit);
  }
}

/** Width (texels) of all data textures; indices map to (i % W, i / W). Power of two so the shader can use shifts. */
export const TEX_W = 4096;
export const TEX_W_LOG2 = 12;

export interface DataTexture { tex: WebGLTexture; width: number; height: number; count: number }

/**
 * Upload a flat array as a 2D data texture with row wrapping.
 * channels: 1..4; internalFormat/format/type must match WebGL2's sized-format table.
 */
export function dataTexture(
  gl: WebGL2RenderingContext,
  data: ArrayBufferView,
  count: number,
  channels: number,
  internalFormat: number,
  format: number,
  type: number,
  label = "tex",
): DataTexture {
  const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (TEX_W > maxSize) throw new GLError(`MAX_TEXTURE_SIZE ${maxSize} < ${TEX_W}`);
  const height = Math.max(1, Math.ceil(count / TEX_W));
  if (height > maxSize) throw new GLError(`${label}: ${count} texels need ${height} rows > MAX_TEXTURE_SIZE`);
  const need = TEX_W * height * channels;
  let padded: ArrayBufferView = data;
  const len = (data as unknown as { length: number }).length;
  if (len < need) {
    const Ctor = data.constructor as new (n: number) => ArrayBufferView & { set(a: ArrayLike<number>): void };
    const buf = new Ctor(need);
    buf.set(data as unknown as ArrayLike<number>);
    padded = buf;
  }
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, TEX_W, height, 0, format, type, padded as ArrayBufferView);
  const err = gl.getError();
  if (err !== gl.NO_ERROR) throw new GLError(`${label}: texImage2D error 0x${err.toString(16)}`);
  return { tex, width: TEX_W, height, count };
}

export interface RenderTarget { fbo: WebGLFramebuffer; tex: WebGLTexture; width: number; height: number; internalFormat: number }

export function renderTarget(gl: WebGL2RenderingContext, width: number, height: number, internalFormat: number, filter: number = gl.NEAREST): RenderTarget {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, width, height);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) throw new GLError(`framebuffer incomplete 0x${status.toString(16)}`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, tex, width, height, internalFormat };
}

export function destroyTarget(gl: WebGL2RenderingContext, rt: RenderTarget) {
  gl.deleteFramebuffer(rt.fbo);
  gl.deleteTexture(rt.tex);
}

/** A VAO drawing one fullscreen triangle (no vertex buffer; positions from gl_VertexID). */
export class FullscreenQuad {
  private vao: WebGLVertexArrayObject;
  constructor(private gl: WebGL2RenderingContext) {
    this.vao = gl.createVertexArray()!;
  }
  draw() {
    this.gl.bindVertexArray(this.vao);
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
    this.gl.bindVertexArray(null);
  }
}

export const FULLSCREEN_VS = `#version 300 es
void main() {
  // one big triangle covering clip space
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export interface MrtTarget { fbo: WebGLFramebuffer; texs: WebGLTexture[]; width: number; height: number }

/** Framebuffer with n colour attachments of the given sized formats (needs EXT_color_buffer_float). */
export function mrtTarget(gl: WebGL2RenderingContext, width: number, height: number, formats: number[]): MrtTarget {
  const fbo = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const texs: WebGLTexture[] = [];
  formats.forEach((fmt, i) => {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, fmt, width, height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, tex, 0);
    texs.push(tex);
  });
  gl.drawBuffers(formats.map((_, i) => gl.COLOR_ATTACHMENT0 + i));
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) throw new GLError(`MRT framebuffer incomplete 0x${status.toString(16)}`);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, texs, width, height };
}
export function destroyMrt(gl: WebGL2RenderingContext, t: MrtTarget) { gl.deleteFramebuffer(t.fbo); t.texs.forEach((x) => gl.deleteTexture(x)); }
