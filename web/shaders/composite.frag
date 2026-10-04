#version 300 es
// Upscale the offscreen walk target to the canvas. Phase 1: plain bilinear + clamp.
// Tone mapping / bloom land here in Phase 8.
precision highp float;
uniform sampler2D uColor;
uniform vec2 uResolution;
out vec4 oColor;
void main() {
  vec2 uv = gl_FragCoord.xy / uResolution;
  vec3 c = texture(uColor, uv).rgb;
  oColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
