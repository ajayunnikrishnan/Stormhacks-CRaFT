// Shared GLSL ES 3.00 helpers for data-texture access with row wrapping.
// All data textures are TEX_W (= 4096 = 1 << TEX_W_LOG2) texels wide.
#define TEX_W_LOG2 12
#define TEX_W_MASK 4095
#define INT_MAX_ID 0x7FFFFFFF
// every includer uses both sampler kinds; declare the precisions here so no shader forgets
precision highp sampler2D;
precision highp usampler2D;

ivec2 texCoord(int i) { return ivec2(i & TEX_W_MASK, i >> TEX_W_LOG2); }

vec4 fetch4(sampler2D t, int i) { return texelFetch(t, texCoord(i), 0); }
uint fetchU(usampler2D t, int i) { return texelFetch(t, texCoord(i), 0).r; }
