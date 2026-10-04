// Dumps every fundamental domain as subdivided geodesic face meshes in conformal coordinates
// (Poincaré ball / stereographic / metres) so tools/render_polyhedra.py can draw real 3D
// pictures from the SAME face-pairing data the ray walker uses. Run: node tools/dump_domains.mjs
import { createServer } from "vite";
import { writeFileSync } from "node:fs";

const server = await createServer({ configFile: "vite.config.ts", server: { middlewareMode: true }, logLevel: "error" });
const dom = await server.ssrLoadModule("/src/topology/domain.ts");
const sp = await server.ssrLoadModule("/src/geometry/space.ts");
const gal = await server.ssrLoadModule("/src/ui/gallery.ts");
const rend = await server.ssrLoadModule("/src/render/renderer.ts");

const flatHalf = [3.955, 1.66, 3.955]; // synth_open box: bbox half-extents, bottom face 5 cm inside the floor slab (see main.ts domHalf)
const N = 8; // subdivisions per fan triangle edge
const out = { pairColours: rend.Renderer.PAIR_COLOURS, domains: [] };
const conf = (x) => Array.from(sp.toConformal(x));
// point at geodesic fraction t from a to b
const lerp = (k, a, b, t) => { const { u, d } = sp.tangentToward(k, a, b); return sp.geodesic(k, a, u, d * t); };

for (const id of dom.DOMAIN_IDS) {
  const d = dom.makeDomain(id, flatHalf);
  if (!d) continue;
  const faces = d.faces.map((F, f) => {
    const on = d.vertices.map((v, i) => ({ v, i })).filter(({ v }) => Math.abs(F.w[0] * v[0] + F.w[1] * v[1] + F.w[2] * v[2] + F.w[3] * v[3]) < 1e-6);
    const ring = on.map(({ v, i }) => { const l = sp.apply(F.invFrame, v); return { i, ang: Math.atan2(l[2], l[1]) }; }).sort((a, b) => a.ang - b.ang).map((p) => d.vertices[p.i]);
    const c = F.centre;
    const tris = [];
    for (let e = 0; e < ring.length; e++) {
      const a = ring[e], b = ring[(e + 1) % ring.length];
      // grid: s in [0,1] radial from centre, t along edge
      const P = (s, t) => (s < 1e-9 ? conf(c) : conf(lerp(d.kappa, c, lerp(d.kappa, a, b, t), s)));
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const s0 = i / N, s1 = (i + 1) / N, t0 = j / N, t1 = (j + 1) / N;
        tris.push([P(s0, t0), P(s1, t0), P(s1, t1)]);
        if (i > 0) tris.push([P(s0, t0), P(s1, t1), P(s0, t1)]);
      }
    }
    const outline = [];
    for (let e = 0; e < ring.length; e++) { const a = ring[e], b = ring[(e + 1) % ring.length]; for (let j = 0; j <= 2 * N; j++) outline.push(conf(lerp(d.kappa, a, b, j / (2 * N)))); }
    const tw = gal.pairingTwist(d, f);
    return { pair: Math.floor(f / 2), centre: conf(c), dir: Array.from(F.dir), tris, outline, twistDeg: tw.angleDeg, mirrored: tw.mirrored };
  });
  out.domains.push({ id, name: d.name, kappa: d.kappa, faces });
}
writeFileSync("tools/domains.json", JSON.stringify(out));
console.log("wrote", out.domains.length, "domains");
await server.close();
