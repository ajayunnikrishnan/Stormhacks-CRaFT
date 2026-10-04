import { createContext } from "./render/gl";
import { loadScene, type RepoCamera } from "./foam/scene";
import { Renderer, type RenderOptions, type IsoCameraState } from "./render/renderer";
import { FlyCamera, repoCameraState } from "./game/camera";
import { IsoCamera } from "./game/isocamera";
import { curvatureParams } from "./foam/curved";
import { DOMAIN_IDS, makeDomain, type Domain, type DomainId } from "./topology/domain";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("gl");
const hud = $("hud"), status = $("status");
const scaleEl = $<HTMLInputElement>("scale"), scaleV = $("scaleV");
const curvEl = $<HTMLInputElement>("curv"), curvV = $("curvV");
const camsEl = $<HTMLSelectElement>("cams");
const topoEl = $<HTMLSelectElement>("topo");
const modeEl = $<HTMLInputElement>("mode"), nearCullEl = $<HTMLInputElement>("nearcull"), repoPixEl = $<HTMLInputElement>("repopix"), floorEl = $<HTMLInputElement>("floor");

const params = new URLSearchParams(location.search);
const sceneUrl = params.get("scene") ?? "scenes/synth_open/scene.json";

const TOPO_LABELS: Record<DomainId, string> = {
  none: "open space (no tiling)", torus3: "3-torus (E³)", halfturn: "half-turn space (E³)", klein: "Klein space (E³, non-orientable)",
  e434: "{4,3,4} cubic tiling (E³)", h435: "{4,3,5} cube honeycomb (H³)", s433: "{4,3,3} tesseract (S³)",
  pds: "Poincaré dodecahedral space (S³)", sw: "Seifert–Weber space (H³)",
};

async function main() {
  const gl = createContext(canvas);
  const scene = await loadScene(gl, sceneUrl, (m) => (status.textContent = m));
  const renderer = new Renderer(gl, scene);
  const flyCam = new FlyCamera();
  const isoCam = new IsoCamera();
  flyCam.attach(canvas);
  isoCam.attach(canvas);
  const cameras: RepoCamera[] = scene.manifest.cameras ?? [];
  const kPos = scene.manifest.curved?.k_max ?? 0.05;
  const kNeg = scene.manifest.curved?.k_neg ?? kPos;
  const sliderToK = (u: number) => (u < 0 ? -u * u * kNeg : u * u * kPos);
  const kToSlider = (kk: number) => (kk < 0 ? -Math.sqrt(-kk / kNeg) : Math.sqrt(kk / kPos));
  let k = 0;
  let domain: Domain | null = null;
  // flat box half-sizes from the scene bbox (metres); scene extent for curved fitting
  const bb0 = scene.manifest.bbox_min, bb1 = scene.manifest.bbox_max;
  const flatHalf: [number, number, number] = [(bb1[0] - bb0[0]) / 2, (bb1[1] - bb0[1]) / 2, (bb1[2] - bb0[2]) / 2];
  const horizExtent = Math.max(flatHalf[0], flatHalf[2]);

  const applyCamera = (c: RepoCamera) => {
    flyCam.setFromRepoCamera(c);
    const p = curvatureParams(k);
    isoCam.setFromRepoCamera(c, p.kappa, p.scale, renderer.centre);
  };
  camsEl.innerHTML = cameras.map((c, i) => `<option value="${i}">${c.name}</option>`).join("") || "<option>free</option>";
  topoEl.innerHTML = DOMAIN_IDS.map((id) => `<option value="${id}">${TOPO_LABELS[id]}</option>`).join("");
  if (cameras.length) applyCamera(cameras[0]);
  camsEl.addEventListener("change", () => { const c = cameras[Number(camsEl.value)]; if (c) applyCamera(c); });
  scaleEl.addEventListener("input", () => (scaleV.textContent = Number(scaleEl.value).toFixed(2)));

  const setK = (nk: number) => {
    const before = curvatureParams(k), after = curvatureParams(nk);
    const posM = isoCam.physicalPos(before.kappa, before.scale, renderer.centre);
    const yaw = isoCam.yaw();
    k = nk;
    renderer.setCurvature(k);
    isoCam.setPoseMetres(after.kappa, after.scale, renderer.centre, posM, yaw);
    curvV.textContent = `${k.toExponential(2)} 1/m²  (κ=${after.kappa}${after.kappa ? `, s=${after.scale.toFixed(3)}` : ""})`;
    curvEl.value = String(kToSlider(k));
  };
  const setTopology = (id: DomainId) => {
    domain = makeDomain(id, flatHalf);
    if (domain && domain.kappa !== 0) {
      // lock the curvature so the scene's horizontal extent fills the domain's inradius
      const s = domain.inradius / horizExtent;
      setK(domain.kappa * s * s);
      curvEl.disabled = true;
    } else {
      if (domain) setK(0);
      curvEl.disabled = !!domain;
    }
    renderer.setDomain(domain);
  };
  curvEl.addEventListener("input", () => setK(sliderToK(Number(curvEl.value))));
  topoEl.addEventListener("change", () => setTopology(topoEl.value as DomainId));
  setK(0);
  status.remove();

  const opts = (): RenderOptions => ({ threshold: 1e-2, nearCull: nearCullEl.checked, repoPixelGrid: repoPixEl.checked, background: [0, 0, 0] });
  const isoState = (aspect: number): IsoCameraState => ({ invW: isoCam.invW(), rayO: isoCam.worldPos(), tanHalfFov: isoCam.tanHalfFov(aspect) });

  // Test hooks
  (window as unknown as { surveyor: unknown }).surveyor = {
    scene: scene.manifest,
    render: (c: number | RepoCamera, w: number, h: number, o?: Partial<RenderOptions>) => {
      const rc = typeof c === "number" ? cameras[c] : c;
      return Array.from(renderer.renderToArray(repoCameraState(rc), { ...opts(), repoPixelGrid: true, ...o }, w, h));
    },
    renderCurved: (c: number, kk: number, w: number, h: number, o?: Partial<RenderOptions>) => {
      renderer.setCurvature(kk);
      const p = curvatureParams(kk);
      const cam = new IsoCamera();
      cam.setFromRepoCamera(cameras[c], p.kappa, p.scale, renderer.centre);
      const st: IsoCameraState = { invW: cam.invW(), rayO: cam.worldPos(), tanHalfFov: cam.tanHalfFov(w / h) };
      const out = Array.from(renderer.renderCurvedToArray(st, { ...opts(), repoPixelGrid: true, ...o }, w, h));
      renderer.setCurvature(k);
      return out;
    },
    /** render the live iso camera at the current k / topology */
    renderLive: (w: number, h: number, o?: Partial<RenderOptions>) => Array.from(renderer.renderCurvedToArray(isoState(w / h), { ...opts(), ...o }, w, h)),
    setK, setTopology,
    setFog: (m: number, hops?: number) => { renderer.fogDistanceM = m; if (hops) renderer.maxHops = hops; },
    moveCamera: (dx: number, dy: number, dz: number) => { isoCam.moveBy([dx, dy, dz]); isoCam.recentre(domain); },
    yaw: (a: number) => isoCam.yawBy(a),
    camPos: () => Array.from(isoCam.physicalPos(curvatureParams(k).kappa, curvatureParams(k).scale, renderer.centre)),
    debug: () => ({ worldPos: Array.from(isoCam.worldPos()), faces: domain ? domain.faces.map((f) => { const p = isoCam.worldPos(); return +(f.w[0] * p[0] + f.w[1] * p[1] + f.w[2] * p[2] + f.w[3] * p[3]).toFixed(3); }) : null, centre: Array.from(renderer.centre), flatHalf }),
    stats: () => renderer.stats,
    bench: (c: number, w: number, h: number, frames = 30, curved = false, kk = 0) => {
      if (!curved) return renderer.bench(repoCameraState(cameras[c]), opts(), w, h, frames);
      renderer.setCurvature(kk);
      const p = curvatureParams(kk);
      const cam = new IsoCamera();
      cam.setFromRepoCamera(cameras[c], p.kappa, p.scale, renderer.centre);
      const r = renderer.bench({ invW: cam.invW(), rayO: cam.worldPos(), tanHalfFov: cam.tanHalfFov(w / h) }, opts(), w, h, frames);
      renderer.setCurvature(k);
      return r;
    },
    benchLive: (w: number, h: number, frames = 20) => renderer.bench(isoState(w / h), opts(), w, h, frames),
  };

  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fps = 0;
  let hopsCrossed = 0;
  function loop(now: number) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(canvas.clientWidth * dpr), H = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const p = curvatureParams(k);
    const t0 = performance.now();
    if (modeEl.checked) {
      isoCam.update(dt, p.scale, !floorEl.checked);
      if (isoCam.recentre(domain) >= 0) hopsCrossed++;
      renderer.frameCurved(isoState(W / H), opts(), W, H, Number(scaleEl.value));
    } else {
      flyCam.update(dt);
      renderer.frame(flyCam.state(W / H), opts(), W, H, Number(scaleEl.value));
    }
    const cpuMs = performance.now() - t0;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    const s = renderer.stats;
    const posM = isoCam.physicalPos(p.kappa, p.scale, renderer.centre);
    hud.textContent =
      `Surveyor · ${s.mode}${domain ? ` · ${domain.name}` : ""}\n` +
      `${fps.toFixed(0)} fps · render ${s.width}x${s.height} · canvas ${W}x${H}\n` +
      `gpu sv ${s.svMs.toFixed(2)} ms · walk ${s.walkMs.toFixed(2)} ms · cpu ${cpuMs.toFixed(2)} ms\n` +
      `cells ${scene.n} · k ${scene.k} · D ${scene.d} · start ${s.startCell}\n` +
      `k = ${k.toExponential(2)} 1/m² · κ=${p.kappa} · s=${p.scale.toFixed(3)}\n` +
      `pos (m) ${Array.from(posM).map((v) => v.toFixed(2)).join(", ")}${domain ? ` · wall crossings ${hopsCrossed}` : ""}`;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => { status.textContent = String(e); console.error(e); });
