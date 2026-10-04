/**
 * CRaFT explorer: walk a captured Power Foam scene in H³ / E³ / S³, bend the curvature with a
 * slider, or pick a closed universe from the gallery. Developer panel behind ⚙ / backquote.
 */
import { createContext } from "./render/gl";
import { loadScene, type RepoCamera } from "./foam/scene";
import { Renderer, type RenderOptions, type IsoCameraState, type LightingOptions, DEFAULT_LIGHTING } from "./render/renderer";
import { Lights } from "./render/lights";
import { FlyCamera, repoCameraState } from "./game/camera";
import { IsoCamera } from "./game/isocamera";
import { curvatureParams } from "./foam/curved";
import { DOMAIN_IDS, makeDomain, type Domain, type DomainId } from "./topology/domain";
import { Player } from "./game/player";
import { MapInset } from "./ui/overlay";
import { Gallery, buildCards } from "./ui/gallery";
import { apply, inverse, v4, embedPoint } from "./geometry/space";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("gl");
const hud = $("hud"), status = $("status"), panelEl = $("panel"), devToggle = $<HTMLButtonElement>("devToggle");
const scaleEl = $<HTMLInputElement>("scale"), scaleV = $("scaleV");
const curvEl = $<HTMLInputElement>("curv"), curvV = $("curvV"), curvWord = $("curvWord"), spaceEl = $("space");
const camsEl = $<HTMLSelectElement>("cams"), topoEl = $<HTMLSelectElement>("topo");
const modeEl = $<HTMLInputElement>("mode"), nearCullEl = $<HTMLInputElement>("nearcull"), repoPixEl = $<HTMLInputElement>("repopix"), floorEl = $<HTMLInputElement>("floor");
const collideEl = $<HTMLInputElement>("collide"), lightEl = $<HTMLInputElement>("lighting"), fogEl = $<HTMLInputElement>("fog");
const qualityEl = $<HTMLSelectElement>("quality"), fovEl = $<HTMLInputElement>("fov"), fovV = $("fovV"), sensEl = $<HTMLInputElement>("sens"), tintsEl = $<HTMLInputElement>("tints");
const titleEl = $("title"), helpEl = $("help"), mapCanvas = $<HTMLCanvasElement>("map");

const params = new URLSearchParams(location.search);
const sceneUrl = params.get("scene") ?? "scenes/synth_open/scene.json";

const TOPO_LABELS: Record<DomainId, string> = {
  none: "open space", torus3: "3-torus", halfturn: "half-turn space", klein: "Klein space",
  e434: "{4,3,4} cubic tiling", h435: "{4,3,5} cube honeycomb", s433: "{4,3,3} tesseract",
  pds: "Poincaré dodecahedral space", sw: "Seifert–Weber space",
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
  const player = new Player(scene);
  player.floorY = scene.manifest.bbox_min[1];
  player.centre = renderer.centre;
  const map = new MapInset(mapCanvas);
  const lights = new Lights();
  const lighting: LightingOptions = { ...DEFAULT_LIGHTING, lights, enabled: false, flashlight: false, fogSigmaPerM: 0.03, fogColor: [0.03, 0.035, 0.05] };
  renderer.benchLighting = lighting;

  // ---------------------------------------------------------------- state
  let k = 0;
  let domain: Domain | null = null;
  let paused = false;
  const kPos = scene.manifest.curved?.k_max ?? 0.05;
  const kNeg = scene.manifest.curved?.k_neg ?? kPos;
  const sliderToK = (u: number) => (u < 0 ? -u * u * kNeg : u * u * kPos);
  const kToSlider = (kk: number) => Math.max(-1, Math.min(1, kk < 0 ? -Math.sqrt(-kk / kNeg) : Math.sqrt(kk / kPos)));
  const bb0 = scene.manifest.bbox_min, bb1 = scene.manifest.bbox_max;
  const flatHalf: [number, number, number] = [(bb1[0] - bb0[0]) / 2, (bb1[1] - bb0[1]) / 2, (bb1[2] - bb0[2]) / 2];
  const horizExtent = Math.max(flatHalf[0], flatHalf[2]);

  const applyCamera = (c: RepoCamera) => {
    flyCam.setFromRepoCamera(c);
    const p = curvatureParams(k);
    isoCam.setFromRepoCamera(c, p.kappa, p.scale, renderer.centre);
    player.cell = -1;
  };
  const describe = () => {
    const p = curvatureParams(k);
    curvWord.textContent = p.kappa < 0 ? "hyperbolic" : p.kappa > 0 ? "spherical" : "flat";
    curvV.textContent = p.kappa === 0 ? "k = 0 · ordinary Euclidean space" : `k = ${k.toExponential(2)} per m² · R = ${(1 / p.scale).toFixed(1)} m`;
    spaceEl.textContent = domain ? TOPO_LABELS[domain.id as DomainId] : "open space";
    document.querySelectorAll<HTMLElement>(".chip").forEach((c) => c.classList.toggle("on", Math.sign(Number(c.dataset.k)) === p.kappa));
  };
  const setK = (nk: number, moveSlider = true) => {
    const before = curvatureParams(k), after = curvatureParams(nk);
    const posM = isoCam.physicalPos(before.kappa, before.scale, renderer.centre);
    const yaw = isoCam.yaw();
    k = nk;
    renderer.setCurvature(k);
    isoCam.setPoseMetres(after.kappa, after.scale, renderer.centre, posM, yaw);
    player.cell = -1;
    if (moveSlider) curvEl.value = String(kToSlider(k));
    describe();
  };
  const setTopology = (id: DomainId) => {
    domain = makeDomain(id, flatHalf);
    if (domain && domain.kappa !== 0) { const s = domain.inradius / horizExtent; setK(domain.kappa * s * s); }
    else if (domain) setK(0);
    curvEl.disabled = !!domain;
    renderer.setDomain(domain);
    topoEl.value = id;
    describe();
  };

  // ---------------------------------------------------------------- UI
  camsEl.innerHTML = cameras.map((c, i) => `<option value="${i}">${c.name}</option>`).join("") || "<option>free</option>";
  topoEl.innerHTML = DOMAIN_IDS.map((id) => `<option value="${id}">${TOPO_LABELS[id]}</option>`).join("");
  camsEl.addEventListener("change", () => { const c = cameras[Number(camsEl.value)]; if (c) applyCamera(c); });
  topoEl.addEventListener("change", () => setTopology(topoEl.value as DomainId));
  curvEl.addEventListener("input", () => setK(sliderToK(Number(curvEl.value)), false));
  document.querySelectorAll<HTMLElement>(".chip").forEach((c) => c.addEventListener("click", () => { if (domain) setTopology("none"); setK(sliderToK(Number(c.dataset.k))); }));
  $("openBtn").addEventListener("click", () => setTopology("none"));
  scaleEl.addEventListener("input", () => (scaleV.textContent = Number(scaleEl.value).toFixed(2)));
  fovEl.addEventListener("input", () => { isoCam.fovDeg = Number(fovEl.value); flyCam.fovDeg = isoCam.fovDeg; fovV.textContent = `${fovEl.value}°`; });
  sensEl.addEventListener("input", () => { isoCam.sensitivity = Number(sensEl.value); });
  tintsEl.addEventListener("change", () => { renderer.tintStrength = tintsEl.checked ? 0.35 : 0; });
  isoCam.moveHook = (d) => { player.tryMove(isoCam, renderer.curved, d, curvatureParams(k).scale); };

  const PRESETS: Record<string, { scale: number; shadow: number; hops: number }> = { low: { scale: 0.5, shadow: 0.35, hops: 8 }, medium: { scale: 0.75, shadow: 0.5, hops: 12 }, high: { scale: 1, shadow: 0.5, hops: 16 } };
  let autoScale = 1;
  const applyQuality = () => {
    const q = qualityEl.value;
    if (q === "auto") { lighting.shadowScale = 0.5; renderer.maxHops = 12; }
    else { const pr = PRESETS[q]; scaleEl.value = String(pr.scale); scaleV.textContent = pr.scale.toFixed(2); lighting.shadowScale = pr.shadow; renderer.maxHops = pr.hops; }
  };
  qualityEl.addEventListener("change", applyQuality);

  let devVisible = false;
  const toggleDev = () => { devVisible = !devVisible; panelEl.style.display = devVisible ? "block" : "none"; hud.style.display = devVisible ? "block" : "none"; };
  devToggle.addEventListener("click", toggleDev);

  const gallery = new Gallery(document.body, buildCards(flatHalf), (id) => setTopology(id));
  let thumbsDone = false;
  const renderThumbnails = async () => {
    if (thumbsDone) return;
    thumbsDone = true;
    const saveK = k, saveDomain = (domain?.id ?? "none") as DomainId, savePitch = isoCam.pitch;
    const p0 = curvatureParams(k);
    const savePosM = isoCam.physicalPos(p0.kappa, p0.scale, renderer.centre), saveYaw = isoCam.yaw();
    const lo: LightingOptions = { ...lighting };
    for (const card of gallery.cards) {
      setTopology(card.id as DomainId);
      if (cameras.length) applyCamera(cameras[0]);
      isoCam.yawBy(0.6);
      lo.camFwdWorld = apply(isoCam.invW(), v4(0, 0, 0, -1));
      const w = 128, h = 128;
      const arr = renderer.renderLitToArray(isoState(1), opts(), lo, w, h);
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      const ctx = c.getContext("2d")!; const im = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) { im.data[i * 4] = Math.min(255, arr[i * 3] * 255); im.data[i * 4 + 1] = Math.min(255, arr[i * 3 + 1] * 255); im.data[i * 4 + 2] = Math.min(255, arr[i * 3 + 2] * 255); im.data[i * 4 + 3] = 255; }
      ctx.putImageData(im, 0, 0);
      gallery.setThumb(card.id as DomainId, c.toDataURL("image/png"));
      await new Promise((r) => setTimeout(r, 0));
    }
    setTopology(saveDomain); setK(saveK);
    const p1 = curvatureParams(k);
    isoCam.setPoseMetres(p1.kappa, p1.scale, renderer.centre, savePosM, saveYaw); isoCam.pitch = savePitch; player.cell = -1;
  };
  const openGallery = () => { gallery.show(); renderThumbnails().catch((e) => console.warn("thumbnails", e)); };
  $("galleryBtn").addEventListener("click", openGallery);
  $("helpClose").addEventListener("click", () => { helpEl.style.display = "none"; paused = false; });

  const start = (mode: string) => {
    titleEl.style.display = "none"; paused = false;
    if (mode === "gallery") openGallery();
  };
  titleEl.querySelectorAll<HTMLElement>(".act").forEach((b) => b.addEventListener("click", () => start(b.dataset.mode!)));

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.code === "Escape") {
      if (gallery.visible) { gallery.hide(); return; }
      if (titleEl.style.display !== "none") return;
      const open = helpEl.style.display !== "flex";
      helpEl.style.display = open ? "flex" : "none"; paused = open;
      return;
    }
    if (titleEl.style.display !== "none") return;
    if (e.code === "KeyG") { if (gallery.visible) gallery.hide(); else openGallery(); }
    if (e.code === "KeyM") map.toggle();
    if (e.code === "Backquote") toggleDev();
  });

  // ---------------------------------------------------------------- test hooks
  const opts = (): RenderOptions => ({ threshold: 1e-2, nearCull: nearCullEl.checked, repoPixelGrid: repoPixEl.checked, background: [0, 0, 0] });
  const isoState = (aspect: number): IsoCameraState => ({ invW: isoCam.invW(), rayO: isoCam.worldPos(), tanHalfFov: isoCam.tanHalfFov(aspect) });
  const hooks = {
    scene: scene.manifest,
    render: (c: number | RepoCamera, w: number, h: number, o?: Partial<RenderOptions>) => { const rc = typeof c === "number" ? cameras[c] : c; return Array.from(renderer.renderToArray(repoCameraState(rc), { ...opts(), repoPixelGrid: true, ...o }, w, h)); },
    renderCurved: (c: number, kk: number, w: number, h: number, o?: Partial<RenderOptions>) => {
      renderer.setCurvature(kk); const p = curvatureParams(kk); const cam = new IsoCamera(); cam.setFromRepoCamera(cameras[c], p.kappa, p.scale, renderer.centre);
      const out = Array.from(renderer.renderCurvedToArray({ invW: cam.invW(), rayO: cam.worldPos(), tanHalfFov: cam.tanHalfFov(w / h) }, { ...opts(), repoPixelGrid: true, ...o }, w, h));
      renderer.setCurvature(k); return out;
    },
    renderLive: (w: number, h: number, o?: Partial<RenderOptions>) => Array.from(renderer.renderCurvedToArray(isoState(w / h), { ...opts(), ...o }, w, h)),
    renderLit: (w: number, h: number) => { lighting.camFwdWorld = apply(isoCam.invW(), v4(0, 0, 0, -1)); return Array.from(renderer.renderLitToArray(isoState(w / h), opts(), lighting, w, h)); },
    setK, setTopology, start,
    setSlider: (u: number) => { curvEl.value = String(u); curvEl.dispatchEvent(new Event("input")); },
    setFog: (m: number, hops?: number) => { renderer.fogDistanceM = m; if (hops) renderer.maxHops = hops; },
    moveCamera: (dx: number, dy: number, dz: number) => { isoCam.moveBy([dx, dy, dz]); player.recentre(isoCam, domain); },
    walk: (dx: number, dy: number, dz: number) => { player.tryMove(isoCam, renderer.curved, [dx, dy, dz], curvatureParams(k).scale); player.recentre(isoCam, domain); },
    yaw: (a: number) => isoCam.yawBy(a),
    camPos: () => Array.from(isoCam.physicalPos(curvatureParams(k).kappa, curvatureParams(k).scale, renderer.centre)),
    setPose: (xM: number, yM: number, zM: number, yawDeg: number) => { const p = curvatureParams(k); isoCam.setPoseMetres(p.kappa, p.scale, renderer.centre, [xM, yM, zM], (yawDeg * Math.PI) / 180); player.cell = -1; },
    solidAt: (xM: number, yM: number, zM: number) => { const p = curvatureParams(k); return player.isSolid(renderer.curved, embedPoint(p.kappa, [(xM - renderer.centre[0]) * p.scale, (yM - renderer.centre[1]) * p.scale, (zM - renderer.centre[2]) * p.scale])); },
    pressKey: (code: string) => window.dispatchEvent(new KeyboardEvent("keydown", { code })),
    toggleMap: () => map.toggle(),
    stats: () => renderer.stats,
    gallery: () => ({ cards: gallery.cards.map((c) => ({ id: c.id, name: c.name, badge: c.badge, svgLen: c.svg.length, thumb: gallery.thumbs.has(c.id as DomainId) })) }),
    gallerySvg: (id: string) => gallery.cards.find((c) => c.id === id)?.svg,
    bench: (c: number, w: number, h: number, frames = 30, curved = false, kk = 0) => {
      if (!curved) return renderer.bench(repoCameraState(cameras[c]), opts(), w, h, frames);
      renderer.setCurvature(kk); const p = curvatureParams(kk); const cam = new IsoCamera(); cam.setFromRepoCamera(cameras[c], p.kappa, p.scale, renderer.centre);
      const r = renderer.bench({ invW: cam.invW(), rayO: cam.worldPos(), tanHalfFov: cam.tanHalfFov(w / h) }, opts(), w, h, frames); renderer.setCurvature(k); return r;
    },
    benchLive: (w: number, h: number, frames = 20) => renderer.bench(isoState(w / h), opts(), w, h, frames),
  };
  (window as unknown as { craft: unknown }).craft = hooks;

  // ---------------------------------------------------------------- start
  status.remove();
  applyQuality();
  setTopology("none");
  if (cameras.length) applyCamera(cameras[0]);
  const deep = params.get("mode");
  if (deep) start(deep);

  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fps = 0, hopsCrossed = 0;
  function loop(now: number) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(canvas.clientWidth * dpr), H = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const p = curvatureParams(k);
    const t0 = performance.now();
    if (modeEl.checked) {
      if (!paused && titleEl.style.display === "none" && !gallery.visible) {
        player.collisions = collideEl.checked;
        isoCam.update(dt, p.scale, !floorEl.checked);
        if (player.recentre(isoCam, domain) >= 0) hopsCrossed++;
        player.locate(renderer.curved, isoCam.worldPos());
      }
      lighting.enabled = lightEl.checked; lighting.flashlight = lightEl.checked;
      lighting.fogSigmaPerM = Number(fogEl.value);
      lighting.camFwdWorld = apply(isoCam.invW(), v4(0, 0, 0, -1));
      renderer.frameCurved(isoState(W / H), opts(), W, H, Number(scaleEl.value), lighting);
      const invWb = inverse(p.kappa, isoCam.Wb);
      const sh = scene.manifest.bbox_max.map((v, i) => ((v - scene.manifest.bbox_min[i]) / 2) * p.scale) as [number, number, number];
      map.draw({ kappa: p.kappa, scale: p.scale, W: isoCam.W(), camWorld: isoCam.worldPos(), headingWorld: apply(invWb, v4(0, 0, 0, -1)), domain, sceneHalfModel: sh });
    } else {
      flyCam.update(dt);
      renderer.frame(flyCam.state(W / H), opts(), W, H, Number(scaleEl.value));
    }
    const cpuMs = performance.now() - t0;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) {
      fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0;
      if (qualityEl.value === "auto" && modeEl.checked) {
        if (fps < 28 && autoScale > 0.4) autoScale = Math.max(0.4, autoScale * 0.9);
        else if (fps > 50 && autoScale < 1) autoScale = Math.min(1, autoScale * 1.08);
        scaleEl.value = autoScale.toFixed(2); scaleV.textContent = autoScale.toFixed(2);
      }
    }
    const s = renderer.stats;
    const posM = isoCam.physicalPos(p.kappa, p.scale, renderer.centre);
    hud.textContent =
      `${s.mode}${domain ? ` · ${domain.name}` : ""}\n` +
      `${fps.toFixed(0)} fps · render ${s.width}x${s.height} · canvas ${W}x${H}\n` +
      `gpu sv ${s.svMs.toFixed(2)} ms · walk ${s.walkMs.toFixed(2)} ms · cpu ${cpuMs.toFixed(2)} ms\n` +
      `cells ${scene.n} · k ${scene.k} · D ${scene.d} · start ${s.startCell}\n` +
      `k = ${k.toExponential(2)} 1/m² · κ=${p.kappa} · s=${p.scale.toFixed(3)}\n` +
      `pos (m) ${Array.from(posM).map((v) => v.toFixed(2)).join(", ")} · cell ${player.cell}${domain ? ` · wall crossings ${hopsCrossed}` : ""}`;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => { status.textContent = String(e); console.error(e); });
