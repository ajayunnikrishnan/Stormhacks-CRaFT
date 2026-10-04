import { createContext } from "./render/gl";
import { loadScene, type RepoCamera } from "./foam/scene";
import { Renderer, type RenderOptions, type IsoCameraState, type LightingOptions, DEFAULT_LIGHTING } from "./render/renderer";
import { Lights } from "./render/lights";
import { CurvedAudio } from "./audio/curvedAudio";
import { FlyCamera, repoCameraState } from "./game/camera";
import { IsoCamera } from "./game/isocamera";
import { curvatureParams, type CurvatureParams } from "./foam/curved";
import { DOMAIN_IDS, makeDomain, type Domain, type DomainId } from "./topology/domain";
import { Player } from "./game/player";
import { Overlay } from "./ui/overlay";
import { Hud } from "./ui/hud";
import { Gallery, buildCards } from "./ui/gallery";
import { LEVELS, LESSON_CARDS, examCurvature, examScore, type Level, type LevelState, type ToolId } from "./game/levels";
import { LightMeter, scalarIrradiance, triangle } from "./game/tools";
import { apply, inverse, v4, geodesic, tangentialize, embedPoint, logAtOrigin, type V4 } from "./geometry/space";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("gl");
const hud = $("hud"), status = $("status");
const scaleEl = $<HTMLInputElement>("scale"), scaleV = $("scaleV");
const curvEl = $<HTMLInputElement>("curv"), curvV = $("curvV");
const camsEl = $<HTMLSelectElement>("cams");
const topoEl = $<HTMLSelectElement>("topo");
const levelEl = $<HTMLSelectElement>("level");
const modeEl = $<HTMLInputElement>("mode"), nearCullEl = $<HTMLInputElement>("nearcull"), repoPixEl = $<HTMLInputElement>("repopix"), floorEl = $<HTMLInputElement>("floor");
const collideEl = $<HTMLInputElement>("collide");
const lightEl = $<HTMLInputElement>("lighting"), ambientEl = $<HTMLInputElement>("ambient"), fogEl = $<HTMLInputElement>("fog"), audioEl = $<HTMLInputElement>("audio");
const overlayCanvas = $<HTMLCanvasElement>("overlay");
const qualityEl = $<HTMLSelectElement>("quality"), fovEl = $<HTMLInputElement>("fov"), fovV = $("fovV"), sensEl = $<HTMLInputElement>("sens"), tintsEl = $<HTMLInputElement>("tints");
const titleEl = $("title"), devToggle = $<HTMLButtonElement>("devToggle"), panelEl = $("panel"), playerPanel = $("playerPanel"), curvWord = $("curvWord");
const estimateEl = $("estimate"), estSlider = $<HTMLInputElement>("estSlider"), estV = $("estV"), estSubmit = $<HTMLButtonElement>("estSubmit");

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
  const player = new Player(scene);
  player.floorY = scene.manifest.bbox_min[1];
  player.centre = renderer.centre;
  const overlay = new Overlay(overlayCanvas);
  const gameHud = new Hud(document.body);
  const lights = new Lights();
  const audio = new CurvedAudio();
  const meter = new LightMeter();
  const lighting: LightingOptions = { ...DEFAULT_LIGHTING, lights };
  renderer.benchLighting = lighting;

  // ---------------------------------------------------------------- state
  let k = 0;
  let domain: Domain | null = null;
  let flashOn = true;
  let laser: { o: V4; v: V4; length: number } | null = null;
  const beacons: V4[] = [];
  let level: Level = LEVELS[LEVELS.length - 1];
  let levelState: LevelState = { beacons, meterSamples: 0, lampsPlaced: 0, laserUsed: false, compassLoopDeg: 0, targetIrradiance: 0, submittedK: null, elapsed: 0, movedM: 0 };
  let examK = 0; // hidden true curvature in the exam
  let examReveal: { t0: number; from: number; score: number; errPct: number } | null = null;
  let paused = false;
  let targetWorld: V4 | null = null;
  const kPos = scene.manifest.curved?.k_max ?? 0.05;
  const kNeg = scene.manifest.curved?.k_neg ?? kPos;
  const sliderToK = (u: number) => (u < 0 ? -u * u * kNeg : u * u * kPos);
  const kToSlider = (kk: number) => Math.max(-1, Math.min(1, kk < 0 ? -Math.sqrt(-kk / kNeg) : Math.sqrt(kk / kPos)));
  const bb0 = scene.manifest.bbox_min, bb1 = scene.manifest.bbox_max;
  const flatHalf: [number, number, number] = [(bb1[0] - bb0[0]) / 2, (bb1[1] - bb0[1]) / 2, (bb1[2] - bb0[2]) / 2];
  const horizExtent = Math.max(flatHalf[0], flatHalf[2]);
  const has = (t: ToolId) => level.env.tools.includes(t);
  // quality presets (§5): render scale, shadow resolution, hop cap; "auto" adapts the scale to hit 30 fps
  const PRESETS: Record<string, { scale: number; shadow: number; hops: number }> = { low: { scale: 0.5, shadow: 0.35, hops: 8 }, medium: { scale: 0.75, shadow: 0.5, hops: 12 }, high: { scale: 1, shadow: 0.5, hops: 16 } };
  let autoScale = 1;
  const applyQuality = () => {
    const q = qualityEl.value;
    if (q === "auto") { lighting.shadowScale = 0.5; renderer.maxHops = 12; }
    else { const pr = PRESETS[q]; scaleEl.value = String(pr.scale); scaleV.textContent = pr.scale.toFixed(2); lighting.shadowScale = pr.shadow; renderer.maxHops = pr.hops; }
  };
  qualityEl.addEventListener("change", applyQuality);
  fovEl.addEventListener("input", () => { isoCam.fovDeg = Number(fovEl.value); flyCam.fovDeg = isoCam.fovDeg; fovV.textContent = `${fovEl.value}°`; });
  sensEl.addEventListener("input", () => { isoCam.sensitivity = Number(sensEl.value); });
  tintsEl.addEventListener("change", () => { renderer.tintStrength = tintsEl.checked ? 0.35 : 0; });

  // ---------------------------------------------------------------- helpers
  const toModel = (m: [number, number, number]) => { const p = curvatureParams(k); return embedPoint(p.kappa, [(m[0] - renderer.centre[0]) * p.scale, (m[1] - renderer.centre[1]) * p.scale, (m[2] - renderer.centre[2]) * p.scale]); };
  /** re-embed a model point from one curvature to another through its metric coordinates */
  const remap = (x: V4, before: CurvatureParams, after: CurvatureParams): V4 => {
    const l = logAtOrigin(before.kappa, x);
    const f = after.scale / before.scale;
    return embedPoint(after.kappa, [l[0] * f, l[1] * f, l[2] * f]);
  };
  const aheadWorld = (dM: number, vertical = 0) => {
    const p = curvatureParams(k);
    const invWb = inverse(p.kappa, isoCam.Wb);
    const dir = tangentialize(p.kappa, v4(1, 0, 0, 0), v4(0, 0, vertical, -1));
    return apply(invWb, geodesic(p.kappa, v4(1, 0, 0, 0), dir, dM * p.scale));
  };
  const applyCamera = (c: RepoCamera) => {
    flyCam.setFromRepoCamera(c);
    const p = curvatureParams(k);
    isoCam.setFromRepoCamera(c, p.kappa, p.scale, renderer.centre);
    player.cell = -1;
  };
  const setK = (nk: number, moveSlider = true) => {
    const before = curvatureParams(k), after = curvatureParams(nk);
    const posM = isoCam.physicalPos(before.kappa, before.scale, renderer.centre);
    const yaw = isoCam.yaw();
    k = nk;
    renderer.setCurvature(k);
    isoCam.setPoseMetres(after.kappa, after.scale, renderer.centre, posM, yaw);
    player.cell = -1;
    for (const l of lights.list) l.pos = remap(l.pos, before, after);
    for (let i = 0; i < beacons.length; i++) beacons[i] = remap(beacons[i], before, after);
    if (level.targetM) targetWorld = toModel(level.targetM);
    if (!level.env.hideCurvature) {
      curvV.textContent = `k = ${k.toExponential(2)} per m²  (κ=${after.kappa}${after.kappa ? `, radius of curvature ${(1 / after.scale).toFixed(1)} m` : ""})`;
      curvWord.textContent = after.kappa < 0 ? "hyperbolic" : after.kappa > 0 ? "spherical" : "flat";
    }
    if (moveSlider) curvEl.value = String(kToSlider(k));
  };
  const setTopology = (id: DomainId) => {
    domain = makeDomain(id, flatHalf);
    if (domain && domain.kappa !== 0) { const s = domain.inradius / horizExtent; setK(domain.kappa * s * s); curvEl.disabled = true; }
    else { if (domain) setK(0); curvEl.disabled = !!domain || !has("slider"); }
    renderer.setDomain(domain);
    topoEl.value = id;
    gameHud.lesson(level.id === "sandbox" && id !== "none" ? LESSON_CARDS[id] : level.lesson || null);
  };
  const clearLights = () => { for (const l of lights.list) audio.remove(l.id); lights.list.length = 0; };
  const gallery = new Gallery(document.body, buildCards(flatHalf), (id) => setTopology(id));
  const openGallery = () => { if (!has("topology")) return; gallery.show(); renderThumbnails().catch((e) => console.warn("thumbnails", e)); };
  $("galleryBtn").addEventListener("click", openGallery);
  /** Thumbnails: render each space from a fixed pose with our own renderer, then restore the state. */
  let thumbsDone = false;
  const renderThumbnails = async () => {
    if (thumbsDone) return;
    thumbsDone = true;
    // save the pose in physical coordinates so it survives the curvature/topology switches below
    const saveK = k, saveDomain = (domain?.id ?? "none") as DomainId, savePitch = isoCam.pitch;
    const p0 = curvatureParams(k);
    const savePosM = isoCam.physicalPos(p0.kappa, p0.scale, renderer.centre), saveYaw = isoCam.yaw();
    const saveLevelLesson = gameHud.lessonEl.style.display;
    const lo: LightingOptions = { ...lighting, enabled: true, flashlight: true, ambient: 0.3, fogSigmaPerM: 0.03 };
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
    gameHud.lessonEl.style.display = saveLevelLesson;
  };

  const startLevel = (lv: Level) => {
    level = lv;
    levelState = { beacons, meterSamples: 0, lampsPlaced: 0, laserUsed: false, compassLoopDeg: 0, targetIrradiance: 0, submittedK: null, elapsed: 0, movedM: 0 };
    beacons.length = 0; clearLights(); meter.clear(); meter.active = false; laser = null; examReveal = null;
    const env = lv.env;
    setTopology(env.topology);
    examK = lv.id === "exam" ? examCurvature(Date.now() & 0xffff) : 0;
    setK(lv.id === "exam" ? examK : env.k, lv.id !== "exam");
    if (lv.id === "exam") { curvV.textContent = "hidden"; curvWord.textContent = "?"; }
    ambientEl.value = String(env.ambient); fogEl.value = String(env.fogSigma);
    flashOn = env.flashlight; lightEl.checked = true;
    curvEl.disabled = !has("slider") || domain !== null;
    topoEl.disabled = !has("topology");
    overlay.showMap = has("map"); overlay.showCompass = has("compass");
    if (cameras.length) applyCamera(cameras[Math.min(cameras.length - 1, env.spawn ?? 0)]);
    targetWorld = lv.targetM ? toModel(lv.targetM) : null;
    gameHud.lesson(lv.lesson || null);
    gameHud.banner(`<b>${lv.title}</b><br><span style="font-size:14px">${lv.goal}</span>`, 6000);
    levelEl.value = lv.id;
    // player-facing panels: sandbox gets the curvature/gallery panel, the exam gets the estimate panel
    playerPanel.style.display = lv.id === "sandbox" ? "block" : "none";
    estimateEl.style.display = lv.id === "exam" ? "block" : "none";
    if (lv.id === "exam") { estSlider.value = "0"; estV.textContent = "k = 0 (flat)"; estSubmit.disabled = false; estSubmit.textContent = "Submit guess (Enter)"; }
  };
  const submitExam = () => {
    if (level.id !== "exam" || levelState.submittedK !== null) return;
    const guess = sliderToK(Number(estSlider.value));
    levelState.submittedK = guess;
    const { score, errorPct } = examScore(examK, guess);
    examReveal = { t0: performance.now(), from: examK, score, errPct: errorPct };
    estSubmit.disabled = true; estSubmit.textContent = `score ${score}/100`;
    estV.textContent = `true k = ${examK.toExponential(2)} · yours ${guess.toExponential(2)}`;
    gameHud.banner(`<b>True curvature: ${examK.toExponential(2)} per m² (${examK < 0 ? "hyperbolic" : "spherical"})</b><br>your estimate ${guess.toExponential(2)} · error ${errorPct.toFixed(0)}% · <b>score ${score}/100</b><br><span style="font-size:13px">watch the space flatten and bend back… (R to retry with a new hidden value)</span>`, 9000);
  };
  estSlider.addEventListener("input", () => { const g = sliderToK(Number(estSlider.value)); estV.textContent = `k = ${g.toExponential(2)} per m² (${g < -1e-9 ? "hyperbolic" : g > 1e-9 ? "spherical" : "flat"})`; });
  estSubmit.addEventListener("click", submitExam);

  // ---------------------------------------------------------------- UI wiring
  camsEl.innerHTML = cameras.map((c, i) => `<option value="${i}">${c.name}</option>`).join("") || "<option>free</option>";
  topoEl.innerHTML = DOMAIN_IDS.map((id) => `<option value="${id}">${TOPO_LABELS[id]}</option>`).join("");
  levelEl.innerHTML = LEVELS.map((l) => `<option value="${l.id}">${l.title}</option>`).join("");
  camsEl.addEventListener("change", () => { const c = cameras[Number(camsEl.value)]; if (c) applyCamera(c); });
  scaleEl.addEventListener("input", () => (scaleV.textContent = Number(scaleEl.value).toFixed(2)));
  curvEl.addEventListener("input", () => setK(sliderToK(Number(curvEl.value)), false));
  topoEl.addEventListener("change", () => setTopology(topoEl.value as DomainId));
  levelEl.addEventListener("change", () => startLevel(LEVELS.find((l) => l.id === levelEl.value)!));
  isoCam.moveHook = (d) => { player.tryMove(isoCam, renderer.curved, d, curvatureParams(k).scale); };
  isoCam.yawHook = (R) => player.onYaw(R);

  let toggleDev: () => void = () => {};
  let showTitle: () => void = () => {};
  let chooseMode: (mode: string) => void = () => {};
  const controlsHtml = `<h3 style="margin:0 0 8px">CRaFT · Surveyor — paused</h3>
    <b>Move</b> WASD (shift = run) · <b>Look</b> drag · <b>Esc</b> resume<br>
    <b>B</b> beacon (3 → triangle) · <b>I</b> light meter · <b>L</b> laser · <b>M</b> map · <b>C</b> compass<br>
    <b>P</b> lamp · <b>T</b> flare · <b>F</b> flashlight · <b>X</b> clear lights · <b>R</b> restart level · <b>N</b> next level<br>
    <b>Enter</b> submit curvature (exam)<br><br>
    <span style="color:#aaa">CRaFT (Curved Radiance Foam Tracing): a captured Power Foam scene ray traced along exact geodesics of H³, E³ or S³; see docs/WRITEUP.md.</span>`;

  const makeLaser = () => { const p = curvatureParams(k); const o = isoCam.worldPos(); return { o, v: tangentialize(p.kappa, o, apply(isoCam.invW(), v4(0, 0, 0, -1))), length: 30 * p.scale }; };

  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.code === "Escape") {
      if (gallery.visible) { gallery.hide(); return; }
      if (titleEl.style.display !== "none") { titleEl.style.display = "none"; paused = false; return; }
      paused = !paused; gameHud.showMenu(paused ? controlsHtml + `<br><button id="menuTitle" class="pbtn" style="width:auto">Back to the main menu</button>` : null);
      if (paused) document.getElementById("menuTitle")?.addEventListener("click", () => { gameHud.showMenu(null); showTitle(); });
      return;
    }
    if (e.code === "KeyG" && has("topology")) { if (gallery.visible) gallery.hide(); else openGallery(); return; }
    if (paused || gallery.visible) return;
    const p = curvatureParams(k);
    if (e.code === "KeyL" && has("laser")) { laser = laser ? null : makeLaser(); if (laser) levelState.laserUsed = true; }
    if (e.code === "KeyM" && has("map")) overlay.showMap = !overlay.showMap;
    if (e.code === "KeyC" && has("compass")) overlay.showCompass = !overlay.showCompass;
    if (e.code === "KeyB" && has("beacons")) { if (beacons.length >= 3) beacons.length = 0; beacons.push(isoCam.worldPos()); }
    if (e.code === "KeyI" && has("meter")) { meter.active = !meter.active; if (meter.active) meter.clear(); }
    if (e.code === "KeyF" && has("flashlight")) flashOn = !flashOn;
    if (e.code === "KeyP" && has("lamps")) {
      const budget = level.env.lampBudget;
      if (budget < 0 || levelState.lampsPlaced < budget) {
        const pos = aheadWorld(1.5, 0.25);
        const l = lights.addLamp(pos, [9, 8, 6.5]);
        levelState.lampsPlaced++;
        if (audioEl.checked) audio.addHum(l.id, pos);
      } else gameHud.banner("no lamps left — X clears them (restarts the count)", 2000);
    }
    if (e.code === "KeyT" && has("flares")) {
      const o = isoCam.worldPos();
      const dir = tangentialize(p.kappa, o, apply(isoCam.invW(), v4(0, 0, 0, -1)));
      const l = lights.throwFlare(o, dir, 7 * p.scale);
      if (audioEl.checked) audio.addCrackle(l.id, l.pos);
    }
    if (e.code === "KeyX") { clearLights(); levelState.lampsPlaced = 0; }
    if (e.code === "KeyR") startLevel(level);
    if (e.code === "KeyN") { const i = LEVELS.indexOf(level); startLevel(LEVELS[(i + 1) % LEVELS.length]); }
    if (e.code === "Enter") submitExam();
    if (e.code === "Backquote") toggleDev();
  });
  canvas.addEventListener("pointerdown", () => { if (audioEl.checked) audio.ensure(); });

  // ---------------------------------------------------------------- test hooks
  const opts = (): RenderOptions => ({ threshold: 1e-2, nearCull: nearCullEl.checked, repoPixelGrid: repoPixEl.checked, background: [0, 0, 0] });
  const isoState = (aspect: number): IsoCameraState => ({ invW: isoCam.invW(), rayO: isoCam.worldPos(), tanHalfFov: isoCam.tanHalfFov(aspect) });
  (window as unknown as { surveyor: unknown }).surveyor = {
    scene: scene.manifest,
    render: (c: number | RepoCamera, w: number, h: number, o?: Partial<RenderOptions>) => { const rc = typeof c === "number" ? cameras[c] : c; return Array.from(renderer.renderToArray(repoCameraState(rc), { ...opts(), repoPixelGrid: true, ...o }, w, h)); },
    renderCurved: (c: number, kk: number, w: number, h: number, o?: Partial<RenderOptions>) => {
      renderer.setCurvature(kk); const p = curvatureParams(kk); const cam = new IsoCamera(); cam.setFromRepoCamera(cameras[c], p.kappa, p.scale, renderer.centre);
      const out = Array.from(renderer.renderCurvedToArray({ invW: cam.invW(), rayO: cam.worldPos(), tanHalfFov: cam.tanHalfFov(w / h) }, { ...opts(), repoPixelGrid: true, ...o }, w, h));
      renderer.setCurvature(k); return out;
    },
    renderLive: (w: number, h: number, o?: Partial<RenderOptions>) => Array.from(renderer.renderCurvedToArray(isoState(w / h), { ...opts(), ...o }, w, h)),
    renderLit: (w: number, h: number) => { lighting.camFwdWorld = apply(isoCam.invW(), v4(0, 0, 0, -1)); return Array.from(renderer.renderLitToArray(isoState(w / h), opts(), lighting, w, h)); },
    setK, setTopology, startLevel: (id: string) => startLevel(LEVELS.find((l) => l.id === id)!),
    levelState: () => ({ ...levelState, beacons: levelState.beacons.length, progress: level.check(levelState), examK, level: level.id }),
    chooseMode: (m: string) => chooseMode(m),
    submitExam: () => submitExam(),
    setEstimate: (u: number) => { estSlider.value = String(u); estSlider.dispatchEvent(new Event("input")); },
    pressKey: (code: string) => window.dispatchEvent(new KeyboardEvent("keydown", { code })),
    setSlider: (u: number) => { curvEl.value = String(u); curvEl.dispatchEvent(new Event("input")); },
    setFog: (m: number, hops?: number) => { renderer.fogDistanceM = m; if (hops) renderer.maxHops = hops; },
    addLamp: (dAheadM = 1.5, up = 0.25) => { const pos = aheadWorld(dAheadM, up); lights.addLamp(pos, [9, 8, 6.5]); levelState.lampsPlaced++; return Array.from(pos); },
    clearLights, setLighting: (o: Partial<LightingOptions>) => Object.assign(lighting, o),
    moveCamera: (dx: number, dy: number, dz: number) => { isoCam.moveBy([dx, dy, dz]); player.recentre(isoCam, domain); },
    walk: (dx: number, dy: number, dz: number) => { player.tryMove(isoCam, renderer.curved, [dx, dy, dz], curvatureParams(k).scale); player.recentre(isoCam, domain); },
    yaw: (a: number) => isoCam.yawBy(a), compassAngle: () => player.compassAngle(),
    camPos: () => Array.from(isoCam.physicalPos(curvatureParams(k).kappa, curvatureParams(k).scale, renderer.centre)),
    triangle: () => (beacons.length === 3 ? triangle(curvatureParams(k).kappa, curvatureParams(k).scale, beacons[0], beacons[1], beacons[2]) : null),
    setLaser: (on: boolean) => { laser = on ? makeLaser() : null; },
    setPose: (xM: number, yM: number, zM: number, yawDeg: number) => { const p = curvatureParams(k); isoCam.setPoseMetres(p.kappa, p.scale, renderer.centre, [xM, yM, zM], (yawDeg * Math.PI) / 180); player.cell = -1; },
    targetIrradiance: () => levelState.targetIrradiance,
    solidAt: (xM: number, yM: number, zM: number) => player.isSolid(renderer.curved, toModel([xM, yM, zM])),
    antipodeM: () => { if (!targetWorld) return null; const p = curvatureParams(k); const a = v4(-targetWorld[0], -targetWorld[1], -targetWorld[2], -targetWorld[3]); const l = logAtOrigin(p.kappa, a); return [l[0] / p.scale + renderer.centre[0], l[1] / p.scale + renderer.centre[1], l[2] / p.scale + renderer.centre[2]]; },
    cellOf: (xM: number, yM: number, zM: number) => { const p = curvatureParams(k); const x = toModel([xM, yM, zM]); const kk = p.kappa === 0 ? 1 : p.kappa; let best = -1, bv = -Infinity; const A = renderer.curved.A; for (let c = 0, o = 0; o < A.length; c++, o += 4) { const vv = kk * A[o] * x[0] + A[o + 1] * x[1] + A[o + 2] * x[2] + A[o + 3] * x[3]; if (vv > bv) { bv = vv; best = c; } } return { cell: best, sigma: (player as unknown as { info: { sigma: Float32Array } }).info.sigma[best], R: renderer.curved.rad[2 * best] / p.scale }; },
    overlayPng: (w: number, h: number) => { const c = document.createElement("canvas"); c.width = w; c.height = h; c.getContext("2d")!.drawImage(overlayCanvas, 0, 0, w, h); return c.toDataURL("image/png").split(",")[1]; },
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

  // ---------------------------------------------------------------- start
  status.remove();
  applyQuality();
  let devVisible = false;
  toggleDev = () => { devVisible = !devVisible; panelEl.style.display = devVisible ? "block" : "none"; hud.style.display = devVisible ? "block" : "none"; };
  devToggle.addEventListener("click", toggleDev);
  showTitle = () => { titleEl.style.display = "flex"; paused = true; };
  chooseMode = (mode: string) => {
    titleEl.style.display = "none"; paused = false; gameHud.showMenu(null);
    startLevel(LEVELS.find((l) => l.id === (mode === "play" ? "tutorial" : mode))!);
  };
  titleEl.querySelectorAll<HTMLElement>(".mode").forEach((m) => m.addEventListener("click", () => chooseMode(m.dataset.mode!)));
  $("pickLevelBtn").addEventListener("click", showTitle);
  // deep links (?level=…) skip the title screen; otherwise show it over the sandbox
  const deep = params.get("level");
  if (deep) chooseMode(deep === "tutorial" ? "play" : deep); else { startLevel(LEVELS[LEVELS.length - 1]); showTitle(); }

  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fps = 0, hopsCrossed = 0, lastMeterLamp = -1;
  let lastPosM: Float64Array | null = null;
  function loop(now: number) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(canvas.clientWidth * dpr), H = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const p = curvatureParams(k);
    const t0 = performance.now();
    if (modeEl.checked) {
      if (!paused) {
        player.collisions = collideEl.checked;
        isoCam.update(dt, p.scale, !floorEl.checked);
        if (player.recentre(isoCam, domain) >= 0) hopsCrossed++;
        player.locate(renderer.curved, isoCam.worldPos());
        for (const id of lights.update(p.kappa, dt, (x) => player.isSolid(renderer.curved, x))) audio.remove(id);
        for (const l of lights.list) { const srcPos = audio.sources.get(l.id); if (srcPos) srcPos.pos = l.pos; }
        levelState.elapsed += dt;
        { const pm = isoCam.physicalPos(p.kappa, p.scale, renderer.centre); if (lastPosM) levelState.movedM += Math.hypot(pm[0] - lastPosM[0], pm[2] - lastPosM[2]) < 1 ? Math.hypot(pm[0] - lastPosM[0], pm[2] - lastPosM[2]) : 0; lastPosM = pm; }
        // exam reveal: morph the rendering from the true curvature to flat and back over 6 s
        if (examReveal) {
          const u = (performance.now() - examReveal.t0) / 6000;
          if (u < 1) setK(examReveal.from * (0.5 + 0.5 * Math.cos(u * 2 * Math.PI)), false);
          else { setK(examReveal.from, false); examReveal = null; }
        }
      }
      // tools
      const camWorld = isoCam.worldPos();
      if (meter.active && lights.list.length) {
        let best = 0, bd = Infinity;
        lights.list.forEach((l, i) => { const d = scalarIrradiance(p.kappa, camWorld, l.pos, 1).d; if (d < bd) { bd = d; best = i; } });
        if (best !== lastMeterLamp) { meter.clear(); lastMeterLamp = best; }
        const L = lights.list[best];
        meter.sample(p.kappa, p.scale, camWorld, L.pos, (L.color[0] + L.color[1] + L.color[2]) / 3);
        levelState.meterSamples = Math.max(levelState.meterSamples, meter.samples.length);
      }
      if (targetWorld) {
        let E = 0;
        for (const l of lights.list) E += scalarIrradiance(p.kappa, targetWorld, l.pos, (l.color[0] + l.color[1] + l.color[2]) / 3).E * p.scale * p.scale;
        levelState.targetIrradiance = E; // model-unit irradiance × s² = W/m²
      }
      const prog = level.check(levelState);
      gameHud.setGoal(level.title, level.goal, prog.steps, prog.done);
      const tri = beacons.length === 3 ? triangle(p.kappa, p.scale, beacons[0], beacons[1], beacons[2]) : null;
      gameHud.setReadout(has("beacons") && beacons.length ? gameHud.triangleText(tri, beacons.length) : null);
      gameHud.drawMeter(meter.samples, (d) => meter.flatCurve(d), meter.active);

      lighting.enabled = lightEl.checked; lighting.flashlight = flashOn && lightEl.checked;
      lighting.ambient = Number(ambientEl.value); lighting.fogSigmaPerM = Number(fogEl.value);
      lighting.camFwdWorld = apply(isoCam.invW(), v4(0, 0, 0, -1));
      renderer.frameCurved(isoState(W / H), opts(), W, H, Number(scaleEl.value), lighting);
      audio.enabled = audioEl.checked;
      audio.update(p.kappa, p.scale, camWorld, isoCam.W());
      overlay.resize(W, H);
      const invWb = inverse(p.kappa, isoCam.Wb);
      const sh = scene.manifest.bbox_max.map((v, i) => ((v - scene.manifest.bbox_min[i]) / 2) * p.scale) as [number, number, number];
      overlay.draw({
        kappa: p.kappa, scale: p.scale, W: isoCam.W(), invW: isoCam.invW(), camWorld,
        headingWorld: apply(invWb, v4(0, 0, 0, -1)), compassAngle: player.compassAngle(), domain,
        sceneHalfModel: sh, tanHalfFov: isoCam.tanHalfFov(W / H), laser, beacons, target: targetWorld, lights: lights.list.map((l) => l.pos),
      });
    } else {
      flyCam.update(dt);
      renderer.frame(flyCam.state(W / H), opts(), W, H, Number(scaleEl.value));
      overlay.resize(W, H);
      overlay.draw({ kappa: 0, scale: 1, W: isoCam.W(), invW: isoCam.invW(), camWorld: isoCam.worldPos(), headingWorld: v4(0, 0, 0, -1), compassAngle: 0, domain: null, sceneHalfModel: [1, 1, 1], tanHalfFov: [1, 1], laser: null, beacons: [], target: null, lights: [] });
    }
    const cpuMs = performance.now() - t0;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) {
      fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0;
      if (qualityEl.value === "auto" && modeEl.checked) {
        // dynamic resolution (§5): aim for 30 fps; step the scale by ±10 % per half second
        if (fps < 28 && autoScale > 0.4) autoScale = Math.max(0.4, autoScale * 0.9);
        else if (fps > 50 && autoScale < 1) autoScale = Math.min(1, autoScale * 1.08);
        scaleEl.value = autoScale.toFixed(2); scaleV.textContent = autoScale.toFixed(2);
      }
    }
    const s = renderer.stats;
    const posM = isoCam.physicalPos(p.kappa, p.scale, renderer.centre);
    hud.textContent =
      `CRaFT · Surveyor · ${s.mode}${domain ? ` · ${domain.name}` : ""}\n` +
      `${fps.toFixed(0)} fps · render ${s.width}x${s.height} · canvas ${W}x${H}\n` +
      `gpu sv ${s.svMs.toFixed(2)} ms · walk ${s.walkMs.toFixed(2)} ms · cpu ${cpuMs.toFixed(2)} ms\n` +
      `cells ${scene.n} · k ${scene.k} · D ${scene.d} · start ${s.startCell}\n` +
      (level.env.hideCurvature ? `k = hidden · s = hidden\n` : `k = ${k.toExponential(2)} 1/m² · κ=${p.kappa} · s=${p.scale.toFixed(3)}\n`) +
      `pos (m) ${Array.from(posM).map((v) => v.toFixed(2)).join(", ")} · cell ${player.cell}${domain ? ` · wall crossings ${hopsCrossed}` : ""}\n` +
      `Esc menu · lights ${lights.list.length}`;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => { status.textContent = String(e); console.error(e); });
