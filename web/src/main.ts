import { createContext } from "./render/gl";
import { loadScene, type RepoCamera } from "./foam/scene";
import { Renderer, type RenderOptions, type IsoCameraState } from "./render/renderer";
import { FlyCamera, repoCameraState } from "./game/camera";
import { IsoCamera } from "./game/isocamera";
import { curvatureParams } from "./foam/curved";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const canvas = $<HTMLCanvasElement>("gl");
const hud = $("hud"), status = $("status");
const scaleEl = $<HTMLInputElement>("scale"), scaleV = $("scaleV");
const curvEl = $<HTMLInputElement>("curv"), curvV = $("curvV");
const camsEl = $<HTMLSelectElement>("cams");
const modeEl = $<HTMLInputElement>("mode"), nearCullEl = $<HTMLInputElement>("nearcull"), repoPixEl = $<HTMLInputElement>("repopix"), floorEl = $<HTMLInputElement>("floor");

const params = new URLSearchParams(location.search);
const sceneUrl = params.get("scene") ?? "scenes/synth_room/scene.json";

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
  // slider u ∈ [−1,1] → k = sign(u)·u²·(kNeg|kPos): quadratic so the interesting weak-curvature range is wide
  const sliderToK = (u: number) => (u < 0 ? -u * u * kNeg : u * u * kPos);
  let k = 0;

  const applyCamera = (c: RepoCamera) => {
    flyCam.setFromRepoCamera(c);
    const p = curvatureParams(k);
    isoCam.setFromRepoCamera(c, p.kappa, p.scale, renderer.centre);
  };
  camsEl.innerHTML = cameras.map((c, i) => `<option value="${i}">${c.name}</option>`).join("") || "<option>free</option>";
  if (cameras.length) applyCamera(cameras[0]);
  camsEl.addEventListener("change", () => { const c = cameras[Number(camsEl.value)]; if (c) applyCamera(c); });
  scaleEl.addEventListener("input", () => (scaleV.textContent = Number(scaleEl.value).toFixed(2)));
  const setK = (nk: number) => {
    // keep the camera's physical pose when the curvature changes: re-embed from metres
    const before = curvatureParams(k), after = curvatureParams(nk);
    const posM = isoCamPhysicalPos(isoCam, before.kappa, before.scale, renderer.centre);
    const yaw = isoCamYaw(isoCam);
    k = nk;
    renderer.setCurvature(k);
    isoCam.setPoseMetres(after.kappa, after.scale, renderer.centre, posM, yaw);
    curvV.textContent = `${k.toExponential(2)} 1/m²  (κ=${after.kappa}${after.kappa ? `, s=${after.scale.toFixed(3)}` : ""})`;
  };
  curvEl.addEventListener("input", () => setK(sliderToK(Number(curvEl.value))));
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
    setK,
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
  };

  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fps = 0;
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
      renderer.frameCurved(isoState(W / H), opts(), W, H, Number(scaleEl.value));
    } else {
      flyCam.update(dt);
      renderer.frame(flyCam.state(W / H), opts(), W, H, Number(scaleEl.value));
    }
    const cpuMs = performance.now() - t0;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    const s = renderer.stats;
    const posM = isoCamPhysicalPos(isoCam, p.kappa, p.scale, renderer.centre);
    hud.textContent =
      `Surveyor · ${s.mode}\n` +
      `${fps.toFixed(0)} fps · render ${s.width}x${s.height} · canvas ${W}x${H}\n` +
      `gpu sv ${s.svMs.toFixed(2)} ms · walk ${s.walkMs.toFixed(2)} ms · cpu ${cpuMs.toFixed(2)} ms\n` +
      `cells ${scene.n} · k ${scene.k} · D ${scene.d} · start ${s.startCell}\n` +
      `k = ${k.toExponential(2)} 1/m² · κ=${p.kappa} · s=${p.scale.toFixed(3)}\n` +
      `pos (m) ${Array.from(posM).map((v) => v.toFixed(2)).join(", ")}`;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

/** Physical position (metres, scene frame) of the iso camera: log map of W⁻¹o, unscaled, plus centre. */
function isoCamPhysicalPos(cam: IsoCamera, kappa: -1 | 0 | 1, scale: number, centre: Float64Array): Float64Array {
  return cam.physicalPos(kappa, scale, centre);
}
function isoCamYaw(cam: IsoCamera): number { return cam.yaw(); }

main().catch((e) => { status.textContent = String(e); console.error(e); });
