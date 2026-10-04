import { createContext } from "./render/gl";
import { loadScene, type RepoCamera } from "./foam/scene";
import { Renderer, type RenderOptions } from "./render/renderer";
import { FlyCamera, repoCameraState } from "./game/camera";

const canvas = document.getElementById("gl") as HTMLCanvasElement;
const hud = document.getElementById("hud")!;
const status = document.getElementById("status")!;
const scaleEl = document.getElementById("scale") as HTMLInputElement;
const scaleV = document.getElementById("scaleV")!;
const camsEl = document.getElementById("cams") as HTMLSelectElement;
const nearCullEl = document.getElementById("nearcull") as HTMLInputElement;
const repoPixEl = document.getElementById("repopix") as HTMLInputElement;

const params = new URLSearchParams(location.search);
const sceneUrl = params.get("scene") ?? "scenes/synth_room/scene.json";

async function main() {
  const gl = createContext(canvas);
  const scene = await loadScene(gl, sceneUrl, (m) => (status.textContent = m));
  const renderer = new Renderer(gl, scene);
  const cam = new FlyCamera();
  cam.attach(canvas);
  const cameras: RepoCamera[] = scene.manifest.cameras ?? [];
  camsEl.innerHTML = cameras.map((c, i) => `<option value="${i}">${c.name}</option>`).join("") || "<option>free</option>";
  if (cameras.length) cam.setFromRepoCamera(cameras[0]);
  camsEl.addEventListener("change", () => { const c = cameras[Number(camsEl.value)]; if (c) cam.setFromRepoCamera(c); });
  scaleEl.addEventListener("input", () => (scaleV.textContent = Number(scaleEl.value).toFixed(2)));
  status.remove();

  const opts = (): RenderOptions => ({ threshold: 1e-2, nearCull: nearCullEl.checked, repoPixelGrid: repoPixEl.checked, background: [0, 0, 0] });

  // Test hook: window.surveyor.render(cameraIndexOrRepoCamera, w, h) -> Float32Array RGB (rows top-down)
  (window as unknown as { surveyor: unknown }).surveyor = {
    scene: scene.manifest,
    render: (c: number | RepoCamera, w: number, h: number, o?: Partial<RenderOptions>) => {
      const rc = typeof c === "number" ? cameras[c] : c;
      return Array.from(renderer.renderToArray(repoCameraState(rc), { ...opts(), repoPixelGrid: true, ...o }, w, h));
    },
    stats: () => renderer.stats,
    bench: (c: number, w: number, h: number, frames = 30) => renderer.bench(repoCameraState(cameras[c]), opts(), w, h, frames),
  };

  let last = performance.now();
  let fpsAcc = 0, fpsN = 0, fps = 0;
  function loop(now: number) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.floor(canvas.clientWidth * dpr), H = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    cam.update(dt);
    const t0 = performance.now();
    renderer.frame(cam.state(W / H), opts(), W, H, Number(scaleEl.value));
    const cpuMs = performance.now() - t0;
    fpsAcc += dt; fpsN++;
    if (fpsAcc > 0.5) { fps = fpsN / fpsAcc; fpsAcc = 0; fpsN = 0; }
    const s = renderer.stats;
    hud.textContent =
      `Surveyor · flat port (Phase 1)\n` +
      `${fps.toFixed(0)} fps · render ${s.width}x${s.height} · canvas ${W}x${H}\n` +
      `gpu sv ${s.svMs.toFixed(2)} ms · walk ${s.walkMs.toFixed(2)} ms · cpu ${cpuMs.toFixed(2)} ms\n` +
      `cells ${scene.n} · k ${scene.k} · D ${scene.d} · start ${s.startCell}\n` +
      `eye ${Array.from(cam.eye).map((v) => v.toFixed(2)).join(", ")}`;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

main().catch((e) => { status.textContent = String(e); console.error(e); });
