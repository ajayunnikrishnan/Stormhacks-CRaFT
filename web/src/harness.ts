/**
 * Validation page: renders every manifest camera with the GPU walk at the reference
 * resolution and compares against /test/ref_cam{i}_{W}.f32 (written by tools/ref_render.py).
 * Reports PSNR, max abs error and the number of pixels off by > 0.02 (seam tie-breaks).
 * Results are also exposed as window.harnessResults for scripted checks.
 */
import { createContext } from "./render/gl";
import { loadScene } from "./foam/scene";
import { Renderer } from "./render/renderer";
import { repoCameraState } from "./game/camera";

const W = 160, H = 120;
const params = new URLSearchParams(location.search);
const sceneUrl = params.get("scene") ?? "scenes/synth_room/scene.json";
const log = document.getElementById("log")!;
const tbl = document.getElementById("tbl") as HTMLTableElement;
const imgs = document.getElementById("imgs")!;

function toCanvas(rgb: Float32Array, label: string) {
  const c = document.createElement("canvas");
  c.width = W; c.height = H; c.style.width = `${W * 2}px`; c.style.height = `${H * 2}px`;
  const ctx = c.getContext("2d")!;
  const im = ctx.createImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    im.data[i * 4] = Math.min(255, Math.max(0, rgb[i * 3] * 255));
    im.data[i * 4 + 1] = Math.min(255, Math.max(0, rgb[i * 3 + 1] * 255));
    im.data[i * 4 + 2] = Math.min(255, Math.max(0, rgb[i * 3 + 2] * 255));
    im.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
  const d = document.createElement("div");
  d.appendChild(c);
  d.appendChild(document.createTextNode(label));
  imgs.appendChild(d);
}

async function main() {
  const gl = createContext(document.getElementById("gl") as HTMLCanvasElement);
  const scene = await loadScene(gl, sceneUrl, (m) => (log.textContent = m));
  const renderer = new Renderer(gl, scene);
  const cams = scene.manifest.cameras ?? [];
  const results: Record<string, unknown>[] = [];
  tbl.innerHTML = "<tr><th>camera</th><th>PSNR dB</th><th>MSE</th><th>max |err|</th><th>px > 0.02</th><th>mean ref</th><th>mean gpu</th></tr>";
  for (let i = 0; i < cams.length; i++) {
    const r = await fetch(`/test/ref_cam${i}_${W}.f32`);
    if (!r.ok) { results.push({ camera: cams[i].name, skipped: true }); continue; }
    const ref = new Float32Array(await r.arrayBuffer());
    const got = renderer.renderToArray(repoCameraState(cams[i]), { threshold: 1e-2, nearCull: true, repoPixelGrid: true, background: [0, 0, 0] }, W, H);
    let se = 0, maxe = 0, nbad = 0, sr = 0, sg = 0;
    const diff = new Float32Array(ref.length);
    for (let j = 0; j < ref.length; j++) {
      const e = Math.abs(ref[j] - got[j]);
      se += e * e; if (e > maxe) maxe = e; if (e > 0.02) nbad++;
      sr += ref[j]; sg += got[j];
      diff[j] = Math.min(1, e * 10);
    }
    const mse = se / ref.length;
    const psnr = 10 * Math.log10(1 / mse);
    const row = { camera: cams[i].name, psnr, mse, maxAbs: maxe, pxOver002: nbad, meanRef: sr / ref.length, meanGpu: sg / ref.length };
    results.push(row);
    const ok = psnr > 40;
    tbl.insertAdjacentHTML("beforeend", `<tr><td>${row.camera}</td><td class="${ok ? "ok" : "bad"}">${psnr.toFixed(2)}</td><td>${mse.toExponential(2)}</td><td>${maxe.toFixed(3)}</td><td>${nbad}</td><td>${row.meanRef.toFixed(4)}</td><td>${row.meanGpu.toFixed(4)}</td></tr>`);
    toCanvas(ref, `${row.camera} ref`);
    toCanvas(got, `${row.camera} gpu`);
    toCanvas(diff, `${row.camera} |diff|×10`);
  }
  (window as unknown as { harnessResults: unknown }).harnessResults = results;
  log.textContent = `done: ${results.length} cameras, ${W}x${H}, threshold 1e-2, near-cull on, repo pixel grid`;
}
main().catch((e) => { log.textContent = String(e); console.error(e); });
