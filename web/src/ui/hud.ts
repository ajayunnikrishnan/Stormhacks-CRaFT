/** DOM HUD: goal banner, tool readouts (beacon triangle, light meter), level banners, pause menu. */
import type { Triangle } from "../game/tools";
import type { MeterSample } from "../game/tools";

export class Hud {
  goalEl: HTMLDivElement;
  readoutEl: HTMLDivElement;
  bannerEl: HTMLDivElement;
  menuEl: HTMLDivElement;
  lessonEl: HTMLDivElement;
  meterCanvas: HTMLCanvasElement;
  private bannerTimer = 0;

  constructor(root: HTMLElement) {
    const mk = (id: string, css: string) => { const d = document.createElement("div"); d.id = id; d.style.cssText = css; root.appendChild(d); return d; };
    const base = "position:fixed;color:#eee;font-family:ui-monospace,Menlo,monospace;pointer-events:none;";
    this.goalEl = mk("goal", base + "left:50%;top:12px;transform:translateX(-50%);max-width:70vw;background:rgba(0,0,0,.55);padding:8px 14px;border-radius:8px;font-size:13px;text-align:center;line-height:1.4");
    this.readoutEl = mk("readout", base + "left:12px;top:150px;background:rgba(0,0,0,.5);padding:8px 10px;border-radius:6px;font-size:12px;white-space:pre;display:none");
    this.bannerEl = mk("banner", base + "left:50%;top:40%;transform:translate(-50%,-50%);background:rgba(0,0,0,.75);padding:18px 26px;border-radius:12px;font-size:20px;text-align:center;display:none;border:1px solid rgba(255,220,120,.5)");
    this.lessonEl = mk("lesson", base + "right:12px;bottom:12px;max-width:320px;background:rgba(20,24,40,.8);padding:10px 12px;border-radius:8px;font-size:12px;line-height:1.45;display:none;border-left:3px solid #ffd56a");
    this.menuEl = mk("menu", "position:fixed;inset:0;background:rgba(0,0,0,.7);color:#eee;font-family:ui-monospace,Menlo,monospace;display:none;align-items:center;justify-content:center;font-size:14px");
    this.menuEl.style.pointerEvents = "auto";
    this.meterCanvas = document.createElement("canvas");
    this.meterCanvas.width = 260; this.meterCanvas.height = 150;
    this.meterCanvas.style.cssText = base + "left:50%;bottom:16px;transform:translateX(-50%);background:rgba(0,0,0,.55);border-radius:6px;display:none";
    root.appendChild(this.meterCanvas);
  }

  setGoal(title: string, goal: string, steps: { text: string; done: boolean }[], complete: boolean) {
    const list = steps.map((s) => `<div style="text-align:left;margin:2px 0;color:${s.done ? "#9f9" : "#ffd56a"}">${s.done ? "✓" : "○"} ${s.text}</div>`).join("");
    this.goalEl.innerHTML = `<b>${title}</b><br><span style="color:#ddd">${goal}</span>${list ? `<div style="margin-top:6px">${list}</div>` : ""}${complete ? `<div style="margin-top:6px;color:#9f9"><b>Level complete — press N for the next one</b></div>` : ""}`;
  }
  setReadout(text: string | null) {
    this.readoutEl.style.display = text ? "block" : "none";
    if (text) this.readoutEl.textContent = text;
  }
  banner(text: string, ms = 3500) {
    this.bannerEl.innerHTML = text;
    this.bannerEl.style.display = "block";
    clearTimeout(this.bannerTimer);
    if (ms > 0) this.bannerTimer = window.setTimeout(() => (this.bannerEl.style.display = "none"), ms);
  }
  hideBanner() { this.bannerEl.style.display = "none"; }
  lesson(text: string | null) { this.lessonEl.style.display = text ? "block" : "none"; if (text) this.lessonEl.textContent = text; }

  triangleText(t: Triangle | null, n: number): string {
    if (!t) return `beacons placed: ${n}/3 — press B at another spot`;
    const hint = t.sumDeg < 179.9 ? "less than 180°: hyperbolic" : t.sumDeg > 180.1 ? "more than 180°: spherical" : "180°: flat";
    return `Triangle between your beacons\n angles  ${t.anglesDeg.map((a) => a.toFixed(1).padStart(6)).join(" ")}°\n sum     ${t.sumDeg.toFixed(2)}°  (${hint})\n sides   ${t.sidesM.map((s) => s.toFixed(2).padStart(6)).join(" ")} m`;
  }

  drawMeter(samples: MeterSample[], flat: (d: number) => number, show: boolean) {
    this.meterCanvas.style.display = show ? "block" : "none";
    if (!show) return;
    const c = this.meterCanvas, ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#ddd"; ctx.font = "11px ui-monospace, monospace";
    ctx.fillText("light meter: irradiance vs distance (grey = flat 1/d²)", 8, 14);
    if (samples.length < 2) { ctx.fillText("walk away from the lamp to collect samples", 8, 80); return; }
    const dMin = Math.min(...samples.map((s) => s.dM)), dMax = Math.max(...samples.map((s) => s.dM));
    const eMax = Math.max(...samples.map((s) => s.E), flat(dMin));
    const x = (d: number) => 30 + ((d - dMin) / Math.max(1e-6, dMax - dMin)) * (c.width - 40);
    const y = (e: number) => c.height - 18 - (Math.log10(1 + 9 * e / eMax)) * (c.height - 40);
    ctx.strokeStyle = "#888"; ctx.lineWidth = 1; ctx.beginPath();
    for (let i = 0; i <= 50; i++) { const d = dMin + ((dMax - dMin) * i) / 50; const p = [x(d), y(flat(d))]; if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]); }
    ctx.stroke();
    ctx.fillStyle = "#ffd56a";
    for (const s of samples) { ctx.beginPath(); ctx.arc(x(s.dM), y(s.E), 2.5, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = "#aaa"; ctx.fillText(`${dMin.toFixed(1)} m`, 24, c.height - 4); ctx.fillText(`${dMax.toFixed(1)} m`, c.width - 44, c.height - 4);
  }

  showMenu(html: string | null) {
    this.menuEl.style.display = html ? "flex" : "none";
    if (html) this.menuEl.innerHTML = `<div style="background:rgba(10,12,20,.95);padding:22px 28px;border-radius:12px;max-width:640px;line-height:1.6">${html}</div>`;
  }
}
