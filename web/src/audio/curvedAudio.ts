/**
 * Curved-space spatial audio (§3.10): gain ∝ 1/sn_κ(d)², normalised at a reference distance,
 * panned by the direction of arrival in camera coordinates (Eq. 5 tangent). Lamps hum, flares
 * crackle. Web Audio only; created lazily on the first user gesture (browser policy).
 */
import { type Kappa, type V4, type M4, apply, sn, tangentToward } from "../geometry/space";

export interface AudioSource { id: number; pos: V4; kind: "hum" | "crackle"; node: AudioNode; gain: GainNode; pan: StereoPannerNode }

export class CurvedAudio {
  ctx: AudioContext | null = null;
  master: GainNode | null = null;
  sources = new Map<number, AudioSource>();
  refDistanceM = 1.5;
  enabled = true;

  ensure() {
    if (this.ctx) return;
    const AC = (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext);
    this.ctx = new AC();
    this.master = this.ctx.createGain();
    this.master.gain.value = 0.5;
    this.master.connect(this.ctx.destination);
  }

  addHum(id: number, pos: V4) {
    this.ensure();
    const ctx = this.ctx!;
    const osc = ctx.createOscillator();
    osc.type = "sawtooth";
    osc.frequency.value = 55 + (id % 3) * 5; // slightly detuned lamps
    const lp = ctx.createBiquadFilter();
    lp.type = "lowpass"; lp.frequency.value = 220; lp.Q.value = 4;
    const gain = ctx.createGain(); gain.gain.value = 0;
    const pan = ctx.createStereoPanner();
    osc.connect(lp); lp.connect(gain); gain.connect(pan); pan.connect(this.master!);
    osc.start();
    this.sources.set(id, { id, pos, kind: "hum", node: osc, gain, pan });
  }

  addCrackle(id: number, pos: V4) {
    this.ensure();
    const ctx = this.ctx!;
    const len = ctx.sampleRate * 2;
    const buf = ctx.createBuffer(1, len, ctx.sampleRate);
    const data = buf.getChannelData(0);
    for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (Math.random() < 0.02 ? 1 : 0.08);
    const src = ctx.createBufferSource(); src.buffer = buf; src.loop = true;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 1200;
    const gain = ctx.createGain(); gain.gain.value = 0;
    const pan = ctx.createStereoPanner();
    src.connect(hp); hp.connect(gain); gain.connect(pan); pan.connect(this.master!);
    src.start();
    this.sources.set(id, { id, pos, kind: "crackle", node: src, gain, pan });
  }

  remove(id: number) {
    const s = this.sources.get(id);
    if (!s) return;
    try { (s.node as OscillatorNode).stop(); } catch { /* buffer sources also stop */ }
    s.gain.disconnect(); s.pan.disconnect();
    this.sources.delete(id);
  }

  /** Update gains and pans for the listener at camWorld with world→camera isometry W. */
  update(kappa: Kappa, scale: number, camWorld: V4, W: M4) {
    if (!this.ctx) return;
    const ref = sn(kappa, this.refDistanceM * scale);
    const now = this.ctx.currentTime;
    for (const s of this.sources.values()) {
      const { u, d } = tangentToward(kappa, camWorld, s.pos);
      const sd = Math.max(1e-3, Math.abs(sn(kappa, d)));
      let g = this.enabled ? Math.min(1, (ref * ref) / (sd * sd)) : 0; // 1/sn² law, normalised at the reference distance
      if (kappa > 0) { const s2 = Math.max(1e-3, Math.abs(Math.sin(2 * Math.PI - d))); g = Math.min(1, g + (ref * ref) / (s2 * s2) * (this.enabled ? 1 : 0)); } // long way round
      const uc = apply(W, u); // arrival direction in camera coords
      const panv = Math.max(-1, Math.min(1, uc[1]));
      s.gain.gain.setTargetAtTime(g * (s.kind === "hum" ? 0.35 : 0.5), now, 0.05);
      s.pan.pan.setTargetAtTime(panv, now, 0.05);
    }
  }
}
