/** Light list (§3.9): lamps are static point lights; flares travel along geodesics (§4 Tools). */
import { type Kappa, type V4, geodesic, geodesicDir, project, tangentialize } from "../geometry/space";

export interface Light {
  id: number;
  pos: V4; // world model coords (home domain)
  color: [number, number, number]; // colour · power (W)
  kind: "lamp" | "flare";
  /** flares: geodesic motion state */
  dir?: V4;
  speed?: number; // model units / s
  ttl?: number; // seconds
}

export class Lights {
  list: Light[] = [];
  private nextId = 1;
  maxLights = 8;

  addLamp(pos: V4, color: [number, number, number]): Light {
    const l: Light = { id: this.nextId++, pos, color, kind: "lamp" };
    this.list.push(l);
    if (this.list.length > this.maxLights) this.list.shift();
    return l;
  }

  throwFlare(pos: V4, dir: V4, speed: number, color: [number, number, number] = [6, 3.5, 1.5], ttl = 12): Light {
    const l: Light = { id: this.nextId++, pos, dir, speed, ttl, color, kind: "flare" };
    this.list.push(l);
    if (this.list.length > this.maxLights) this.list.shift();
    return l;
  }

  /** Advance flares along their geodesics; `blocked(x)` stops them at dense foam. Returns ids removed. */
  update(kappa: Kappa, dt: number, blocked: (x: V4) => boolean): number[] {
    const removed: number[] = [];
    for (const l of this.list) {
      if (l.kind !== "flare") continue;
      l.ttl! -= dt;
      if (l.speed! > 0) {
        const step = l.speed! * dt;
        const nx = geodesic(kappa, l.pos, l.dir!, step);
        const nd = geodesicDir(kappa, l.pos, l.dir!, step);
        if (blocked(nx)) { l.speed = 0; } else { l.pos = project(kappa, nx); l.dir = tangentialize(kappa, l.pos, nd); }
      }
      // flicker
      const f = 0.85 + 0.15 * Math.sin(performance.now() * 0.03 + l.id);
      l.color = [6 * f, 3.5 * f, 1.5 * f];
    }
    for (const l of this.list.filter((x) => x.kind === "flare" && x.ttl! <= 0)) { removed.push(l.id); }
    this.list = this.list.filter((x) => !(x.kind === "flare" && x.ttl! <= 0));
    return removed;
  }
}
