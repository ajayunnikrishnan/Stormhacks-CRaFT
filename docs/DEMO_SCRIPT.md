# CRaFT — demo script (2–3 minutes)

> Note: the tool-based levels described below were removed from the shipped build; the demo now walks the explorer: title → Explore → slider (flat → hyperbolic → spherical) → gallery (3-torus, {4,3,5}, Poincaré dodecahedral space, Klein) → enlarged map. Keep the narration about measurement as spoken explanation.

Open `web/` in Chrome (`npx vite`, then http://127.0.0.1:5173/?level=tutorial). Quality "auto".
Have the pause menu (Esc) ready for the controls.

**0:00 — "A normal room."** Start in level 1, flat. Walk a few steps (WASD), look around.
"This is CRaFT: a real captured scene — a Power Foam reconstruction — ray traced along exact geodesics on the GPU."
Press **B** three times while walking a short triangle. Point at the readout: *angles sum to
180.00°*. Press **P** for a lamp, **I** for the light meter, walk away: the dots sit on the grey
1/d² curve. "Flat. Nothing to see. Now watch."

**0:35 — Break it.** Press **N** → level 2 (hyperbolic). The far floor drops away, straight lines
bow. Press **L**: the laser is a visible geodesic. Place three beacons: the triangle now sums to
*less* than 180°. Walk a loop and look at the compass needle: it came back rotated — holonomy.
"Every one of these is κ times an area. The player can measure the shape of space."

**1:10 — The lamps are dying.** Place a lamp (**P**) and walk away with the meter on: the dots fall
*below* the 1/d² curve. "In hyperbolic space light spreads over 4π sinh² d. You must get closer
than you think." Walk up to the ★ marker, drop lamps until the goal turns green.

**1:40 — The light comes home.** **N** → level 3 (spherical). Everything looks magnified; walk and
the room curves back toward you. The marker floats out over the void, unreachable. Turn your back
on it and walk straight: half-way round the universe, place a lamp. The marker lights up from
across the universe — the lamp's light refocused at its antipode.

**2:10 — Topology.** Esc, choose Sandbox; press **G** for the gallery. Diagrams and thumbnails are
generated from the same face-pairing data the walker uses. Pick the **3-torus**: walk through a
wall, arrive from the opposite wall. Pick **{4,3,5}**: five cubes around every edge, copies of the
room exploding into the distance; walls tinted by face pair match the diagram. Pick **Poincaré
dodecahedral space** (36° twist) or **Klein space**: come back mirrored.

**2:40 — The exam.** Level 4: fog, dark, hidden curvature. "Measure it, dial it, press Enter." The
true value appears with a score, and the space morphs flat and back.

Close: "One geometry layer, κ ∈ {−1, 0, +1}, closed-form geodesic intersections, a captured scene,
and a player who can prove what shape their universe is."

Backup: the validation page `/harness.html` (PSNR vs reference, GLSL-vs-TS probe table) if a
judge asks about correctness.
