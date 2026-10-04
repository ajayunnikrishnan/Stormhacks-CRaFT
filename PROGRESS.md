# PROGRESS

## Phase 0 — Recon (read-only) — DONE, awaiting approval
- Read paper §3, App. B–C; read `powerfoam/` (raytrace, rendering_math, rasterize
  benchmark/visualize kernels, color_fn, scene, texture, bvh, camera, benchmark.py,
  train/test/view/render).
- Wrote `docs/ARCHITECTURE.md` (checkpoint format, exact cell math, walk order of
  operations, adjacency + Steiner as implemented, flat-space assumption list,
  bytes/cell, web budget, prompt-vs-repo discrepancies, open questions).
- Power Foam pinned as git submodule `powerfoam/` @ 9639225.

## Next
- Get answers to ARCHITECTURE.md §12 (GPU box, scene path, target GPU, hours/team).
- Phase 1: exporter + flat GLSL walk + PSNR harness vs `RayTracer.benchmark`.

## Known issues
- No CUDA on the dev Mac: training and reference renders must run elsewhere.
- `{{SCENE_PATH}}`, `{{TARGET_GPU_AND_OS}}`, `{{HOURS}}`, `{{TEAM_SIZE}}` unfilled.

## Current FPS
- n/a (no renderer yet)
