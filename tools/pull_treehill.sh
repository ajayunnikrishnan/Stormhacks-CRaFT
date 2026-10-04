#!/bin/bash
# Pull the trained treehill checkpoint from the Vast instance (ssh alias "vast"), canonicalise,
# export for the web, and place reference renders next to the export for web/harness.html.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
EXP=treehill_150k_k4_d4
RAW=scenes/treehill_raw
mkdir -p $RAW
rsync -az --info=progress2 -e ssh vast:/workspace/powerfoam/output/$EXP/{model.pt,config.yaml,refs,refs_train} $RAW/
du -sh $RAW
python tools/canonicalize_scene.py $RAW --out scenes/treehill \
  --cameras $RAW/refs_train/cameras_all.json --cam-height 1.5 \
  --crop-radius "${CROP:-12}" --crop-below 1.5 --crop-above 12
python tools/export_scene.py scenes/treehill --out web/public/scenes/treehill --curved --sweep 12 \
  --kmax "${KMAX:-0.015}" --steiner-box 2.0 --steiner-iters 14 --centre-y 1.6 \
  --walk-box -4 0 -4 4 3.2 4 ${VERIFY:+--verify}
mkdir -p web/public/scenes/treehill/refs && cp scenes/treehill/refs/*.png scenes/treehill/refs/cameras.json web/public/scenes/treehill/refs/
# raw float32 HxWx3 for web/harness.html?scene=scenes/treehill/scene.json&refs=scenes/treehill/refs
python - <<'PY'
import numpy as np, glob, os
for f in sorted(glob.glob("scenes/treehill/refs/cam_*.npy")):
    a = np.load(f).astype(np.float32); a.tofile("web/public/scenes/treehill/refs/" + os.path.basename(f).replace(".npy", ".f32")); print(f, a.shape)
PY
ls -la web/public/scenes/treehill
