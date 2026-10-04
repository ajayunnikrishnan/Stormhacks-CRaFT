"""Render ground-truth images with Power Foam's OWN ray tracer (the Warp kernel in
powerfoam/raytrace.py) for a trained checkpoint, plus the cameras, so the web renderer can be
compared against the real thing (Phase 1 acceptance: PSNR > 40 dB).

Run this INSIDE the Power Foam repo environment on a CUDA machine, from the repo root:

    cd powerfoam
    python /path/to/stormhacks/tools/render_reference.py -c output/<experiment>/config.yaml --out refs --n 4

It mirrors benchmark.py's `--render_type raytrace` path exactly (Steiner points, Morton sort,
regular-triangulation adjacency, fp16 adjacency differences, fp16 attributes) and writes:
    refs/cam_XXX.png   tone-free RGB (clamped to [0,1]) as rendered
    refs/cam_XXX.npy   float32 HxWx3, unclamped
    refs/cameras.json  [{name, eye, right, up, width, height}] in the repo's TorchCamera convention
Copy `refs/` next to the exported scene; `web/harness.html` picks it up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import warp as wp
from PIL import Image

sys.path.insert(0, os.getcwd())  # the Power Foam repo root
from configs import Params, add_group  # noqa: E402
from data_loader import DataHandler  # noqa: E402
from powerfoam.scene import PowerfoamScene  # noqa: E402
from benchmark import add_steiner_points, build_power_adjacency, get_steiner_points  # noqa: E402


def main(args, config_path, out_dir, n_cams, split):
    wp.init()
    checkpoint = config_path.replace("/config.yaml", "")
    os.makedirs(out_dir, exist_ok=True)

    data = DataHandler(args)
    data.reload(split, downsample=args.downsample[-1])

    model = PowerfoamScene(args, attr_dtype="half")
    model.initialize_from_dataset(data, device="cuda")
    model.load_pt(f"{checkpoint}/model.pt")
    model.declare_optimizers(args, args.iterations)
    model.sort_points()

    with torch.no_grad():
        steiner_points, steiner_radii = get_steiner_points(model.points, model.get_radii().to(torch.float32), data.cameras)
        add_steiner_points(model, steiner_points, steiner_radii, model.tscalar)
        model.sort_points()

        points = model.points
        radii = model.get_radii()
        adjacency, adjacency_offsets = build_power_adjacency(points, radii, alpha_complex=False)
        num_adjs = adjacency_offsets.diff()
        pm = 0.5 * (points.norm(dim=-1) ** 2 - radii**2)
        self_points = points.repeat_interleave(num_adjs, dim=0)
        adjacency_diff = points[adjacency] - self_points
        pm_diff = pm[adjacency] - pm.repeat_interleave(num_adjs, dim=0)
        adjacency_diff = torch.cat([adjacency_diff, pm_diff[:, None]], dim=-1).to(torch.float16)

        density = model.get_density()
        normals = model.get_normals()
        tangents, bitangent = model.get_tangents()
        offsets = model.texel_sites * radii[:, None, None]
        offsets = offsets[..., 0:1] * tangents[:, None, :] + offsets[..., 1:2] * bitangent[:, None, :]
        texel_sites = model.points[:, None, :] + offsets
        att_sites, att_values, att_temps = model.get_att_sv()
        texel_height = model.texel_height * radii[:, None]

        cams = data.cameras[: n_cams]
        meta = []
        for i, camera in enumerate(cams):
            eye = camera.eye.to(model.device)
            start = int(torch.argmin(torch.linalg.norm(points - eye[None, :], dim=-1) ** 2 - radii**2))
            texel_rgb = model.sv.forward(texel_sites.view(-1, 3).detach(), camera, att_sites, att_values, att_temps)
            texel_rgb = texel_rgb.view(model.points.shape[0], args.num_texel_sites, 3)
            rgb = model.raytracer.benchmark(camera, start, points, radii, density, normals, texel_sites, texel_rgb, texel_height, adjacency, adjacency_offsets, adjacency_diff, 1e-2)
            img = rgb.float().cpu().numpy()
            np.save(os.path.join(out_dir, f"cam_{i:03d}.npy"), img)
            Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(os.path.join(out_dir, f"cam_{i:03d}.png"))
            meta.append({
                "name": f"cam_{i:03d}", "eye": camera.eye.tolist(), "right": camera.right.tolist(), "up": camera.up.tolist(),
                "width": int(camera.width), "height": int(camera.height), "start_point_idx": start,
            })
            print(f"rendered {i}: {camera.width}x{camera.height}, start cell {start}")
        with open(os.path.join(out_dir, "cameras.json"), "w") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump({"n_points_with_steiner": int(points.shape[0]), "n_steiner": int(steiner_points.shape[0]), "n_edges_directed": int(adjacency.shape[0]), "split": split, "downsample": int(args.downsample[-1])}, f, indent=1)
    print("done:", out_dir)


if __name__ == "__main__":
    import configargparse

    parser = configargparse.ArgParser()
    get_params = add_group(parser, Params)
    parser.add_argument("-c", "--config", is_config_file=True, help="Path to config file")
    parser.add_argument("--out", default="refs")
    parser.add_argument("--n", type=int, default=4, help="number of cameras to render")
    parser.add_argument("--split", default="test", choices=["test", "train", "all"])
    a = parser.parse_args()
    main(get_params(a), a.config, a.out, a.n, a.split)
