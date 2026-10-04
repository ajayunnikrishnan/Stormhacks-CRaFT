# Training a web-budget Power Foam scene on Vast.ai

Goal: one trained checkpoint (`model.pt` + `config.yaml`) at the web budget, plus reference images
from Power Foam's own ray tracer. About 45 minutes wall clock, a few dollars.

## 1. Rent the machine (5 min)

1. vast.ai → Console → Templates → pick **PyTorch (CUDA 12.x)** (any `pytorch/pytorch:2.x-cuda12.8-cudnn9-devel` image is fine).
2. Filters: GPU **RTX 4090** (or 3090 / A5000; anything ≥ 24 GB and compute capability ≥ 7.0),
   **disk ≥ 60 GB**, CUDA version ≥ 12.4, high internet speed. On-demand, not interruptible.
3. Under "Launch mode" choose **SSH**. Add your SSH public key in Account → SSH keys first
   (`cat ~/.ssh/id_ed25519.pub` on the Mac; create one with `ssh-keygen -t ed25519` if missing).
4. Rent. When it shows "running", copy the SSH command from the instance card
   (`ssh -p <port> root@<ip>`).

## 2. Set up Power Foam (10 min)

Paste these on the instance:

```bash
apt-get update && apt-get install -y git unzip wget
git clone https://github.com/theialab/powerfoam.git && cd powerfoam
python --version          # needs 3.10 or 3.11; the PyTorch images ship 3.11
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import warp as wp; wp.init(); import torch; print(torch.cuda.get_device_name(0))"
```

If the last line prints the GPU name, you're good. (If pip complains about `open3d`, run
`pip install -r requirements.txt --no-deps` then `pip install open3d` separately; it is only
used for mesh export.)

## 3. Get the data (5–10 min; 12 GB download in the datacentre)

```bash
mkdir -p data/mipnerf360 && cd data/mipnerf360
wget -q --show-progress http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -q 360_v2.zip 'room/*'          # or 'bonsai/*', 'kitchen/*', 'counter/*'
rm 360_v2.zip && ls room               # expect: images images_2 images_4 images_8 sparse poses_bounds.npy
cd ../..
```

`room` is the easiest to walk around in. Bonsai is the scene the paper benchmarks most.

## 4. Train at the web budget (~30 min on a 4090)

```bash
python train.py -c configs/mipnerf360_indoor.yaml --scene room \
  --final_points 150000 --num_texel_sites 4 --sv_dof 4 --experiment_name room_150k_k4_d4
```

Progress prints every iteration; previews land in `output/room_150k_k4_d4/test/`. If you have time,
also start the variant with 8 SV axes (the paper never ablates this number):

```bash
python train.py -c configs/mipnerf360_indoor.yaml --scene room \
  --final_points 150000 --num_texel_sites 4 --sv_dof 8 --experiment_name room_150k_k4_d8
```

## 5. Reference images from Power Foam's own ray tracer (2 min)

```bash
wget -q https://raw.githubusercontent.com/<your-github>/stormhacks/main/tools/render_reference.py -O render_reference.py \
  || scp -P <port> ~/projects/active-projects/stormhacks/tools/render_reference.py root@<ip>:~/powerfoam/   # run this one from the Mac instead if the repo is not public
python render_reference.py -c output/room_150k_k4_d4/config.yaml --out output/room_150k_k4_d4/refs --n 4
```

## 6. Copy back to the Mac (from the Mac)

```bash
mkdir -p ~/projects/active-projects/stormhacks/scenes
scp -P <port> -r root@<ip>:~/powerfoam/output/room_150k_k4_d4 ~/projects/active-projects/stormhacks/scenes/
```

Only `model.pt`, `config.yaml` and `refs/` are needed (skip `test/` previews if you want it faster).

## 7. Destroy the instance

Vast bills until you click **Destroy** on the instance card. Do that last.

Then tell Claude the folder name; export, retuning and the PSNR check are automated from there.
