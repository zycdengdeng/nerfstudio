# Nerfacto baseline (SSE) — fair comparison with 3DGS / GS-Net

Nerfacto is a **per-scene** baseline for the GS-Net experiments. To make its
numbers directly comparable to the 3DGS / GS-Net SSE table
(`gsnet/run_sse.py`), this driver reuses **the same data, the same train/test
split, and the same metric code**.

## What "fair" means here

| | 3DGS / GS-Net (`gsnet/run_sse.py`) | Nerfacto (`run_nerfacto_sse.py`) |
|---|---|---|
| Test scenes | `110 210 310 410 510` (seq-10 of each of the 5 CARLA-NVS scenes) | identical |
| Data | `<io_dir>/<id>_base/{images,sparse/0}` | identical |
| Split | hold out frames `{4,9}` per block of 10 → **12 test / 48 train**, via `sparse/0/test.txt` | identical split, via nerfstudio `train_list.txt` / `val_list.txt` / `test_list.txt` |
| Resolution | **1600×900** (3DGS auto-caps width at 1600) | **1600×900** — both pipelines read the *same* pre-resized dataset (see `prep_1600.py`) |
| Iterations | 30000 | 30000 |
| Metrics | `gsnet/metrics.py` — PSNR / SSIM / LPIPS(**vgg**) | **the same `gsnet/metrics.py`**, run on Nerfacto's rendered test views |
| Timing | optim seconds logged | train + render seconds logged |

NeRF is optimized per scene, so the 5 SSE *test* sequences `110…510` are exactly
the right baseline set: same scenes, same held-out views, same scoring. The
GS-Net training sequences (`seq 1–9`) are irrelevant to Nerfacto — Nerfacto does
not generalize across scenes, it just fits each test scene directly.

## Environment

Nerfacto needs its **own** conda env (separate from the 3DGS / `gsnet` env). We
intentionally do **not** hard-code a CUDA version — pick the toolkit that
matches your driver and the torch build. nerfstudio 1.1.5 (this repo) is known
good with Python 3.8 + torch 2.1.x. Fill in `<CUXXX>` after checking the server
(see the chat for the exact diagnostic commands).

```bash
conda create -n nerfacto -y python=3.8
conda activate nerfacto
pip install --upgrade pip setuptools

# torch matching the chosen CUDA, e.g. cu118:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/<CUXXX>

# CUDA toolkit for building tiny-cuda-nn (match <CUXXX>):
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# nerfstudio itself (editable, from this repo root):
cd /path/to/nerfstudio && pip install -e .
ns-install-cli            # optional shell completion
```

Sanity check:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
ns-train --help | head
```

## Step 0 — build the canonical 1600×900 dataset (once)

Both pipelines must read identical GT. We pre-resize the 5 test sequences to
1600×900 (images + COLMAP intrinsics) so neither tool's internal resize is
involved:

```bash
python -m gsnet_baseline.prep_1600 \
    --io_dir   /mnt/zihanw/carla/input_output \
    --out_root /mnt/zihanw/carla/input_output_1600
```

Then point **3DGS** at the same `--io_dir /mnt/zihanw/carla/input_output_1600`
(width is already 1600, so 3DGS does no further resize).

## Run

From the **nerfstudio repo root** (`/mnt/zihanw/nerfstudio`), inside the
`nerfacto` env:

```bash
python -m gsnet_baseline.run_nerfacto_sse \
    --io_dir      /mnt/zihanw/carla/input_output_1600 \
    --gsnet_repo  /mnt/zihanw/gaussian-splatting \
    --gsnet_python /home/wzh/miniconda3/envs/gaussian_splatting/bin/python \
    --out_dir     runs/nerfacto_sse \
    --gpus 4 5 6 7
```

* `--gsnet_python` is the python of the **3DGS env** — `metrics.py` is run with
  it so PSNR/SSIM/LPIPS use byte-for-byte the same implementation as the 3DGS
  table.
* Jobs (one per test scene) are distributed one-per-GPU; the run resumes/merges
  if re-invoked.

### Outputs (per scene `<id>`, under `--out_dir`)

```
<out_dir>/<id>/train/<id>/nerfacto/run/    # nerfstudio training run + config.yml
<out_dir>/<id>/render/test/{rgb,gt-rgb}/   # rendered test views + GT
<out_dir>/<id>/test/ours_nerfacto/{renders,gt}/   # 3DGS-style layout for metrics.py
<out_dir>/<id>/results.json                # PSNR/SSIM/LPIPS from gsnet/metrics.py
<out_dir>/nerfacto_sse_results.{json,md}   # aggregated table + timings
```

The Markdown table mirrors `gsnet/runs/sse/sse_results.md`, so the Nerfacto row
can be dropped straight into the comparison.

## Notes / knobs

* `--downscale_factor 1` keeps the pre-resized 1600×900 images as-is. Always run
  the driver against the `*_1600` dataset produced by `prep_1600.py`.
* SSE split parameters (`--num_images 60 --block 10 --holdout 4 9 --ext .png`)
  mirror `gsnet/make_sse_split.py`; change them only if the 3DGS split changed.
* `--method` defaults to `nerfacto`; `nerfacto-big` / `depth-nerfacto` etc. can
  be swapped in for ablations.
