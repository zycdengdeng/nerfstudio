# Nerfacto baseline results (CARLA-NVS)

Per-scene Nerfacto baseline on the 5 CARLA-NVS test sequences
(`110/210/310/410/510`), produced with `run_nerfacto_sse.py`. By construction
these are directly comparable to the 3DGS / GS-Net tables:

* **Same data** — the per-sequence COLMAP workspaces.
* **Same resolution** — 1600×900 (the canonical pre-resized datasets from
  `prep_1600.py` / `prep_cse_1600.py`; both Nerfacto and 3DGS read identical GT).
* **Same train/test split** — taken verbatim from each scene's
  `sparse/0/test.txt` (the split 3DGS uses).
* **Same metrics** — scored with `gsnet/metrics.py` (PSNR / SSIM / LPIPS-vgg),
  the identical implementation used for the 3DGS numbers.
* **Iterations** — 30000.

Method: `nerfacto` (nerfstudio v1.1.5). nerfacto does **not** use the SfM point
cloud (`--load-3D-points False`); geometry is learned from poses + images.

## SSE — Same-Sensor Evaluation (48 train / 12 test per scene)

| Seq | PSNR | SSIM | LPIPS | Train(min) | Render(min) | Total(min) |
|-----|------|------|-------|------------|-------------|------------|
| 110 | 24.58 | 0.865 | 0.190 | 31.3 | 0.8 | 32.2 |
| 210 | 25.55 | 0.911 | 0.156 | 25.4 | 0.8 | 26.2 |
| 310 | 25.96 | 0.885 | 0.176 | 29.3 | 0.9 | 30.2 |
| 410 | 22.37 | 0.841 | 0.207 | 26.2 | 0.7 | 26.9 |
| 510 | 24.15 | 0.815 | 0.228 | 29.3 | 0.7 | 30.1 |
| **Avg** | **24.52** | **0.863** | **0.192** | **28.3** | **0.8** | **29.1** |

## CSE — Cross-Sensor Evaluation (60 odd train / 60 even test per scene)

Reconstruct on the 60 odd cameras; evaluate on the 60 even cameras (sensor
positions absent during reconstruction).

| Seq | PSNR | SSIM | LPIPS | Train(min) | Render(min) | Total(min) |
|-----|------|------|-------|------------|-------------|------------|
| 110 | 19.09 | 0.718 | 0.312 | 43.5 | 3.5 | 47.0 |
| 210 | 20.83 | 0.745 | 0.241 | 44.1 | 3.5 | 47.5 |
| 310 | 20.52 | 0.728 | 0.268 | 60.4 | 4.4 | 64.8 |
| 410 | 20.28 | 0.732 | 0.291 | 59.8 | 3.7 | 63.4 |
| 510 | 20.97 | 0.722 | 0.272 | 47.4 | 3.8 | 51.2 |
| **Avg** | **20.34** | **0.729** | **0.277** | **51.0** | **3.8** | **54.8** |

> Train times include GPU-contention overhead on a shared 8×A100 node; treat the
> per-scene wall-clock as indicative rather than a controlled timing benchmark.
> The machine-readable records (with per-view metrics) live on the server under
> `runs/nerfacto_sse/` and `runs/nerfacto_cse/` (`*_results.json`).
