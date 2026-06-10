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

Scene labels: **s1 = 110, s2 = 210, s3 = 310, s4 = 410, s5 = 510.**

## SSE — Same-Sensor Evaluation (48 train / 12 test per scene)

| Metric | s1 | s2 | s3 | s4 | s5 | Average |
|--------|------|------|------|------|------|---------|
| PSNR↑       | 24.58 | 25.55 | 25.96 | 22.37 | 24.15 | 24.52 |
| SSIM↑       | 0.865 | 0.911 | 0.885 | 0.841 | 0.815 | 0.863 |
| LPIPS↓      | 0.190 | 0.156 | 0.176 | 0.207 | 0.228 | 0.192 |

## CSE — Cross-Sensor Evaluation (60 odd train / 60 even test per scene)

Reconstruct on the 60 odd cameras; evaluate on the 60 even cameras (sensor
positions absent during reconstruction).

| Metric | s1 | s2 | s3 | s4 | s5 | Average |
|--------|------|------|------|------|------|---------|
| PSNR↑       | 19.09 | 20.83 | 20.52 | 20.28 | 20.97 | 20.34 |
| SSIM↑       | 0.718 | 0.745 | 0.728 | 0.732 | 0.722 | 0.729 |
| LPIPS↓      | 0.312 | 0.241 | 0.268 | 0.291 | 0.272 | 0.277 |


> Train times include GPU-contention overhead on a shared 8×A100 node; treat the
> per-scene wall-clock as indicative rather than a controlled timing benchmark.
> The machine-readable records (with per-view metrics) live on the server under
> `runs/nerfacto_sse/` and `runs/nerfacto_cse/` (`*_results.json`).
