#!/usr/bin/env python
#
# Nerfacto SSE baseline driver, multi-GPU.
#
# This mirrors gsnet/run_sse.py (the 3DGS SSE driver) so that the Nerfacto
# numbers are *directly* comparable to the 3DGS / GS-Net numbers:
#   * SAME data            : the per-sequence CARLA-NVS COLMAP workspaces
#                            <io_dir>/<id>_base/{images,sparse/0}
#   * SAME train/test split: 60 imgs = 6 cams x 10 frames; hold out frames
#                            {4,9} within each block of 10 -> 12 test / 48 train.
#                            Written as nerfstudio split-list files
#                            (train_list.txt / val_list.txt / test_list.txt) so
#                            the ColmapDataParser uses EXACTLY those frames.
#   * SAME metric code     : after training, render the 12 test views + GT and
#                            score them with gsnet/metrics.py (the identical
#                            PSNR / SSIM / LPIPS-vgg implementation used for 3DGS).
#
# Per scene (one independent job, distributed one-per-GPU across --gpus):
#   1. write the SSE split-list files into <io_dir>/<id>_base
#   2. ns-train nerfacto on <id>_base (colmap dataparser, downscale 1)   [timed]
#   3. ns-render dataset --split test  (rgb + gt-rgb for the 12 test views) [timed]
#   4. reorganize renders/gt into the 3DGS model layout
#        <out_dir>/<id>/test/ours_nerfacto/{renders,gt}
#   5. score with gsnet/metrics.py  ->  <out_dir>/<id>/results.json
#
# All wall-clock times (train + render) and PSNR/SSIM/LPIPS are recorded to
# <out_dir>/nerfacto_sse_results.{json,md}; the summary is refreshed as each job
# finishes (and resumes/merges across invocations).
#
# Usage (4 A100s on cards 4,5,6,7):
#   python -m gsnet_baseline.run_nerfacto_sse \
#       --io_dir /mnt/zihanw/carla/input_output \
#       --gsnet_repo /home/.../gsnet \
#       --gsnet_python /home/.../miniconda3/envs/gsnet/bin/python \
#       --out_dir runs/nerfacto_sse --gpus 4 5 6 7
#
# (run from the nerfstudio repo root, inside the nerfstudio conda env)

import argparse
import glob
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = "ours_nerfacto"  # method label under <model>/test/ (3DGS layout)


def run(cmd, gpu=None, cwd=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    tag = f"[gpu{gpu}] " if gpu is not None else ""
    print(f"\n{tag}$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True, env=env, cwd=cwd)
    return time.time() - t0


# ---------------------------------------------------------------------------
# train/test split -> nerfstudio split-list files
# ---------------------------------------------------------------------------
def sse_test_names(num_images=60, block=10, holdout=(4, 9), ext=".png"):
    """The 12 held-out test frames (identical rule to gsnet/make_sse_split.py)."""
    holdout0 = set(h - 1 for h in holdout)
    return [f"{n}{ext}" for n in range(1, num_images + 1)
            if (n - 1) % block in holdout0]


def write_split_lists(seq_dir, colmap_path="sparse/0", images_path="images",
                      num_images=60, block=10, holdout=(4, 9), ext=".png"):
    """Write train_list.txt / val_list.txt / test_list.txt for the ColmapDataParser.

    The test split is taken **verbatim from ``<colmap_path>/test.txt``** when it
    exists, so the split is byte-identical to the one 3DGS uses (true for both
    the SSE ``<id>_base`` workspaces and the CSE scenes, which both ship a
    ``test.txt``). Only if no test.txt is present do we fall back to the SSE
    block rule. Train = every other image present in ``images/``.

    Filenames are bare (relative to ``images/``); val == test so nerfstudio's
    during-training 'val' split and the final 'test' split evaluate the same
    held-out views.
    """
    img_dir = os.path.join(seq_dir, images_path)
    all_names = sorted(f for f in os.listdir(img_dir) if f.endswith(ext))

    test_txt = os.path.join(seq_dir, colmap_path, "test.txt")
    if os.path.exists(test_txt):
        with open(test_txt) as f:
            test_names = [ln.strip() for ln in f if ln.strip()]
        src = "test.txt"
    else:
        test_set = set(sse_test_names(num_images, block, holdout, ext))
        test_names = [n for n in all_names if n in test_set]
        src = "SSE block rule"

    test_set = set(test_names)
    missing = test_set.difference(all_names)
    assert not missing, f"test.txt names not found in {img_dir}: {sorted(missing)}"
    train_names = [n for n in all_names if n not in test_set]
    test_sorted = [n for n in all_names if n in test_set]

    def _write(name, names):
        with open(os.path.join(seq_dir, name), "w") as f:
            f.write("\n".join(names) + "\n")

    _write("train_list.txt", train_names)
    _write("val_list.txt", test_sorted)
    _write("test_list.txt", test_sorted)
    print(f"[split:{src}] {seq_dir}: {len(train_names)} train / "
          f"{len(test_sorted)} test")


# ---------------------------------------------------------------------------
# per-scene job
# ---------------------------------------------------------------------------
def find_config(train_out):
    hits = sorted(glob.glob(os.path.join(train_out, "**", "config.yml"), recursive=True))
    assert hits, f"no config.yml produced under {train_out}"
    return hits[-1]


def reorganize_for_metrics(render_out, model_path, ext=".png"):
    """Lay rendered rgb / gt-rgb out as <model>/test/<METHOD>/{renders,gt}."""
    src_render = os.path.join(render_out, "test", "rgb")
    src_gt = os.path.join(render_out, "test", "gt-rgb")
    assert os.path.isdir(src_render), f"missing renders {src_render}"
    assert os.path.isdir(src_gt), f"missing gt {src_gt}"
    dst = os.path.join(model_path, "test", METHOD)
    for sub, src in (("renders", src_render), ("gt", src_gt)):
        d = os.path.join(dst, sub)
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(src):
            if fn.endswith(ext):
                shutil.copy2(os.path.join(src, fn), os.path.join(d, fn))


def read_results(model_path):
    with open(os.path.join(model_path, "results.json")) as f:
        d = json.load(f)
    method = sorted(d.keys())[-1]
    return {k: d[method][k] for k in ("PSNR", "SSIM", "LPIPS")}


def job_nerfacto(sid, args, gpu):
    seq_dir = os.path.join(args.io_dir, f"{sid}{args.seq_suffix}")
    model_path = os.path.join(args.out_dir, sid)
    train_out = os.path.join(model_path, "train")
    render_out = os.path.join(model_path, "render")
    os.makedirs(model_path, exist_ok=True)

    # 2. train nerfacto (eval driven by the split-list files written up front)
    train_s = run([
        "ns-train", args.method,
        "--data", seq_dir,
        "--output-dir", train_out,
        "--experiment-name", sid,
        "--timestamp", "run",
        "--vis", "tensorboard",
        "--max-num-iterations", str(args.iterations),
        "--pipeline.model.predict-normals", "False",
        "colmap",
        "--colmap-path", args.colmap_path,
        "--images-path", args.images_path,
        "--downscale-factor", str(args.downscale_factor),
    ], gpu=gpu)

    config = find_config(train_out)

    # 3. render the 12 test views + GT
    if os.path.exists(render_out):
        shutil.rmtree(render_out)
    render_s = run([
        "ns-render", "dataset",
        "--load-config", config,
        "--output-path", render_out,
        "--split", "test",
        "--image-format", "png",          # lossless: keep GT byte-exact (no JPEG)
        "--rendered-output-names", "rgb", "gt-rgb",
    ], gpu=gpu)

    # 4. reorganize into the 3DGS model layout
    reorganize_for_metrics(render_out, model_path, ext=args.ext)

    # 5. score with the IDENTICAL gsnet metrics.py, run inside the gsnet repo
    #    (its package-relative imports: utils/, lpipsPyTorch/) using the gsnet
    #    conda env's python (torch + lpips-vgg). Model path is absolute.
    run([args.gsnet_python, "metrics.py", "-m", os.path.abspath(model_path)],
        gpu=gpu, cwd=args.gsnet_repo)

    metrics = read_results(model_path)
    return {**metrics, "train_seconds": train_s, "render_seconds": render_s,
            "total_seconds": train_s + render_s}


# ---------------------------------------------------------------------------
# summary table (mirrors gsnet/run_sse.py)
# ---------------------------------------------------------------------------
def summarize(records, out_dir, method_label, tag="sse"):
    def avg(key):
        vals = [r[method_label][key] for r in records if method_label in r]
        return sum(vals) / len(vals) if vals else float("nan")

    summary = {"per_sequence": records, "averages": {}}
    if any(method_label in r for r in records):
        summary["averages"][method_label] = {
            k: avg(k) for k in ("PSNR", "SSIM", "LPIPS",
                                "train_seconds", "render_seconds", "total_seconds")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"nerfacto_{tag}_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["", "| Seq | Method | PSNR | SSIM | LPIPS | Train(min) | Render(min) | Total(min) |",
             "|-----|--------|------|------|-------|------------|-------------|------------|"]
    for r in sorted(records, key=lambda x: x["id"]):
        if method_label in r:
            m = r[method_label]
            lines.append(
                f"| {r['id']} | {method_label} | {m['PSNR']:.2f} | {m['SSIM']:.3f} | "
                f"{m['LPIPS']:.3f} | {m['train_seconds']/60:.1f} | "
                f"{m['render_seconds']/60:.1f} | {m['total_seconds']/60:.1f} |")
    if method_label in summary["averages"]:
        a = summary["averages"][method_label]
        lines.append(
            f"| **Avg** | **{method_label}** | **{a['PSNR']:.2f}** | **{a['SSIM']:.3f}** | "
            f"**{a['LPIPS']:.3f}** | {a['train_seconds']/60:.1f} | "
            f"{a['render_seconds']/60:.1f} | {a['total_seconds']/60:.1f} |")
    table = "\n".join(lines)
    with open(os.path.join(out_dir, f"nerfacto_{tag}_results.md"), "w") as f:
        f.write(table + "\n")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True,
                    help="dir holding <id>_base sequence COLMAP workspaces")
    ap.add_argument("--gsnet_repo", required=True,
                    help="path to the gsnet repo (for metrics.py)")
    ap.add_argument("--gsnet_python", required=True,
                    help="python of the gsnet/3DGS conda env (has torch+lpips)")
    ap.add_argument("--out_dir", default="runs/nerfacto_sse")
    ap.add_argument("--method", default="nerfacto")
    ap.add_argument("--test_ids", nargs="+",
                    default=["110", "210", "310", "410", "510"])
    ap.add_argument("--gpus", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--iterations", type=int, default=30000)
    ap.add_argument("--colmap_path", default="sparse/0")
    ap.add_argument("--images_path", default="images")
    ap.add_argument("--downscale_factor", type=int, default=1)
    # seq dir = <io_dir>/<id><seq_suffix> (SSE: "_base"; CSE scenes: "")
    ap.add_argument("--seq_suffix", default="_base")
    # tag for the aggregated result files: nerfacto_<tag>_results.{json,md}
    ap.add_argument("--tag", default="sse")
    # split fallback parameters (only used when no sparse/0/test.txt exists;
    # test.txt — present for both SSE and CSE — always takes precedence)
    ap.add_argument("--num_images", type=int, default=60)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--holdout", type=int, nargs="+", default=[4, 9])
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    # Validate inputs and write split-list files up front.
    for sid in args.test_ids:
        seq_dir = os.path.join(args.io_dir, f"{sid}{args.seq_suffix}")
        assert os.path.isdir(seq_dir), f"missing test seq dir {seq_dir}"
        write_split_lists(seq_dir, args.colmap_path, args.images_path,
                          args.num_images, args.block,
                          tuple(args.holdout), args.ext)

    # Resume/merge previous results.
    records = {}
    results_path = os.path.join(args.out_dir, f"nerfacto_{args.tag}_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            for r in json.load(f).get("per_sequence", []):
                records[r["id"]] = r
        print(f"[resume] loaded {len(records)} existing sequence records")

    lock = threading.Lock()
    gpu_q = queue.Queue()
    for g in args.gpus:
        gpu_q.put(g)

    def worker(sid):
        gpu = gpu_q.get()
        try:
            t0 = time.time()
            res = job_nerfacto(sid, args, gpu)
            with lock:
                rec = records.setdefault(sid, {"id": sid, "scene": int(sid) // 100})
                rec[METHOD] = res
                table = summarize(list(records.values()), args.out_dir, METHOD, args.tag)
            print(f"\n[done] {sid} on gpu{gpu} in {time.time()-t0:.1f}s :: "
                  f"PSNR={res['PSNR']:.2f}\n{table}", flush=True)
        except Exception as e:  # one scene failing must not kill the others
            print(f"\n[FAILED] {sid} on gpu{gpu}: {e}", flush=True)
            return sid
        finally:
            gpu_q.put(gpu)
        return None

    import concurrent.futures as cf
    failed = []
    with cf.ThreadPoolExecutor(max_workers=len(args.gpus)) as ex:
        futs = [ex.submit(worker, sid) for sid in args.test_ids]
        for f in cf.as_completed(futs):
            bad = f.result()
            if bad:
                failed.append(bad)

    print("\n" + summarize(list(records.values()), args.out_dir, METHOD, args.tag))
    print(f"\nResults -> {args.out_dir}/nerfacto_{args.tag}_results.json , .md")
    if failed:
        print(f"\n[!] {len(failed)} scene(s) FAILED: {sorted(failed)} — "
              f"re-run the same command to retry just these (others are cached).")


if __name__ == "__main__":
    main()
