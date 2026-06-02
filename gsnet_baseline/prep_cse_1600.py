#!/usr/bin/env python
#
# Materialize a canonical 1600-wide copy of the gsnet CSE scenes (built by
# gsnet/make_cse_scene.py) so Nerfacto and 3DGS evaluate CSE on byte-identical
# 1600x900 ground truth — same rationale as prep_1600.py for the SSE
# <id>_base workspaces.
#
# A CSE scene <scenes_dir>/<id>/ is a *text* COLMAP workspace with a single
# shared PINHOLE intrinsic, 60 odd (train) + 60 even (test) poses in images.txt,
# symlinked images, and test.txt listing the even images. We:
#   * scale the single intrinsic by s = target_width/width, set width/height
#   * resize every (symlinked) image to (target_width x scaled_height), LANCZOS,
#     writing real PNGs into <out_root>/<id>/images/
#   * copy images.txt (poses are resolution-independent), points3D.*, test.txt
#
# Then run both pipelines against <out_root> (Nerfacto: run_nerfacto_sse.py
# --io_dir <out_root> --seq_suffix "" --tag cse; 3DGS: --scenes_dir <out_root>,
# width already 1600 -> no further resize).
#
# Usage:
#   python -m gsnet_baseline.prep_cse_1600 \
#       --scenes_dir /mnt/zihanw/gaussian-splatting/runs/cse_scenes \
#       --out_root   /mnt/zihanw/carla/cse_scenes_1600
#
# (run inside the nerfstudio conda env; uses nerfstudio's colmap utils + PIL)

import argparse
import os
import shutil

import numpy as np
from PIL import Image

from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_text,
    read_images_text,
    write_cameras_text,
    write_images_text,
)

SCALE_PARAMS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}


def prep_scene(sid, scenes_dir, out_root, target_width):
    sp_in = os.path.join(scenes_dir, sid, "sparse", "0")
    img_in = os.path.join(scenes_dir, sid, "images")
    sp_out = os.path.join(out_root, sid, "sparse", "0")
    img_out = os.path.join(out_root, sid, "images")
    os.makedirs(sp_out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    cameras = read_cameras_text(os.path.join(sp_in, "cameras.txt"))
    images = read_images_text(os.path.join(sp_in, "images.txt"))

    new_cams = {}
    cam_size = {}
    for cid, cam in cameras.items():
        assert cam.model in SCALE_PARAMS, f"unhandled camera model {cam.model}"
        s = target_width / cam.width
        nw, nh = target_width, int(round(cam.height * s))
        n = SCALE_PARAMS[cam.model]
        params = np.array(cam.params, dtype=np.float64).copy()
        params[:n] = params[:n] * s
        new_cams[cid] = cam._replace(width=nw, height=nh, params=params)
        cam_size[cid] = (nw, nh)
    write_cameras_text(new_cams, os.path.join(sp_out, "cameras.txt"))
    # poses & 3D points are resolution-independent -> copy through
    write_images_text(images, os.path.join(sp_out, "images.txt"))
    for fn in ("points3D.txt", "points3D.bin", "points3D.ply", "test.txt"):
        src = os.path.join(sp_in, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(sp_out, fn))

    n_imgs = 0
    for img in images.values():
        w, h = cam_size[img.camera_id]
        src = os.path.join(img_in, img.name)   # symlink -> real source
        dst = os.path.join(img_out, img.name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as im:
            im.convert("RGB").resize((w, h), Image.LANCZOS).save(dst)
        n_imgs += 1
    n_test = sum(1 for _ in open(os.path.join(sp_out, "test.txt"))) \
        if os.path.exists(os.path.join(sp_out, "test.txt")) else 0
    print(f"[cse-prep] {sid}: {len(images)} imgs -> {target_width}px "
          f"({n_imgs} resized, {n_test} test)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_dir", required=True,
                    help="dir holding the gsnet CSE scenes <id>/")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--ids", nargs="+",
                    default=["110", "210", "310", "410", "510"])
    ap.add_argument("--target_width", type=int, default=1600)
    args = ap.parse_args()
    for sid in args.ids:
        seq_in = os.path.join(args.scenes_dir, sid)
        assert os.path.isdir(seq_in), f"missing CSE scene {seq_in}"
        prep_scene(sid, args.scenes_dir, args.out_root, args.target_width)
    print(f"\nDone. CSE 1600 scenes -> {args.out_root}")


if __name__ == "__main__":
    main()
