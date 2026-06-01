#!/usr/bin/env python
#
# Materialize a canonical 1600-wide copy of the CARLA-NVS SSE test sequences so
# that Nerfacto and 3DGS evaluate on byte-identical ground-truth images.
#
# 3DGS auto-downscales images wider than 1600px to 1600px (global_down =
# width/1600); for the CARLA 1920x1080 frames that is exactly 1600x900. nerfstudio
# can only downscale by *integer* factors, so instead of relying on each tool's
# internal resize we pre-resize ONCE here (images + COLMAP intrinsics) and point
# both pipelines at the result:
#   * Nerfacto : run_nerfacto_sse.py --io_dir <out_root>  (downscale_factor 1)
#   * 3DGS     : gsnet/run_sse.py     --io_dir <out_root>  (width==1600 -> no
#                further resize, so it uses these files as-is)
#
# For each <id>_base under --io_dir we write <out_root>/<id>_base with:
#   images/        : every frame resized to (target_width x scaled_height), LANCZOS
#   sparse/0/      : cameras.bin with focal+principal scaled by s=target_width/width
#                    and width/height updated; images.bin & points3D.* copied
#                    unchanged (poses/3D points are resolution-independent);
#                    test.txt copied through.
#
# Usage:
#   python -m gsnet_baseline.prep_1600 \
#       --io_dir   /mnt/zihanw/carla/input_output \
#       --out_root /mnt/zihanw/carla/input_output_1600
#
# (run inside the nerfstudio conda env; uses nerfstudio's colmap utils + PIL)

import argparse
import os
import shutil

import numpy as np
from PIL import Image

from nerfstudio.data.utils.colmap_parsing_utils import (
    Camera,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)

# COLMAP models we support (CARLA COLMAP is undistorted PINHOLE / SIMPLE_PINHOLE,
# matching gsnet/scene/dataset_readers.py which asserts exactly these two).
# Value = number of leading focal+principal-point params to scale by s.
SCALE_PARAMS = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4}


def scale_camera(cam: Camera, target_width: int):
    assert cam.model in SCALE_PARAMS, (
        f"camera model {cam.model} not handled (only PINHOLE / SIMPLE_PINHOLE, "
        "matching 3DGS)")
    s = target_width / cam.width
    new_w = target_width
    new_h = int(round(cam.height * s))
    n = SCALE_PARAMS[cam.model]
    params = np.array(cam.params, dtype=np.float64).copy()
    params[:n] = params[:n] * s
    return Camera(id=cam.id, model=cam.model, width=new_w, height=new_h,
                  params=params), s, (new_w, new_h)


def prep_sequence(seq_in, seq_out, target_width, resample):
    sp_in = os.path.join(seq_in, "sparse", "0")
    sp_out = os.path.join(seq_out, "sparse", "0")
    img_in = os.path.join(seq_in, "images")
    img_out = os.path.join(seq_out, "images")
    os.makedirs(sp_out, exist_ok=True)
    os.makedirs(img_out, exist_ok=True)

    cameras = read_cameras_binary(os.path.join(sp_in, "cameras.bin"))
    images = read_images_binary(os.path.join(sp_in, "images.bin"))

    new_cameras = {}
    cam_size = {}
    for cid, cam in cameras.items():
        new_cam, s, (w, h) = scale_camera(cam, target_width)
        new_cameras[cid] = new_cam
        cam_size[cid] = (w, h)
    write_cameras_binary(new_cameras, os.path.join(sp_out, "cameras.bin"))
    # poses (images.bin) and 3D points are resolution-independent -> copy as-is
    write_images_binary(images, os.path.join(sp_out, "images.bin"))
    for fn in ("points3D.bin", "points3D.ply", "points3D.txt", "test.txt"):
        src = os.path.join(sp_in, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(sp_out, fn))

    # resize each frame to its camera's target size
    n_imgs = 0
    for img in images.values():
        w, h = cam_size[img.camera_id]
        src = os.path.join(img_in, img.name)
        dst = os.path.join(img_out, img.name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB").resize((w, h), resample)
            im.save(dst)
        n_imgs += 1
    print(f"[prep] {seq_out}: {len(new_cameras)} cam(s) -> {target_width}px, "
          f"{n_imgs} images resized")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--io_dir", required=True,
                    help="dir holding the original <id>_base workspaces")
    ap.add_argument("--out_root", required=True,
                    help="output dir for the resized <id>_base workspaces")
    ap.add_argument("--test_ids", nargs="+",
                    default=["110", "210", "310", "410", "510"])
    ap.add_argument("--target_width", type=int, default=1600)
    args = ap.parse_args()

    resample = Image.LANCZOS
    for sid in args.test_ids:
        seq_in = os.path.join(args.io_dir, f"{sid}_base")
        seq_out = os.path.join(args.out_root, f"{sid}_base")
        assert os.path.isdir(seq_in), f"missing {seq_in}"
        prep_sequence(seq_in, seq_out, args.target_width, resample)
    print(f"\nDone. Point both pipelines at: {args.out_root}")


if __name__ == "__main__":
    main()
