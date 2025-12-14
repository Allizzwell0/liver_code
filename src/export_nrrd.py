#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 nnUNet 风格的 CT .b2nd + 预测 seg.npy (+ properties.pkl)
导出为 3D Slicer 可读的 NRRD 文件。

会生成两个文件：
  - <out_prefix>_ct.nrrd   : CT 体数据（float32）
  - <out_prefix>_seg.nrrd  : 分割标签（int16）

使用示例：
  python export_nrrd.py \
    --img_b2nd /home/my/data/liver_data/self_data/prepocessed_data/my_case001.b2nd \
    --seg_npy  /home/my/data/liver_data/eval_outputs/my_data/my_case001_pred_3class.npy \
    --out_prefix /home/my/data/liver_data/self_data/slicer/my_case001
"""

import argparse
from pathlib import Path
import pickle as pkl

import blosc2
import numpy as np
import SimpleITK as sitk


# ---------- 读取 CT ----------

def load_ct_from_b2nd(img_b2nd_path: Path) -> np.ndarray:
    dparams = {"nthreads": 1}
    arr = blosc2.open(urlpath=str(img_b2nd_path), mode="r", dparams=dparams)[:]
    print(f"[INFO] raw img_b2nd shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim == 4:
        # (C, Z, Y, X)，LiTS 一般 C=1
        ct = arr[0].astype(np.float32)
    elif arr.ndim == 3:
        ct = arr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected CT array shape: {arr.shape}")

    print(f"[INFO] CT volume shape={ct.shape}")
    return ct


# ---------- 读取 seg.npy ----------

def load_seg_from_npy(seg_path: Path) -> np.ndarray:
    arr = np.load(seg_path)
    print(f"[INFO] raw seg.npy shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim == 3:
        seg = arr
    elif arr.ndim == 4:
        if arr.shape[0] == 1:
            seg = arr[0]
        else:
            # (C, Z, Y, X) 概率 / logits → argmax
            seg = np.argmax(arr, axis=0)
    else:
        raise ValueError(f"Unexpected seg array shape: {arr.shape}")

    seg = seg.astype(np.int16)
    print(f"[INFO] parsed seg shape={seg.shape}, unique={np.unique(seg)}")
    return seg


# ---------- 从 pkl 里读 spacing / origin / direction ----------

def load_geom_from_pkl(pkl_path: Path):
    """
    从 preprocess_ct_data.py 写出的 pkl 里读：
      - target_spacing_zyx  （就是你预处理用的 spacing）
      - orig_origin / orig_direction 若有就用，否则用默认
    """
    spacing_zyx = None
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    if not pkl_path.is_file():
        print(f"[WARN] properties pkl not found: {pkl_path}")
        return spacing_zyx, origin, direction

    try:
        with open(pkl_path, "rb") as f:
            props = pkl.load(f)
    except Exception as e:
        print(f"[WARN] failed to load props from {pkl_path}: {e}")
        return spacing_zyx, origin, direction

    # spacing 优先读取 target_spacing_zyx
    for key in ["target_spacing_zyx", "spacing_zyx", "spacing"]:
        if key in props:
            spacing_zyx = list(props[key])
            print(f"[INFO] spacing_zyx loaded from props[{key}] = {spacing_zyx}")
            break

    if "orig_origin" in props:
        try:
            origin = tuple(props["orig_origin"])
        except Exception:
            pass

    if "orig_direction" in props and len(props["orig_direction"]) == 9:
        try:
            direction = tuple(props["orig_direction"])
        except Exception:
            pass

    return spacing_zyx, origin, direction


# ---------- 主导出函数 ----------

def export_to_nrrd(
    img_b2nd: Path,
    seg_npy: Path,
    out_prefix: Path,
    spacing_zyx_arg: list[float] | None = None,
):
    # 1) 读 CT & seg
    ct = load_ct_from_b2nd(img_b2nd)    # (Z, Y, X)
    seg = load_seg_from_npy(seg_npy)    # (Z, Y, X)

    if ct.shape != seg.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape} vs seg {seg.shape}")

    # 2) 读几何信息（spacing, origin, direction）
    pkl_path = img_b2nd.with_suffix(".pkl")
    spacing_zyx_pkl, origin, direction = load_geom_from_pkl(pkl_path)

    if spacing_zyx_arg is not None:
        spacing_zyx = list(spacing_zyx_arg)
        print(f"[INFO] spacing_zyx overridden from args = {spacing_zyx}")
    elif spacing_zyx_pkl is not None:
        spacing_zyx = spacing_zyx_pkl
    else:
        print("[WARN] no spacing in args or pkl, using [1,1,1]")
        spacing_zyx = [1.0, 1.0, 1.0]

    print(f"[INFO] final spacing_zyx (z,y,x) = {spacing_zyx}")
    spacing_xyz = (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))
    print(f"[INFO] spacing_xyz for SimpleITK = {spacing_xyz}")

    # 3) 构建 SimpleITK Image
    ct_img = sitk.GetImageFromArray(ct)   # float32
    seg_img = sitk.GetImageFromArray(seg) # int16

    ct_img.SetSpacing(spacing_xyz)
    seg_img.SetSpacing(spacing_xyz)

    ct_img.SetOrigin(origin)
    seg_img.SetOrigin(origin)
    ct_img.SetDirection(direction)
    seg_img.SetDirection(direction)

    # 4) 写 NRRD
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_ct = out_prefix.parent / f"{out_prefix.stem}_ct.nrrd"
    out_seg = out_prefix.parent / f"{out_prefix.stem}_seg.nrrd"

    sitk.WriteImage(ct_img, str(out_ct))
    sitk.WriteImage(seg_img, str(out_seg))

    print(f"[INFO] CT NRRD saved to : {out_ct}")
    print(f"[INFO] SEG NRRD saved to: {out_seg}")
    print("[INFO] 现在可以在 3D Slicer 里同时加载这两个文件进行查看。")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_b2nd", type=str, required=True,
                        help="预处理后的 CT .b2nd")
    parser.add_argument("--seg_npy",  type=str, required=True,
                        help="推理得到的 seg.npy")
    parser.add_argument("--out_prefix", type=str, required=True,
                        help="输出前缀（不带后缀），例如 /path/to/my_case001")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("SZ", "SY", "SX"),
        help="可选：手动指定 spacing_zyx；若不指定则优先从 .pkl 读取"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_to_nrrd(
        img_b2nd=Path(args.img_b2nd),
        seg_npy=Path(args.seg_npy),
        out_prefix=Path(args.out_prefix),
        spacing_zyx_arg=args.spacing,
    )


if __name__ == "__main__":
    main()
