#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 LiTS 和 自己数据 的 3D 预处理结果（.b2nd），保存中间切片图片到磁盘。

用法示例：
  python visualize_b2nd_compare.py \
    --lits_b2nd /home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres/liver_0.b2nd \
    --self_b2nd /home/my/data/liver_data/self_data/prepocessed_data/my_case001.b2nd \
    --out_dir ./vis_b2nd
"""

import argparse
from pathlib import Path

import blosc2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 重要：无图形界面环境
import matplotlib.pyplot as plt


def load_b2nd(path: Path) -> np.ndarray:
    arr = blosc2.open(urlpath=str(path), mode="r")[:].astype(np.float32)
    # 形状 (C, Z, Y, X)
    if arr.ndim != 4:
        raise RuntimeError(f"Expect 4D array (C, Z, Y, X), got {arr.shape} for {path}")
    return arr


def save_mid_slices(arr: np.ndarray, title_prefix: str, out_dir: Path):
    """
    arr: (C, Z, Y, X)
    保存 axial / coronal / sagittal 三个中间切片各一张 PNG
    """
    c, Z, Y, X = arr.shape
    vol = arr[0]  # 用第一个通道 (Z, Y, X)

    z_mid = Z // 2
    y_mid = Y // 2 + 10
    x_mid = X // 2

    # Axial
    plt.figure(figsize=(5, 5))
    plt.imshow(vol[z_mid], cmap="gray")
    plt.title(f"{title_prefix} axial (z={z_mid})")
    plt.axis("off")
    out_path = out_dir / f"{title_prefix}_axial.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    # Coronal
    plt.figure(figsize=(5, 5))
    plt.imshow(vol[:, y_mid, :], cmap="gray")
    plt.title(f"{title_prefix} coronal (y={y_mid})")
    plt.axis("off")
    out_path = out_dir / f"{title_prefix}_coronal.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()

    # Sagittal
    plt.figure(figsize=(5, 5))
    plt.imshow(vol[:, :, x_mid], cmap="gray")
    plt.title(f"{title_prefix} sagittal (x={x_mid})")
    plt.axis("off")
    out_path = out_dir / f"{title_prefix}_sagittal.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lits_b2nd", type=str, required=True,
                        help="LiTS 预处理后的 .b2nd 路径，例如 liver_0.b2nd")
    parser.add_argument("--self_b2nd", type=str, required=True,
                        help="自己数据预处理后的 .b2nd 路径，例如 my_case001.b2nd")
    parser.add_argument("--out_dir", type=str, default="./vis_b2nd",
                        help="输出 PNG 图片的文件夹")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading LiTS b2nd from {args.lits_b2nd}")
    arr_lits = load_b2nd(Path(args.lits_b2nd))
    print(f"  shape={arr_lits.shape}")

    print(f"[INFO] Loading Self b2nd from {args.self_b2nd}")
    arr_self = load_b2nd(Path(args.self_b2nd))
    print(f"  shape={arr_self.shape}")

    print("[INFO] Saving mid-slice images...")
    save_mid_slices(arr_lits, "lits_liver0", out_dir)
    save_mid_slices(arr_self, "self_case001", out_dir)
    print(f"[INFO] Done. Images saved in: {out_dir}")


if __name__ == "__main__":
    main()
