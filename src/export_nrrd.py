#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 nnUNet 预处理空间里的预测 seg.npy，
重采样回「原始 CT nrrd」的空间，生成一个可以在 3D Slicer 里
直接叠加查看的 seg.nrrd。

新增：
  --flip_z / --flip_y / --flip_x 开关，用于在某个轴上翻转 seg，
  解决上下/左右颠倒的问题。
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def load_seg_from_npy(seg_path: Path) -> np.ndarray:
    """
    读取 seg.npy，并整理成 (Z, Y, X) 的 int16 标签体积。
    支持：
      - (Z, Y, X)           直接标签
      - (1, Z, Y, X)        单通道标签
      - (C, Z, Y, X)        多通道 logits/prob（对 C 取 argmax）
    """
    arr = np.load(seg_path)
    print(f"[INFO] raw seg.npy shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim == 3:
        seg = arr
    elif arr.ndim == 4:
        if arr.shape[0] == 1:
            seg = arr[0]
        else:
            seg = np.argmax(arr, axis=0)
    else:
        raise ValueError(f"Unexpected seg array shape: {arr.shape}")

    seg = seg.astype(np.int16)
    print(f"[INFO] parsed seg shape={seg.shape}, unique={np.unique(seg)}")
    return seg


def resample_seg_to_original(
    orig_ct_path: Path,
    seg_npy_path: Path,
    spacing_zyx: list[float],
    out_seg_path: Path,
    flip_z: bool = False,
    flip_y: bool = False,
    flip_x: bool = False,
):
    # 1) 读取原始 CT nrrd（这就是你在 Slicer 里看的正常图像）
    ct_img = sitk.ReadImage(str(orig_ct_path))
    print("[INFO] Original CT:")
    print("  size    :", ct_img.GetSize())
    print("  spacing :", ct_img.GetSpacing())
    print("  origin  :", ct_img.GetOrigin())
    print("  direction:", ct_img.GetDirection())

    # 2) 读取预测 seg.npy（预处理空间）
    seg_arr = load_seg_from_npy(seg_npy_path)  # (Z, Y, X)

    # ---------- 关键：根据需要翻转轴 ----------
    # 通常「上下反」对应 slice 顺序反了，多数情况是 Z 轴：
    #   axis=0  -> 翻转 Z（切片索引）
    #   axis=1  -> 翻转 Y
    #   axis=2  -> 翻转 X
    if flip_z:
        print("[INFO] Flipping seg along Z axis (axis=0)")
        seg_arr = np.flip(seg_arr, axis=0).copy()
    if flip_y:
        print("[INFO] Flipping seg along Y axis (axis=1)")
        seg_arr = np.flip(seg_arr, axis=1).copy()
    if flip_x:
        print("[INFO] Flipping seg along X axis (axis=2)")
        seg_arr = np.flip(seg_arr, axis=2).copy()
    # -----------------------------------------

    # 3) 给 seg.npy 构造一个「预处理空间」的 SimpleITK Image
    # spacing_zyx 是 nnUNet 预处理后的 spacing（例如 [1.0, 0.7675, 0.7675]）
    spacing_xyz = (
        float(spacing_zyx[2]),
        float(spacing_zyx[1]),
        float(spacing_zyx[0]),
    )
    print(f"[INFO] seg preproc spacing_zyx = {spacing_zyx}")
    print(f"[INFO] seg preproc spacing_xyz = {spacing_xyz}")

    seg_preproc_img = sitk.GetImageFromArray(seg_arr)
    seg_preproc_img.SetSpacing(spacing_xyz)

    # 此处仍按你原来的假设处理：预处理阶段只改 spacing，没有动 origin/direction
    seg_preproc_img.SetOrigin(ct_img.GetOrigin())
    seg_preproc_img.SetDirection(ct_img.GetDirection())

    # 4) 用原始 CT 作为参考，把 seg 重采样回原始 CT 的网格
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_img)  # 目标 size/spacing/origin/direction 全跟 CT 一样
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    seg_on_orig = resampler.Execute(seg_preproc_img)

    # 5) 写出新的 seg.nrrd
    out_seg_path = Path(out_seg_path)
    out_seg_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(seg_on_orig, str(out_seg_path))

    print(f"[INFO] Resampled SEG saved to: {out_seg_path}")
    print("[INFO] 现在在 3D Slicer 里：")
    print("  1) 打开原始 CT nrrd（比如 14.nrrd）")
    print("  2) 再打开这个 seg nrrd 作为标签，勾选显示，就可以看到正常 CT 上的分割轮廓了。")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_ct", type=str, required=True,
                        help="原始 CT nrrd 文件，例如 14.nrrd")
    parser.add_argument("--seg_npy", type=str, required=True,
                        help="在预处理空间推理得到的 seg.npy")
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        required=True,
        metavar=("SZ", "SY", "SX"),
        help="预处理空间的 spacing_zyx，例如 1.0 0.767578125 0.767578125"
    )
    parser.add_argument("--out_seg", type=str, required=True,
                        help="输出的 seg.nrrd 路径，例如 /path/to/my_case001_seg_on_orig.nrrd")

    # 新增三个可选开关
    parser.add_argument("--flip_z", action="store_true",
                        help="在 Z 轴上翻转 seg（常用于“上下反了”的情况）")
    parser.add_argument("--flip_y", action="store_true",
                        help="在 Y 轴上翻转 seg")
    parser.add_argument("--flip_x", action="store_true",
                        help="在 X 轴上翻转 seg")
    return parser.parse_args()


def main():
    args = parse_args()
    resample_seg_to_original(
        orig_ct_path=Path(args.orig_ct),
        seg_npy_path=Path(args.seg_npy),
        spacing_zyx=list(args.spacing),
        out_seg_path=Path(args.out_seg),
        flip_z=args.flip_z,
        flip_y=args.flip_y,
        flip_x=args.flip_x,
    )


if __name__ == "__main__":
    main()
