#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 nrrd 图像重排成 RAS 方向并保存。

用法示例：
1) 处理单个 nrrd：
   python reorient_to_RAS.py \
       --input /path/to/Delay_Phase_B10f.nrrd \
       --output /path/to/Delay_Phase_B10f_RAS.nrrd

2) 批量处理文件夹下所有 .nrrd，到新的目录：
   python reorient_to_RAS.py \
       --input /path/to/original_nrrd_folder \
       --output /path/to/ras_nrrd_folder
"""

import argparse
from pathlib import Path
import SimpleITK as sitk


def reorient_single_nrrd(in_path: Path, out_path: Path):
    img = sitk.ReadImage(str(in_path))

    # 当前方向（方便你检查）
    direction = img.GetDirection()
    try:
        curr_orient = sitk.DICOMOrientImageFilter_GetOrientationFromDirection(direction)
    except Exception:
        curr_orient = "UNKNOWN"

    print(f"[INFO] {in_path.name}: orientation {curr_orient} -> RAS")

    # 重排到 RAS
    img_ras = sitk.DICOMOrient(img, "RAS")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img_ras, str(out_path), useCompression=True)
    print(f"[INFO] saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入：nrrd 文件 或 包含 nrrd 的文件夹",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出：单文件路径（对单个输入）或 目标文件夹（对文件夹输入）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if in_path.is_file():
        # 单个 nrrd
        if out_path.is_dir():
            out_path = out_path / (in_path.stem + "_RAS.nrrd")
        reorient_single_nrrd(in_path, out_path)

    elif in_path.is_dir():
        # 批量处理文件夹
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)

        nrrd_files = sorted(list(in_path.glob("*.nrrd")))
        if not nrrd_files:
            print(f"[WARN] No .nrrd files found in {in_path}")
            return

        for f in nrrd_files:
            out_f = out_dir / f.name.replace(".nrrd", "_RAS.nrrd")
            reorient_single_nrrd(f, out_f)
    else:
        raise FileNotFoundError(f"Input path not found: {in_path}")


if __name__ == "__main__":
    main()
