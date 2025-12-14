#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 nrrd 预处理并保存为 nnUNet 风格的 .b2nd + .pkl，供 load_case_np 直接读取。

示例（只有图像，无标注）:
  python preprocess_to_b2nd.py \
    --input_nrrd /data/my_case/Delay_Phase_B10f.nrrd \
    --plans_json /home/my/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans.json \
    --case_id my_case001 \
    --out_dir /home/my/liver/custom_preprocessed

示例（有图像 + segmentation）:
  python preprocess_to_b2nd.py \
    --input_nrrd /data/my_case/Delay_Phase_B10f.nrrd \
    --seg_nrrd   /data/my_case/Segmentation.seg.nrrd \
    --plans_json /home/my/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans.json \
    --case_id my_case001 \
    --out_dir /home/my/liver/custom_preprocessed
"""

import argparse
import json
import pickle
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk


def load_plans(plans_json: str, config: str = "3d_fullres"):
    """
    从 nnUNetPlans.json 读取 target spacing (3d_fullres) 和 CTNormalization 统计量。
    """
    plans_json = Path(plans_json)
    with open(plans_json, "r") as f:
        plans = json.load(f)

    cfg = plans["configurations"][config]
    target_spacing_zyx = cfg["spacing"]  # [z, y, x]

    fg_props = plans["foreground_intensity_properties_per_channel"]["0"]
    stats = {
        "mean": float(fg_props["mean"]),
        "std": float(fg_props["std"]),
        "p00_5": float(fg_props["percentile_00_5"]),
        "p99_5": float(fg_props["percentile_99_5"]),
    }
    return target_spacing_zyx, stats


def resample_image_sitk(img: sitk.Image, target_spacing_zyx, is_seg: bool = False):
    """
    用 SimpleITK 重采样到目标 spacing。
    nnUNet spacing 为 [z,y,x]，SimpleITK 需要 (x,y,z)。
    segmentation 用最近邻，图像用 BSpline。
    """
    orig_spacing = np.array(list(img.GetSpacing()), dtype=float)  # (x,y,z)
    orig_size = np.array(list(img.GetSize()), dtype=int)

    target_spacing_zyx = np.array(target_spacing_zyx, dtype=float)
    target_spacing_xyz = np.array(
        [target_spacing_zyx[2], target_spacing_zyx[1], target_spacing_zyx[0]],
        dtype=float,
    )

    new_size = np.round(orig_size * (orig_spacing / target_spacing_xyz)).astype(int)

    resampler = sitk.ResampleImageFilter()
    if is_seg:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    resampler.SetOutputSpacing(tuple(target_spacing_xyz))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkFloat32 if not is_seg else sitk.sitkInt16)

    return resampler.Execute(img)


def ct_normalize_like_nnunet(volume_zyx: np.ndarray, stats: dict) -> np.ndarray:
    """
    按 nnUNet 的 CTNormalization：
      - clip 到 [p00_5, p99_5]
      - 减 mean / 除 std
    输入: (Z, Y, X)，输出: (1, Z, Y, X)
    """
    v = volume_zyx.astype(np.float32)
    v = np.clip(v, stats["p00_5"], stats["p99_5"])
    std = stats["std"] if stats["std"] > 0 else 1.0
    v = (v - stats["mean"]) / std
    v = v[None, ...]  # (1, Z, Y, X)
    return v.astype(np.float32)


def save_b2nd(array: np.ndarray, out_path: Path):
    """
    使用 blosc2 保存 numpy array 到 .b2nd
    最终格式：
      - 4D: (C, Z, Y, X)
      - dtype: float32 (image) 或 int16 (seg)

    允许输入:
      - 3D: (Z, Y, X)  → 自动在前面加一个通道维度
      - 4D: (C, Z, Y, X)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.ascontiguousarray(array)

    # 如果是 3D，就自动变成 (1, Z, Y, X)
    if array.ndim == 3:
        array = array[None, ...]
    elif array.ndim != 4:
        raise ValueError(
            f"save_b2nd expects 3D or 4D array, got shape={array.shape}"
        )

    blosc2.set_nthreads(1)

    C, Z, Y, X = array.shape

    # 压缩参数：ZSTD + 中等压缩等级
    cparams = blosc2.CParams(
        clevel=5,
        codec=blosc2.Codec.ZSTD,
        typesize=int(array.dtype.itemsize),
    )
    dparams = {"nthreads": 1}

    # chunk 形状，尽量不要太大，nnUNet 里通常每维不超过 128
    chunks = (
        min(1, C),            # 通道一般就是 1
        min(128, Z),
        min(128, Y),
        min(128, X),
    )

    # 用 blosc2.open 创建磁盘后端 NDArray，然后整体写入
    b2 = blosc2.open(
        urlpath=str(out_path),
        mode="w",
        dtype=array.dtype,
        shape=array.shape,
        chunks=chunks,
        cparams=cparams,
        dparams=dparams,
    )
    b2[:] = array  # 一次性写入数据

    print(f"[INFO] Saved b2nd: {out_path}, shape={array.shape}, dtype={array.dtype}")


def preprocess_to_b2nd(
    input_nrrd: str,
    plans_json: str,
    case_id: str,
    out_dir: str,
    seg_nrrd: str | None = None,
    config: str = "3d_fullres",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 nnUNet plans
    target_spacing_zyx, stats = load_plans(plans_json, config=config)
    print(f"[INFO] target_spacing (zyx) = {target_spacing_zyx}")

    # 2) 读原始 CT
    img = sitk.ReadImage(str(input_nrrd))
    orig_spacing = img.GetSpacing()
    orig_origin = img.GetOrigin()
    orig_direction = img.GetDirection()
    print(f"[INFO] original spacing (xyz) = {orig_spacing}")

    # 3) 重采样 CT 到 nnUNet spacing
    img_resampled = resample_image_sitk(img, target_spacing_zyx, is_seg=False)
    vol_zyx = sitk.GetArrayFromImage(img_resampled)  # (Z, Y, X)

    # 4) CTNormalization
    vol_norm = ct_normalize_like_nnunet(vol_zyx, stats)  # (1, Z, Y, X)

    # 5) 保存 image.b2nd
    img_b2nd_path = out_dir / f"{case_id}.b2nd"
    save_b2nd(vol_norm, img_b2nd_path)

    # 6) 如果给了 segmentation，也一起做 b2nd
    seg_arr_resampled = None
    if seg_nrrd is not None:
        seg_img = sitk.ReadImage(str(seg_nrrd))
        # 假设 seg 的物理空间和 input_nrrd 是对齐的（通常是）
        seg_img_resampled = resample_image_sitk(seg_img, target_spacing_zyx, is_seg=True)
        seg_zyx = sitk.GetArrayFromImage(seg_img_resampled).astype(np.int16)  # (Z, Y, X)

        seg_b2nd_path = out_dir / f"{case_id}_seg.b2nd"
        # 这里直接传 3D，save_b2nd 内部会自动加通道维
        save_b2nd(seg_zyx, seg_b2nd_path)
        seg_arr_resampled = seg_zyx


    # 7) 保存 properties.pkl
    props = {
        "case_id": case_id,
        "orig_spacing_xyz": tuple(orig_spacing),
        "orig_origin": tuple(orig_origin),
        "orig_direction": tuple(orig_direction),
        "target_spacing_zyx": tuple(target_spacing_zyx),
        "image_shape_after_preproc": vol_norm.shape,  # (1, Z, Y, X)
    }
    if seg_arr_resampled is not None:
        props["seg_shape_after_preproc"] = seg_arr_resampled.shape  # (Z, Y, X)
        props["seg_unique_labels"] = np.unique(seg_arr_resampled).tolist()

    prop_path = out_dir / f"{case_id}.pkl"
    with open(prop_path, "wb") as f:
        pickle.dump(props, f)
    print(f"[INFO] Saved properties pkl: {prop_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nrrd", type=str, required=True,
                        help="原始 CT nrrd 文件")
    parser.add_argument("--plans_json", type=str, required=True,
                        help="LiTS nnUNetPlans.json 路径")
    parser.add_argument("--case_id", type=str, required=True,
                        help="保存时使用的 case_id（文件前缀）")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="输出目录（会生成 case_id.b2nd / case_id.pkl 等）")
    parser.add_argument("--seg_nrrd", type=str, default=None,
                        help="可选：对应的分割 nrrd，用于生成 case_id_seg.b2nd")
    parser.add_argument("--config", type=str, default="3d_fullres",
                        help="使用的 nnUNet 配置（默认 3d_fullres）")
    args = parser.parse_args()

    preprocess_to_b2nd(
        input_nrrd=args.input_nrrd,
        plans_json=args.plans_json,
        case_id=args.case_id,
        out_dir=args.out_dir,
        seg_nrrd=args.seg_nrrd,
        config=args.config,
    )


if __name__ == "__main__":
    main()
