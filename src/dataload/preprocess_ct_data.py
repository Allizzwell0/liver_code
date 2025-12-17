#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path

import blosc2
import numpy as np
import SimpleITK as sitk


# ------------------ nnUNet plans 读取 ------------------ #
def load_plans(plans_json: str, config: str = "3d_fullres"):
    plans_json = Path(plans_json)
    plans = json.loads(plans_json.read_text())

    cfg = plans["configurations"][config]
    target_spacing_zyx = cfg["spacing"]  # [z,y,x]

    fg_props = plans["foreground_intensity_properties_per_channel"]["0"]
    stats = {
        "mean": float(fg_props["mean"]),
        "std": float(fg_props["std"]),
        "p00_5": float(fg_props["percentile_00_5"]),
        "p99_5": float(fg_props["percentile_99_5"]),
    }
    return target_spacing_zyx, stats


# ------------------ 方向统一到 RAS ------------------ #
def reorient_to_ras(img: sitk.Image) -> sitk.Image:
    """
    把图像重排到 RAS（Right-Anterior-Superior）方向。
    这是“物理空间一致”的重排：会更新 direction/origin，并对应重排像素。
    """
    # SimpleITK 内置：DICOMOrientImageFilter 接受 "RAS"
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


# ------------------ 重采样到 target spacing ------------------ #
def resample_to_spacing(img: sitk.Image, target_spacing_zyx, is_seg=False) -> sitk.Image:
    """
    nnUNet 的 spacing 是 [z,y,x]，SITK spacing 是 (x,y,z)
    """
    target_spacing_xyz = (float(target_spacing_zyx[2]),
                          float(target_spacing_zyx[1]),
                          float(target_spacing_zyx[0]))

    orig_spacing = img.GetSpacing()
    orig_size = img.GetSize()

    new_size = [
        int(np.round(orig_size[i] * (orig_spacing[i] / target_spacing_xyz[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_xyz)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_seg:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(sitk.sitkInt16)
    else:
        # nnUNet CT 常用 BSpline / Linear 都行，BSpline更平滑但更慢
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)

    return resampler.Execute(img)


# ------------------ nnUNet CTNormalization ------------------ #
def ct_normalize_like_nnunet(vol_zyx: np.ndarray, stats: dict) -> np.ndarray:
    """
    输入: (Z,Y,X) float32（最好是 HU）
    输出: (1,Z,Y,X) float32
    """
    v = vol_zyx.astype(np.float32)
    v = np.clip(v, stats["p00_5"], stats["p99_5"])
    std = stats["std"] if stats["std"] > 1e-8 else 1.0
    v = (v - stats["mean"]) / std
    return v[None, ...].astype(np.float32)


# ------------------ 保存 b2nd ------------------ #
def save_b2nd_4d(array_4d: np.ndarray, out_path: Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    array_4d = np.ascontiguousarray(array_4d)
    assert array_4d.ndim == 4, f"expect (C,Z,Y,X), got {array_4d.shape}"

    blosc2.set_nthreads(1)
    if out_path.exists():
        out_path.unlink()

    cparams = blosc2.CParams(
        clevel=5,
        codec=blosc2.Codec.ZSTD,
        typesize=int(array_4d.dtype.itemsize),
    )
    dparams = {"nthreads": 1}

    _ = blosc2.asarray(array_4d, urlpath=str(out_path), cparams=cparams, dparams=dparams)
    print(f"[OK] Saved b2nd: {out_path} | shape={array_4d.shape} | dtype={array_4d.dtype}")


# ------------------ 主流程 ------------------ #
def preprocess_one_case(
    input_nrrd: str,
    plans_json: str,
    case_id: str,
    out_dir: str,
    config: str = "3d_fullres",
    seg_nrrd: str | None = None,
    force_ras: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_spacing_zyx, stats = load_plans(plans_json, config=config)
    print("[INFO] target_spacing_zyx:", target_spacing_zyx)
    print("[INFO] nnUNet CT stats:", stats)

    # 1) 读取
    img = sitk.ReadImage(str(input_nrrd))
    print("[INFO] original:")
    print("  size     :", img.GetSize())
    print("  spacing  :", img.GetSpacing())
    print("  origin   :", img.GetOrigin())
    print("  direction:", img.GetDirection())

    # 2) 方向统一（强烈建议做，尤其你已经遇到左右/上下翻转问题）
    if force_ras:
        img = reorient_to_ras(img)
        print("[INFO] after RAS reorient:")
        print("  direction:", img.GetDirection())

    # 3) 重采样到 nnUNet spacing
    img_r = resample_to_spacing(img, target_spacing_zyx, is_seg=False)
    vol_zyx = sitk.GetArrayFromImage(img_r)  # (Z,Y,X)

    # 4) 强度处理（nnUNet 的 CTNormalization）
    vol_norm = ct_normalize_like_nnunet(vol_zyx, stats)  # (1,Z,Y,X)

    # 5) 保存 image.b2nd
    img_b2nd_path = out_dir / f"{case_id}.b2nd"
    save_b2nd_4d(vol_norm, img_b2nd_path)

    # 6) seg（如果有），也走同样几何流程：RAS + resample（nearest）
    seg_info = None
    if seg_nrrd is not None:
        seg = sitk.ReadImage(str(seg_nrrd))
        if force_ras:
            seg = reorient_to_ras(seg)
        seg_r = resample_to_spacing(seg, target_spacing_zyx, is_seg=True)
        seg_zyx = sitk.GetArrayFromImage(seg_r).astype(np.int16)  # (Z,Y,X)
        seg_b2nd_path = out_dir / f"{case_id}_seg.b2nd"
        save_b2nd_4d(seg_zyx[None, ...], seg_b2nd_path)
        seg_info = {
            "seg_shape_zyx": tuple(seg_zyx.shape),
            "seg_unique": np.unique(seg_zyx).tolist(),
        }

    # 7) properties.pkl：把“几何信息”记录下来，后续把预测映射回原始空间要用
    props = {
        "case_id": case_id,
        "config": config,
        "target_spacing_zyx": tuple(target_spacing_zyx),
        "nnunet_stats": stats,

        # 关键：存“原始”和“预处理后”的 sitk 元信息
        "orig_nrrd_path": str(input_nrrd),
        "orig_size_xyz": tuple(img.GetSize()),
        "orig_spacing_xyz": tuple(img.GetSpacing()),
        "orig_origin_xyz": tuple(img.GetOrigin()),
        "orig_direction": tuple(img.GetDirection()),

        "preproc_size_xyz": tuple(img_r.GetSize()),
        "preproc_spacing_xyz": tuple(img_r.GetSpacing()),
        "preproc_origin_xyz": tuple(img_r.GetOrigin()),
        "preproc_direction": tuple(img_r.GetDirection()),

        "image_shape_after_preproc": tuple(vol_norm.shape),  # (1,Z,Y,X)
    }
    if seg_info is not None:
        props.update(seg_info)

    pkl_path = out_dir / f"{case_id}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(props, f)
    print(f"[OK] Saved pkl: {pkl_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_nrrd", required=True)
    ap.add_argument("--plans_json", required=True)
    ap.add_argument("--case_id", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--seg_nrrd", default=None)
    ap.add_argument("--no_ras", action="store_true", help="不做 RAS 重排（不推荐）")
    args = ap.parse_args()

    preprocess_one_case(
        input_nrrd=args.input_nrrd,
        plans_json=args.plans_json,
        case_id=args.case_id,
        out_dir=args.out_dir,
        config=args.config,
        seg_nrrd=args.seg_nrrd,
        force_ras=not args.no_ras,
    )


if __name__ == "__main__":
    main()
