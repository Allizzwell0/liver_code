#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage inference on LiTS / MSD Task03_Liver preprocessed by nnUNetv2.

Stage 1: liver model -> liver mask (ROI)
Stage 2: tumor model -> tumor mask (only inside liver ROI)

Input:
  - nnUNetv2 preprocessed folder, e.g.
    /home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres
  - liver_best.pth, tumor_best.pth from your training

Output:
  - for each case_id, saves:
      <out_dir>/<case_id>/liver_mask.npy
      <out_dir>/<case_id>/tumor_mask.npy
    and if SimpleITK installed:
      <out_dir>/<case_id>/liver_mask.nii.gz
      <out_dir>/<case_id>/tumor_mask.nii.gz
"""

import argparse
import pickle as pkl
from pathlib import Path
from typing import Tuple, List

import blosc2
import numpy as np
import torch
import torch.nn.functional as F

from models.unet3D import UNet3D  # 确保和训练时用的是同一个文件


# ----------------- IO: load preprocessed case -----------------


def load_case_b2nd(preproc_dir: Path, case_id: str):
    """
    从 nnUNetv2 预处理目录中读出一个病例:
      - image: float32, (C, Z, Y, X)
      - seg:   int16,  (Z, Y, X) 或 None（如果没有_gt）
      - props: dict (properties.pkl 内容)
    """
    dparams = {"nthreads": 1}

    data_file = preproc_dir / f"{case_id}.b2nd"
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    prop_file = preproc_dir / f"{case_id}.pkl"

    if not data_file.is_file():
        raise FileNotFoundError(data_file)

    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    img = data_b[:].astype(np.float32)  # (C, Z, Y, X)

    seg = None
    if seg_file.is_file():
        seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)
        seg = seg_b[:].astype(np.int16)  # (1, Z, Y, X) or (Z, Y, X)
        if seg.ndim == 4:
            seg = seg[0]

    props = None
    if prop_file.is_file():
        with open(prop_file, "rb") as f:
            props = pkl.load(f)

    return img, seg, props


# ----------------- model loading -----------------


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    """
    从训练保存的 .pth 里恢复 UNet3D 模型、输入通道数、类别数、patch_size.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)
    patch_size = tuple(ckpt.get("patch_size", (96, 160, 160)))

    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base_filters=32)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[load_model] {ckpt_path.name}: in_ch={in_channels}, "
          f"num_classes={num_classes}, patch_size={patch_size}")
    return model, num_classes, patch_size


# ----------------- sliding-window inference -----------------


def pad_to_min_shape(img: np.ndarray, patch_size: Tuple[int, int, int]):
    """
    确保体积在每个维度上 >= patch_size，对右/后面做 0 padding.
    img: (C, Z, Y, X)
    """
    C, Z, Y, X = img.shape
    pz, py, px = patch_size
    Zp = max(Z, pz)
    Yp = max(Y, py)
    Xp = max(X, px)

    pad_z = Zp - Z
    pad_y = Yp - Y
    pad_x = Xp - X

    img_p = np.pad(
        img,
        ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
        mode="constant",
    )
    return img_p, (Z, Y, X)


def get_start_indices(full_size: int, patch_size: int, step: int) -> List[int]:
    """
    沿一个维度生成滑窗起始 index，保证末尾覆盖到结尾。
    """
    if full_size <= patch_size:
        return [0]
    starts = list(range(0, full_size - patch_size + 1, step))
    if starts[-1] != full_size - patch_size:
        starts.append(full_size - patch_size)
    return starts


def sliding_window_segmentation(
    model: torch.nn.Module,
    img: np.ndarray,
    patch_size: Tuple[int, int, int],
    device: torch.device,
    num_classes: int = 2,
    step_fraction: float = 0.5,
):
    """
    对 (C, Z, Y, X) 整体做 3D 滑窗推理，返回:
      - seg:  (Z, Y, X) int64, 每个 voxel 的类别
      - probs: (C_out, Z, Y, X) float32, 概率
    """
    assert img.ndim == 4, f"img must be (C,Z,Y,X), got {img.shape}"
    img_p, orig_shape = pad_to_min_shape(img, patch_size)
    C, Zp, Yp, Xp = img_p.shape
    pz, py, px = patch_size

    step_z = max(int(pz * step_fraction), 1)
    step_y = max(int(py * step_fraction), 1)
    step_x = max(int(px * step_fraction), 1)

    z_starts = get_start_indices(Zp, pz, step_z)
    y_starts = get_start_indices(Yp, py, step_y)
    x_starts = get_start_indices(Xp, px, step_x)

    print(f"[SW] volume={img.shape}, padded={(C, Zp, Yp, Xp)}, "
          f"patch={patch_size}, steps=({step_z},{step_y},{step_x}), "
          f"num_tiles={len(z_starts)*len(y_starts)*len(x_starts)}")

    probs = np.zeros((num_classes, Zp, Yp, Xp), dtype=np.float32)
    counts = np.zeros((1, Zp, Yp, Xp), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for zs in z_starts:
            for ys in y_starts:
                for xs in x_starts:
                    patch = img_p[:, zs:zs+pz, ys:ys+py, xs:xs+px]  # (C,pz,py,px)
                    patch_t = torch.from_numpy(patch[None]).to(device)  # (1,C,pz,py,px)

                    logits = model(patch_t)   # (1,C_out,*,*,*)
                    probs_patch = F.softmax(logits, dim=1)[0].cpu().numpy()  # (C_out,pz,py,px)

                    probs[:, zs:zs+pz, ys:ys+py, xs:xs+px] += probs_patch
                    counts[:, zs:zs+pz, ys:ys+py, xs:xs+px] += 1.0

    counts[counts == 0] = 1.0
    probs = probs / counts

    Z, Y, X = orig_shape
    probs = probs[:, :Z, :Y, :X]  # 去掉 padding
    seg = np.argmax(probs, axis=0).astype(np.int64)
    return seg, probs


# ----------------- ROI: liver bbox -----------------


def compute_liver_bbox(liver_mask: np.ndarray, margin: int = 5):
    """
    liver_mask: (Z,Y,X) bool/0-1
    返回 (z_min,z_max,y_min,y_max,x_min,x_max)
    若未检测到肝，则返回整幅 bbox.
    """
    assert liver_mask.ndim == 3
    Z, Y, X = liver_mask.shape
    coords = np.where(liver_mask > 0)
    if coords[0].size == 0:
        print("[WARN] No liver voxels detected, using full volume as ROI.")
        return 0, Z, 0, Y, 0, X

    z_min, z_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1

    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, Z)
    y_max = min(y_max + margin, Y)
    x_max = min(x_max + margin, X)

    return z_min, z_max, y_min, y_max, x_min, x_max


# ----------------- NIfTI saving -----------------


def save_segmentation(seg: np.ndarray, out_path: Path):
    """
    尝试保存为 NIfTI，如果没有 SimpleITK 就只打印提示。
    seg: (Z,Y,X) int/uint8
    """
    seg = seg.astype(np.uint8)
    try:
        import SimpleITK as sitk
    except ImportError:
        print(f"[WARN] SimpleITK not installed, skip NIfTI save for {out_path}")
        return

    img_sitk = sitk.GetImageFromArray(seg)  # 默认 spacing=(1,1,1)，方便在 Slicer 里简单查看
    sitk.WriteImage(img_sitk, str(out_path))
    print(f"[INFO] Saved NIfTI to {out_path}")


# ----------------- two-stage pipeline for one case -----------------


def run_two_stage_for_case(
    preproc_dir: Path,
    liver_model: torch.nn.Module,
    tumor_model: torch.nn.Module,
    case_id: str,
    out_dir: Path,
    device: torch.device,
    liver_patch_size: Tuple[int, int, int],
    tumor_patch_size: Tuple[int, int, int],
):
    print(f"\n========== Case {case_id} ==========")
    img, seg_gt, props = load_case_b2nd(preproc_dir, case_id)
    print(f"[CASE] img shape={img.shape}, seg_gt={None if seg_gt is None else seg_gt.shape}")

    # -------- Stage 1: liver segmentation --------
    print("[Stage 1] Liver segmentation...")
    liver_seg, liver_probs = sliding_window_segmentation(
        liver_model,
        img,
        liver_patch_size,
        device,
        num_classes=2,
        step_fraction=0.5,
    )
    liver_mask = (liver_seg == 1)

    # -------- ROI: compute bbox from liver --------
    z_min, z_max, y_min, y_max, x_min, x_max = compute_liver_bbox(liver_mask, margin=5)
    print(f"[ROI] liver bbox: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")

    img_roi = img[:, z_min:z_max, y_min:y_max, x_min:x_max]
    liver_mask_roi = liver_mask[z_min:z_max, y_min:y_max, x_min:x_max]

    # -------- Stage 2: tumor segmentation inside ROI --------
    print("[Stage 2] Tumor segmentation in liver ROI...")
    tumor_seg_roi, tumor_probs_roi = sliding_window_segmentation(
        tumor_model,
        img_roi,
        tumor_patch_size,
        device,
        num_classes=2,
        step_fraction=0.5,
    )
    tumor_mask_roi = (tumor_seg_roi == 1)

    # 只允许肿瘤出现在肝内
    tumor_mask_roi = np.logical_and(tumor_mask_roi, liver_mask_roi)

    # 回填到全 volume
    Z, Y, X = liver_mask.shape
    liver_mask_full = liver_mask.astype(np.uint8)
    tumor_mask_full = np.zeros((Z, Y, X), dtype=np.uint8)
    tumor_mask_full[z_min:z_max, y_min:y_max, x_min:x_max] = tumor_mask_roi.astype(np.uint8)

    # -------- Save --------
    case_out_dir = out_dir / case_id
    case_out_dir.mkdir(parents=True, exist_ok=True)

    np.save(case_out_dir / "liver_mask.npy", liver_mask_full)
    np.save(case_out_dir / "tumor_mask.npy", tumor_mask_full)
    print(f"[SAVE] liver_mask.npy & tumor_mask.npy saved to {case_out_dir}")

    save_segmentation(liver_mask_full, case_out_dir / "liver_mask.nii.gz")
    save_segmentation(tumor_mask_full, case_out_dir / "tumor_mask.nii.gz")


# ----------------- main -----------------


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage liver/tumor inference on nnUNet preprocessed LiTS data")
    parser.add_argument("--preproc_dir", type=str, required=True,
                        help="nnUNetv2 preprocessed folder, e.g. .../Dataset003_Liver/nnUNetPlans_3d_fullres")
    parser.add_argument("--liver_ckpt", type=str, required=True,
                        help="Checkpoint path for liver model, e.g. train_logs/lits_b2nd/liver_best.pth")
    parser.add_argument("--tumor_ckpt", type=str, required=True,
                        help="Checkpoint path for tumor model, e.g. train_logs/lits_b2nd/tumor_best.pth")
    parser.add_argument("--out_dir", type=str, default="infer_output",
                        help="Directory to save predictions")
    parser.add_argument("--case_id", type=str, default=None,
                        help="Specific case id (without extension). If not set, run all cases in preproc_dir")
    parser.add_argument("--use_liver_patch_for_tumor", action="store_true",
                        help="Use liver model patch_size for tumor model as well")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load models
    liver_model, liver_num_classes, liver_patch_size = load_model_from_ckpt(Path(args.liver_ckpt), device)
    tumor_model, tumor_num_classes, tumor_patch_size = load_model_from_ckpt(Path(args.tumor_ckpt), device)

    if liver_num_classes != 2 or tumor_num_classes != 2:
        print("[WARN] This script assumes binary models (2 classes).")

    if args.use_liver_patch_for_tumor:
        tumor_patch_size = liver_patch_size
        print(f"[INFO] Using liver_patch_size={liver_patch_size} for tumor model as well.")

    # 2) cases to run
    if args.case_id is not None:
        case_ids = [args.case_id]
    else:
        # 用所有 .pkl 的 stem 作为 case_id
        case_ids = sorted(p.stem for p in preproc_dir.glob("*.pkl"))
        print(f"[INFO] Found {len(case_ids)} cases in {preproc_dir}")

    for cid in case_ids:
        run_two_stage_for_case(
            preproc_dir,
            liver_model,
            tumor_model,
            cid,
            out_dir,
            device,
            liver_patch_size,
            tumor_patch_size,
        )


if __name__ == "__main__":
    main()
