#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
两阶段推理脚本（LiTS, nnUNetv2 预处理结果）：

阶段 1：肝脏分割（liver_best.pth）
阶段 2：在肝脏 ROI 内做肿瘤分割（tumor_best.pth）

输入：
  - nnUNetv2 预处理目录（例如 Dataset003_Liver/nnUNetPlans_3d_fullres）
  - liver 和 tumor 的 checkpoint
  - case_id（可选；不指定则对目录下所有病例做推理）

输出：
  - <out_dir>/<case_id>_pred_liver.npy   (0/1) 肝脏预测
  - <out_dir>/<case_id>_pred_tumor.npy   (0/1) 肿瘤预测
  - <out_dir>/<case_id>_pred_3class.npy  (0/1/2) 最终 3 类标签
  - 若有 GT，会打印 liver/tumor 的 Dice

用法示例：

  PREPROC_DIR=/home/my/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres

  python infer_two_stage.py \
    --preproc_dir $PREPROC_DIR \
    --ckpt_liver train_logs/lits_3dfullres_like/liver_best.pth \
    --ckpt_tumor train_logs/lits_3dfullres_like_tumor/tumor_best.pth \
    --out_dir eval_outputs \
    --case_id Dataset003_Liver_0000

  # 对目录下所有病例推理：
  python infer_two_stage.py \
    --preproc_dir $PREPROC_DIR \
    --ckpt_liver train_logs/lits_3dfullres_like/liver_best.pth \
    --ckpt_tumor train_logs/lits_3dfullres_like_tumor/tumor_best.pth \
    --out_dir eval_outputs
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import blosc2
import numpy as np
import torch
import torch.nn.functional as F

from models.unet3D import UNet3D  # 直接复用你训练时的 U-Net 定义


# ----------------- 一些小工具函数 ----------------- #

def load_case_np(preproc_dir: Path, case_id: str):
    """
    读取 nnUNetv2 预处理后的一个病例：
      img: (C, Z, Y, X), float32
      seg: (Z, Y, X), int16 （如果存在 seg.b2nd，否则 seg=None）
    """
    dparams = {"nthreads": 1}
    img_file = preproc_dir / f"{case_id}.b2nd"
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    prop_file = preproc_dir / f"{case_id}.pkl"

    if not img_file.is_file():
        raise FileNotFoundError(img_file)

    img_b = blosc2.open(urlpath=str(img_file), mode="r", dparams=dparams)
    img = img_b[:].astype(np.float32)  # (C, Z, Y, X)

    seg = None
    if seg_file.is_file():
        seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)
        seg_arr = seg_b[:].astype(np.int16)
        if seg_arr.ndim == 4:
            seg_arr = seg_arr[0]
        seg = seg_arr  # (Z, Y, X)

    properties = None
    if prop_file.is_file():
        import pickle as pkl
        with open(prop_file, "rb") as f:
            properties = pkl.load(f)

    return img, seg, properties


def get_liver_bbox_from_mask(mask: np.ndarray, margin: int = 10) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    从肝脏预测 mask (Z, Y, X, bool 或 0/1) 中提取 3D 包围盒 + margin。
    """
    liver_mask = mask > 0
    if not liver_mask.any():
        return None

    zz, yy, xx = np.where(liver_mask)
    z_min, z_max = int(zz.min()), int(zz.max())
    y_min, y_max = int(yy.min()), int(yy.max())
    x_min, x_max = int(xx.min()), int(xx.max())

    Z, Y, X = mask.shape

    z_min = max(0, z_min - margin)
    y_min = max(0, y_min - margin)
    x_min = max(0, x_min - margin)

    z_max = min(Z - 1, z_max + margin)
    y_max = min(Y - 1, y_max + margin)
    x_max = min(X - 1, x_max + margin)

    return z_min, z_max, y_min, y_max, x_min, x_max


def make_sliding_windows(
    Z: int,
    Y: int,
    X: int,
    pz: int,
    py: int,
    px: int,
    overlap: float = 0.5,
) -> List[Tuple[int, int, int]]:
    """
    为 3D 体数据生成滑窗起点列表 (z0, y0, x0)。
    overlap ∈ [0,1)，比如 0.5 表示步长约为 patch_size 的一半。
    """
    assert 0 <= overlap < 1.0
    def _compute_starts(dim, p):
        if dim <= p:
            return [0]
        step = int(p * (1 - overlap))
        step = max(1, step)
        starts = list(range(0, dim - p + 1, step))
        if starts[-1] != dim - p:
            starts.append(dim - p)
        return starts

    z_starts = _compute_starts(Z, pz)
    y_starts = _compute_starts(Y, py)
    x_starts = _compute_starts(X, px)

    windows = []
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                windows.append((z0, y0, x0))
    return windows


def sliding_window_predict(
    img: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    batch_size: int = 1,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    对整幅体数据 img 做 3D 滑窗推理。
    img: (C, Z, Y, X)  numpy
    返回:
      pred (Z, Y, X) int64, 为 argmax 后的类别（0~C-1）
    """
    model.eval()
    C, Z, Y, X = img.shape
    pz, py, px = patch_size

    windows = make_sliding_windows(Z, Y, X, pz, py, px, overlap=overlap)

    # 累积概率和计数器用于 average
    prob_vol = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_vol = np.zeros((1, Z, Y, X), dtype=np.float32)

    with torch.no_grad():
        # 为了效率，支持简单的 batch 推理
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_imgs = []
            for (z0, y0, x0) in batch_windows:
                patch = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
                batch_imgs.append(patch)

            batch_arr = np.stack(batch_imgs, axis=0)  # (B, C, pz, py, px)
            batch_tensor = torch.from_numpy(batch_arr).to(device=device, dtype=torch.float32)

            logits = model(batch_tensor)  # (B, num_classes, pz, py, px)
            probs = F.softmax(logits, dim=1).cpu().numpy()

            for b, (z0, y0, x0) in enumerate(batch_windows):
                prob_vol[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs[b]
                count_vol[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0

    # 避免除 0
    count_vol[count_vol == 0] = 1.0
    prob_vol /= count_vol

    pred = np.argmax(prob_vol, axis=0).astype(np.int64)  # (Z, Y, X)
    return pred


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    """
    pred, target: (Z, Y, X), 0/1
    """
    pred_fg = (pred > 0).astype(np.float32)
    tgt_fg = (target > 0).astype(np.float32)
    inter = np.sum(pred_fg * tgt_fg)
    union = np.sum(pred_fg) + np.sum(tgt_fg)
    dice = (2 * inter + eps) / (union + eps)
    return float(dice)


# ----------------- 载入模型 ----------------- #

def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Tuple[int, int, int]]:
    """
    根据训练保存的 checkpoint 恢复 UNet3D 模型，并返回模型 + patch_size。
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)
    patch_size = tuple(ckpt.get("patch_size", (128, 128, 128)))

    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base_filters=32, dropout_p=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"[CKPT] Loaded {ckpt_path.name}: in_channels={in_channels}, num_classes={num_classes}, patch_size={patch_size}")
    return model, patch_size


# ----------------- 主推理逻辑 ----------------- #

def run_two_stage_inference_for_case(
    preproc_dir: Path,
    case_id: str,
    liver_model: torch.nn.Module,
    liver_patch_size: Tuple[int, int, int],
    tumor_model: torch.nn.Module,
    tumor_patch_size: Tuple[int, int, int],
    device: torch.device,
    out_dir: Path,
    overlap: float = 0.5,
):
    """
    对单个 case 执行两阶段推理。
    """
    print(f"\n===== Case: {case_id} =====")
    img, seg, _ = load_case_np(preproc_dir, case_id)  # img: (C, Z, Y, X)

    C, Z, Y, X = img.shape
    print(f"[INFO] Image shape: C={C}, Z={Z}, Y={Y}, X={X}")

    # ---------- 阶段 1：肝脏分割 ---------- #
    print("[Stage 1] Liver segmentation...")
    liver_pred = sliding_window_predict(
        img=img,
        model=liver_model,
        device=device,
        patch_size=liver_patch_size,
        num_classes=2,      # liver 模型是二分类：0 背景, 1 肝+瘤
        batch_size=1,
        overlap=overlap,
    )  # (Z, Y, X)
    liver_mask = (liver_pred == 1)

    # ---------- 提取肝脏 ROI ---------- #
    bbox = get_liver_bbox_from_mask(liver_mask, margin=10)
    if bbox is None:
        print("[WARN] Liver ROI not found, tumor stage will run on full volume.")
        roi_zmin, roi_zmax = 0, Z - 1
        roi_ymin, roi_ymax = 0, Y - 1
        roi_xmin, roi_xmax = 0, X - 1
    else:
        roi_zmin, roi_zmax, roi_ymin, roi_ymax, roi_xmin, roi_xmax = bbox
    print(f"[ROI] z: [{roi_zmin}, {roi_zmax}], y: [{roi_ymin}, {roi_ymax}], x: [{roi_xmin}, {roi_xmax}]")

    # ---------- 阶段 2：肿瘤分割（在 ROI 内） ---------- #
    print("[Stage 2] Tumor segmentation within liver ROI...")
    img_roi = img[:, roi_zmin:roi_zmax+1, roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1]

    tumor_pred_roi = sliding_window_predict(
        img=img_roi,
        model=tumor_model,
        device=device,
        patch_size=tumor_patch_size,
        num_classes=2,      # tumor 模型是二分类：0 非瘤, 1 瘤
        batch_size=1,
        overlap=overlap,
    )  # (Z_roi, Y_roi, X_roi)

    # 把 ROI 内的肿瘤预测贴回整幅
    tumor_pred_full = np.zeros((Z, Y, X), dtype=np.int64)
    tumor_pred_full[roi_zmin:roi_zmax+1, roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1] = tumor_pred_roi
    tumor_mask = (tumor_pred_full == 1)

    # ---------- 组合成 3 类标签 ---------- #
    # 0: 背景
    # 1: 肝脏（含肿瘤）
    # 2: 肿瘤
    pred_3class = np.zeros((Z, Y, X), dtype=np.int16)
    pred_3class[liver_mask] = 1
    pred_3class[tumor_mask & liver_mask] = 2   # 确保肿瘤在肝脏内部

    # ---------- 如果有 GT，算一下 Dice ---------- #
    if seg is not None:
        liver_gt = (seg >= 1)     # GT 肝脏：1 或 2
        tumor_gt = (seg == 2)     # GT 肿瘤：2

        liver_dice = binary_dice(liver_mask.astype(np.int32), liver_gt.astype(np.int32))
        tumor_dice = binary_dice(tumor_mask.astype(np.int32), tumor_gt.astype(np.int32))
        print(f"[METRIC] Liver Dice = {liver_dice:.4f}")
        print(f"[METRIC] Tumor Dice = {tumor_dice:.4f}")
    else:
        print("[INFO] No GT segmentation found (.b2nd), skip Dice evaluation.")

    # ---------- 保存结果 ---------- #
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{case_id}_pred_liver.npy", liver_mask.astype(np.uint8))
    np.save(out_dir / f"{case_id}_pred_tumor.npy", tumor_mask.astype(np.uint8))
    np.save(out_dir / f"{case_id}_pred_3class.npy", pred_3class)

    print(f"[SAVE] Liver mask -> {out_dir / (case_id + '_pred_liver.npy')}")
    print(f"[SAVE] Tumor mask -> {out_dir / (case_id + '_pred_tumor.npy')}")
    print(f"[SAVE] 3-class seg -> {out_dir / (case_id + '_pred_3class.npy')}")


# ----------------- CLI ----------------- #

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preproc_dir",
        type=str,
        required=True,
        help="nnUNetv2 预处理目录，例如 Dataset003_Liver/nnUNetPlans_3d_fullres",
    )
    parser.add_argument(
        "--ckpt_liver",
        type=str,
        required=True,
        help="肝脏分割模型 checkpoint 路径，例如 train_logs/.../liver_best.pth",
    )
    parser.add_argument(
        "--ckpt_tumor",
        type=str,
        required=True,
        help="肿瘤分割模型 checkpoint 路径，例如 train_logs/.../tumor_best.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="推理结果保存目录",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="单个 case_id（不填则处理 preproc_dir 下所有 *.pkl 对应病例）",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="滑窗 overlap 比例，0~1，默认 0.5 (步长约 patch_size 一半)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="使用的 GPU id（单机多卡时）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    ckpt_liver = Path(args.ckpt_liver)
    ckpt_tumor = Path(args.ckpt_tumor)

    # 载入 liver / tumor 模型
    liver_model, liver_patch = load_model_from_ckpt(ckpt_liver, device)
    tumor_model, tumor_patch = load_model_from_ckpt(ckpt_tumor, device)

    # 决定要推理哪些 case
    if args.case_id is not None:
        case_ids = [args.case_id]
    else:
        # 用 .pkl 名称作为 case_id（与训练时一致）
        all_pkl = sorted(preproc_dir.glob("*.pkl"))
        case_ids = [p.stem for p in all_pkl]
    print(f"[INFO] Will run inference for {len(case_ids)} cases.")

    for cid in case_ids:
        run_two_stage_inference_for_case(
            preproc_dir=preproc_dir,
            case_id=cid,
            liver_model=liver_model,
            liver_patch_size=liver_patch,
            tumor_model=tumor_model,
            tumor_patch_size=tumor_patch,
            device=device,
            out_dir=out_dir,
            overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
