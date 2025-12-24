#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage inference for LiTS (nnUNetv2 preprocessed .b2nd):

Stage 1: liver segmentation (NEW weights, NEW model: use_coords=True, use_sdf_head=True)
Stage 2: tumor segmentation within liver ROI (OLD weights, OLD model: use_coords=False, use_sdf_head=False)

Outputs:
  - <out_dir>/<case_id>_pred_liver.npy   (0/1) liver mask
  - <out_dir>/<case_id>_pred_tumor.npy   (0/1) tumor mask
  - <out_dir>/<case_id>_pred_3class.npy  (0/1/2) final label map

If GT seg exists (case_id_seg.b2nd), prints liver/tumor Dice.
"""

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import blosc2
import numpy as np
import torch
import torch.nn.functional as F

from models.unet3D import UNet3D


# ----------------- Utils ----------------- #

def get_logits(model_out: object) -> torch.Tensor:
    """
    Compatible with:
      - NEW model: dict {"logits": Tensor, "sdf": Tensor(optional)}
      - OLD model: Tensor logits
      - OLD model: tuple/list (logits, ...)
    """
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


def load_case_np(preproc_dir: Path, case_id: str):
    """
    Read nnUNetv2 preprocessed case:
      img: (C, Z, Y, X), float32
      seg: (Z, Y, X), int16 if exists else None
      properties: loaded from .pkl if exists else None
    """
    dparams = {"nthreads": 1}
    img_file = preproc_dir / f"{case_id}.b2nd"
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    prop_file = preproc_dir / f"{case_id}.pkl"

    if not img_file.is_file():
        raise FileNotFoundError(img_file)

    img_b = blosc2.open(urlpath=str(img_file), mode="r", dparams=dparams)
    img = img_b[:].astype(np.float32)  # (C,Z,Y,X)

    seg = None
    if seg_file.is_file():
        seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)
        seg_arr = seg_b[:].astype(np.int16)
        if seg_arr.ndim == 4:
            seg_arr = seg_arr[0]
        seg = seg_arr  # (Z,Y,X)

    properties = None
    if prop_file.is_file():
        import pickle as pkl
        with open(prop_file, "rb") as f:
            properties = pkl.load(f)

    return img, seg, properties


def binary_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-5) -> float:
    pred_fg = (pred > 0).astype(np.float32)
    tgt_fg = (target > 0).astype(np.float32)
    inter = np.sum(pred_fg * tgt_fg)
    union = np.sum(pred_fg) + np.sum(tgt_fg)
    return float((2 * inter + eps) / (union + eps))


def get_liver_bbox_from_mask(mask: np.ndarray, margin: int = 10) -> Optional[Tuple[int, int, int, int, int, int]]:
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
    Z: int, Y: int, X: int,
    pz: int, py: int, px: int,
    overlap: float = 0.5
) -> List[Tuple[int, int, int]]:
    assert 0 <= overlap < 1.0

    def _starts(dim: int, p: int) -> List[int]:
        if dim <= p:
            return [0]
        step = int(p * (1 - overlap))
        step = max(1, step)
        starts = list(range(0, dim - p + 1, step))
        if starts[-1] != dim - p:
            starts.append(dim - p)
        return starts

    z_starts = _starts(Z, pz)
    y_starts = _starts(Y, py)
    x_starts = _starts(X, px)

    return [(z0, y0, x0) for z0 in z_starts for y0 in y_starts for x0 in x_starts]


def sliding_window_predict(
    img: np.ndarray,                 # (C,Z,Y,X)
    model: torch.nn.Module,
    device: torch.device,
    patch_size: Tuple[int, int, int],
    num_classes: int,
    batch_size: int = 1,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Sliding-window inference for a full volume.
    Returns: pred (Z,Y,X) int64
    """
    model.eval()
    C, Z, Y, X = img.shape
    pz, py, px = patch_size

    windows = make_sliding_windows(Z, Y, X, pz, py, px, overlap=overlap)

    prob_vol = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_vol = np.zeros((1, Z, Y, X), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i + batch_size]
            batch_imgs = []
            for (z0, y0, x0) in batch_windows:
                patch = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
                batch_imgs.append(patch)

            batch_arr = np.stack(batch_imgs, axis=0)  # (B,C,pz,py,px)
            batch_tensor = torch.from_numpy(batch_arr).to(device=device, dtype=torch.float32)

            out = model(batch_tensor)
            logits = get_logits(out)  # (B,num_classes,pz,py,px)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

            for b, (z0, y0, x0) in enumerate(batch_windows):
                prob_vol[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs[b]
                count_vol[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0

    count_vol[count_vol == 0] = 1.0
    prob_vol /= count_vol
    pred = np.argmax(prob_vol, axis=0).astype(np.int64)
    return pred


# ----------------- Model loading (NEW liver, OLD tumor) ----------------- #

def load_liver_model_new(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Tuple[int, int, int]]:
    """
    Liver uses NEW model structure: use_coords=True, use_sdf_head=True.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)
    patch_size = tuple(ckpt.get("patch_size", (128, 128, 128)))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=True,
        use_sdf_head=True,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"[LIVER-NEW] Loaded {ckpt_path.name}: in_ch={in_channels}, classes={num_classes}, patch={patch_size}")
    return model, patch_size


def load_tumor_model_old(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Tuple[int, int, int]]:
    """
    Tumor uses OLD model structure: use_coords=False, use_sdf_head=False.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)
    patch_size = tuple(ckpt.get("patch_size", (128, 128, 128)))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=False,
        use_sdf_head=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    print(f"[TUMOR-OLD] Loaded {ckpt_path.name}: in_ch={in_channels}, classes={num_classes}, patch={patch_size}")
    return model, patch_size


# ----------------- Two-stage inference ----------------- #

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
    roi_margin: int = 10,
):
    print(f"\n===== Case: {case_id} =====")
    img, seg, _ = load_case_np(preproc_dir, case_id)  # img: (C,Z,Y,X)
    C, Z, Y, X = img.shape
    print(f"[INFO] Image shape: C={C}, Z={Z}, Y={Y}, X={X}")

    # ---- Stage 1: liver (NEW) ----
    print("[Stage 1] Liver segmentation (NEW)...")
    liver_pred = sliding_window_predict(
        img=img,
        model=liver_model,
        device=device,
        patch_size=liver_patch_size,
        num_classes=2,
        batch_size=1,
        overlap=overlap,
    )
    liver_mask = (liver_pred == 1)

    # ROI
    bbox = get_liver_bbox_from_mask(liver_mask.astype(np.uint8), margin=roi_margin)
    if bbox is None:
        print("[WARN] Liver ROI not found -> run tumor on full volume.")
        roi_zmin, roi_zmax = 0, Z - 1
        roi_ymin, roi_ymax = 0, Y - 1
        roi_xmin, roi_xmax = 0, X - 1
    else:
        roi_zmin, roi_zmax, roi_ymin, roi_ymax, roi_xmin, roi_xmax = bbox
    print(f"[ROI] z:[{roi_zmin},{roi_zmax}] y:[{roi_ymin},{roi_ymax}] x:[{roi_xmin},{roi_xmax}]")

    # ---- Stage 2: tumor (OLD) within ROI ----
    print("[Stage 2] Tumor segmentation (OLD) within ROI...")
    img_roi = img[:, roi_zmin:roi_zmax+1, roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1]

    tumor_pred_roi = sliding_window_predict(
        img=img_roi,
        model=tumor_model,
        device=device,
        patch_size=tumor_patch_size,
        num_classes=2,
        batch_size=1,
        overlap=overlap,
    )

    tumor_pred_full = np.zeros((Z, Y, X), dtype=np.int64)
    tumor_pred_full[roi_zmin:roi_zmax+1, roi_ymin:roi_ymax+1, roi_xmin:roi_xmax+1] = tumor_pred_roi
    tumor_mask = (tumor_pred_full == 1)

    # ---- Compose 3-class ----
    pred_3class = np.zeros((Z, Y, X), dtype=np.int16)
    pred_3class[liver_mask] = 1
    pred_3class[tumor_mask & liver_mask] = 2  # ensure tumor inside liver

    # ---- Metrics if GT available ----
    if seg is not None:
        liver_gt = (seg >= 1)
        tumor_gt = (seg == 2)

        liver_dice = binary_dice(liver_mask.astype(np.uint8), liver_gt.astype(np.uint8))
        tumor_dice = binary_dice(tumor_mask.astype(np.uint8), tumor_gt.astype(np.uint8))
        print(f"[METRIC] Liver Dice = {liver_dice:.4f}")
        print(f"[METRIC] Tumor Dice = {tumor_dice:.4f}")
    else:
        print("[INFO] No GT seg found, skip Dice.")

    # ---- Save ----
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{case_id}_pred_liver.npy", liver_mask.astype(np.uint8))
    np.save(out_dir / f"{case_id}_pred_tumor.npy", tumor_mask.astype(np.uint8))
    np.save(out_dir / f"{case_id}_pred_3class.npy", pred_3class)

    print(f"[SAVE] {out_dir / (case_id + '_pred_liver.npy')}")
    print(f"[SAVE] {out_dir / (case_id + '_pred_tumor.npy')}")
    print(f"[SAVE] {out_dir / (case_id + '_pred_3class.npy')}")


# ----------------- CLI ----------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--ckpt_liver", type=str, required=True, help="NEW liver ckpt")
    p.add_argument("--ckpt_tumor", type=str, required=True, help="OLD tumor ckpt")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--case_id", type=str, default=None)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--roi_margin", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    ckpt_liver = Path(args.ckpt_liver)
    ckpt_tumor = Path(args.ckpt_tumor)

    liver_model, liver_patch = load_liver_model_new(ckpt_liver, device)
    tumor_model, tumor_patch = load_tumor_model_old(ckpt_tumor, device)

    if args.case_id is not None:
        case_ids = [args.case_id]
    else:
        all_pkl = sorted(preproc_dir.glob("*.pkl"))
        case_ids = [p.stem for p in all_pkl]
    print(f"[INFO] Will run inference for {len(case_ids)} case(s).")

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
            roi_margin=args.roi_margin,
        )


if __name__ == "__main__":
    main()
