#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-volume liver evaluation with sliding-window inference.

Modes:
1) Single-stage: --ckpt <liver_ckpt>
2) Cascade: --ckpt_coarse <coarse_ckpt> --ckpt_refine <refine_ckpt>
   (coarse -> pred bbox -> refine in ROI -> paste back)

Notes:
- If your refine liver was trained with GT bbox (liver_use_bbox=1), evaluating it on full volume
  often fails due to distribution shift. Use cascade evaluation, or use --use_gt_bbox to test refine alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet3D import UNet3D


# -------------------- common utils -------------------- #

def make_coords_zyx(shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    Z, Y, X = shape_zyx
    zz = np.linspace(-1.0, 1.0, Z, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, Y, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, X, dtype=np.float32)[None, None, :]
    zc = np.broadcast_to(zz, (Z, Y, X))
    yc = np.broadcast_to(yy, (Z, Y, X))
    xc = np.broadcast_to(xx, (Z, Y, X))
    return np.stack([zc, yc, xc], axis=0)


def add_coords_if_needed(img_czyx: np.ndarray, need_coords: bool) -> np.ndarray:
    if not need_coords:
        return img_czyx
    coords = make_coords_zyx(img_czyx.shape[1:])
    return np.concatenate([img_czyx, coords], axis=0).astype(np.float32)


def _try_import_scipy():
    try:
        from scipy.ndimage import label as ndi_label, binary_fill_holes
        return ndi_label, binary_fill_holes
    except Exception:
        return None, None


def keep_largest_cc(mask_zyx: np.ndarray) -> np.ndarray:
    ndi_label, _ = _try_import_scipy()
    if ndi_label is None:
        return mask_zyx.astype(np.uint8)
    lab, n = ndi_label(mask_zyx.astype(np.uint8))
    if n == 0:
        return mask_zyx.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


def fill_holes(mask_zyx: np.ndarray) -> np.ndarray:
    _, binary_fill_holes = _try_import_scipy()
    if binary_fill_holes is None:
        return mask_zyx.astype(np.uint8)
    return binary_fill_holes(mask_zyx.astype(bool)).astype(np.uint8)


def postprocess_liver(mask01_zyx: np.ndarray) -> np.ndarray:
    m = (mask01_zyx > 0).astype(np.uint8)
    m = keep_largest_cc(m)
    m = fill_holes(m)
    m = keep_largest_cc(m)
    return m


def get_bbox_from_mask(mask_zyx: np.ndarray, margin: int = 16) -> Optional[Tuple[int, int, int, int, int, int]]:
    zz, yy, xx = np.where(mask_zyx > 0)
    if len(zz) == 0:
        return None
    z0, z1 = int(zz.min()), int(zz.max())
    y0, y1 = int(yy.min()), int(yy.max())
    x0, x1 = int(xx.min()), int(xx.max())
    Z, Y, X = mask_zyx.shape
    z0 = max(0, z0 - margin); z1 = min(Z - 1, z1 + margin)
    y0 = max(0, y0 - margin); y1 = min(Y - 1, y1 + margin)
    x0 = max(0, x0 - margin); x1 = min(X - 1, x1 + margin)
    return z0, z1, y0, y1, x0, x1


def crop_with_bbox(img_czyx: np.ndarray, bbox) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return img_czyx[:, z0:z1+1, y0:y1+1, x0:x1+1]


def paste_roi(full_zyx: np.ndarray, roi_zyx: np.ndarray, bbox):
    z0, z1, y0, y1, x0, x1 = bbox
    full_zyx[z0:z1+1, y0:y1+1, x0:x1+1] = roi_zyx


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    pred_fg = (pred > 0).astype(np.float32)
    tgt_fg = (target > 0).astype(np.float32)
    inter = (pred_fg * tgt_fg).sum()
    union = pred_fg.sum() + tgt_fg.sum()
    return float((2.0 * inter + smooth) / (union + smooth))


# -------------------- data I/O -------------------- #

def list_case_ids(preproc_dir: Path, split: str, train_ratio: float, seed: int) -> List[str]:
    all_pkl = sorted(preproc_dir.glob("*.pkl"))
    case_ids = [p.stem for p in all_pkl]
    if not case_ids:
        raise RuntimeError(f"No .pkl files found in {preproc_dir}")

    import random
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n_train = int(len(case_ids) * train_ratio)

    if split == "all":
        return case_ids
    if split == "train":
        return case_ids[:n_train]
    return case_ids[n_train:]


def load_case_b2nd(preproc_dir: Path, case_id: str):
    dparams = {"nthreads": 1}
    data_file = preproc_dir / f"{case_id}.b2nd"
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    if not seg_file.is_file():
        raise FileNotFoundError(seg_file)

    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)

    img = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    seg = seg_b[:].astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return img, seg


# -------------------- model / sliding window -------------------- #

def _hann_3d(patch_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    pz, py, px = patch_size
    wz = torch.hann_window(pz, periodic=False, device=device).clamp_min(1e-3)
    wy = torch.hann_window(py, periodic=False, device=device).clamp_min(1e-3)
    wx = torch.hann_window(px, periodic=False, device=device).clamp_min(1e-3)
    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    return w[None, None, ...]


def get_logits(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


def sliding_window_prob(
    volume: torch.Tensor,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int = 2,
) -> torch.Tensor:
    model.eval()
    _, _, Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride
    device = volume.device

    prob_map = torch.zeros((num_classes, Z, Y, X), dtype=torch.float32, device=device)
    weight_map = torch.zeros((1, Z, Y, X), dtype=torch.float32, device=device)
    w_patch = _hann_3d(patch_size, device=device)

    z_starts = list(range(0, max(Z - pz + 1, 1), sz))
    y_starts = list(range(0, max(Y - py + 1, 1), sy))
    x_starts = list(range(0, max(X - px + 1, 1), sx))
    if z_starts[-1] + pz < Z: z_starts.append(Z - pz)
    if y_starts[-1] + py < Y: y_starts.append(Y - py)
    if x_starts[-1] + px < X: x_starts.append(X - px)

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    z1, y1, x1 = z0 + pz, y0 + py, x0 + px
                    patch = volume[:, :, z0:z1, y0:y1, x0:x1]
                    logits = get_logits(model(patch))
                    probs = F.softmax(logits, dim=1)
                    prob_map[:, z0:z1, y0:y1, x0:x1] += (probs[0] * w_patch[0])
                    weight_map[:, z0:z1, y0:y1, x0:x1] += w_patch[0, 0]

    prob_map = prob_map / torch.clamp_min(weight_map, 1e-6)
    return prob_map


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    need_coords = bool(ckpt.get("add_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    use_se = bool(ckpt.get("use_se", False))
    patch_size = tuple(ckpt.get("patch_size", (128, 128, 128)))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=float(ckpt.get("dropout_p", 0.0)),
        use_sdf_head=use_sdf_head,
        use_se=use_se,
        use_coords=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = {"need_coords": need_coords, "num_classes": num_classes, "patch_size": patch_size}
    return model, meta


# -------------------- args / main -------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)

    # single-stage
    p.add_argument("--ckpt", type=str, default=None)

    # cascade
    p.add_argument("--ckpt_coarse", type=str, default=None)
    p.add_argument("--ckpt_refine", type=str, default=None)

    p.add_argument("--out_dir", type=str, default="eval_liver_full")
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--split", type=str, choices=["all", "train", "val"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cases", type=int, default=-1)
    p.add_argument("--save_npy", action="store_true")

    p.add_argument("--bbox_margin", type=int, default=24, help="for cascade: margin around coarse-pred bbox")
    p.add_argument("--use_gt_bbox", action="store_true", help="evaluate refine with GT bbox (upper bound)")
    return p.parse_args()


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cascade = args.ckpt_coarse is not None and args.ckpt_refine is not None
    if not use_cascade and args.ckpt is None:
        raise ValueError("Provide --ckpt (single) or --ckpt_coarse + --ckpt_refine (cascade).")

    if use_cascade:
        model_c, meta_c = load_model(Path(args.ckpt_coarse), device)
        model_r, meta_r = load_model(Path(args.ckpt_refine), device)
        print(f"=> Cascade mode\n   coarse={args.ckpt_coarse}\n   refine={args.ckpt_refine}")
        print(f"   refine need_coords={meta_r['need_coords']}, patch_size={meta_r['patch_size']}")
    else:
        model, meta = load_model(Path(args.ckpt), device)
        print(f"=> Single-stage mode\n   ckpt={args.ckpt}")
        print(f"   need_coords={meta['need_coords']}, patch_size={meta['patch_size']}")

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    print(f"Found {len(case_ids)} case(s) for split='{args.split}'.")

    dice_list: List[float] = []

    for case_id in tqdm(case_ids, desc="Evaluating", unit="case"):
        img_np, seg_np = load_case_b2nd(preproc_dir, case_id)
        liver_gt = (seg_np >= 1).astype(np.uint8)

        if use_cascade:
            img_c = add_coords_if_needed(img_np, meta_c["need_coords"])
            vol_c = torch.from_numpy(img_c[None, ...]).to(device)
            prob_c = sliding_window_prob(vol_c, model_c, patch_size=patch_size, stride=stride, num_classes=meta_c["num_classes"])
            pred_c = prob_c.argmax(dim=0).cpu().numpy().astype(np.uint8)
            liver_c = postprocess_liver(pred_c)

            if args.use_gt_bbox:
                bbox = get_bbox_from_mask(liver_gt, margin=args.bbox_margin)
            else:
                bbox = get_bbox_from_mask(liver_c, margin=args.bbox_margin)

            if bbox is None:
                liver_pred = np.zeros_like(liver_gt, dtype=np.uint8)
            else:
                img_roi = crop_with_bbox(img_np, bbox)
                img_r = add_coords_if_needed(img_roi, meta_r["need_coords"])
                vol_r = torch.from_numpy(img_r[None, ...]).to(device)
                prob_r = sliding_window_prob(vol_r, model_r, patch_size=patch_size, stride=stride, num_classes=meta_r["num_classes"])
                pred_r = prob_r.argmax(dim=0).cpu().numpy().astype(np.uint8)
                liver_roi = postprocess_liver(pred_r)

                liver_pred = np.zeros_like(liver_gt, dtype=np.uint8)
                paste_roi(liver_pred, liver_roi, bbox)
                liver_pred = postprocess_liver(liver_pred)
        else:
            img_in = add_coords_if_needed(img_np, meta["need_coords"])
            vol = torch.from_numpy(img_in[None, ...]).to(device)
            prob = sliding_window_prob(vol, model, patch_size=patch_size, stride=stride, num_classes=meta["num_classes"])
            pred_label = prob.argmax(dim=0).cpu().numpy().astype(np.uint8)
            liver_pred = postprocess_liver(pred_label)

        d = dice_score(liver_pred, liver_gt)
        dice_list.append(d)

        if d < 1e-6:
            pred_fg = int((liver_pred > 0).sum())
            gt_fg = int((liver_gt > 0).sum())
            inter = int(((liver_pred > 0) & (liver_gt > 0)).sum())
            uniq = np.unique(liver_pred)
            print(f"  [DBG] {case_id}: pred_fg={pred_fg}, gt_fg={gt_fg}, inter={inter}, unique_pred={uniq}")

        print(f"  Case {case_id}: Dice={d:.4f}")

        if args.save_npy:
            np.save(out_dir / f"{case_id}_liver_pred.npy", liver_pred)
            np.save(out_dir / f"{case_id}_liver_gt.npy", liver_gt)

    dice_arr = np.array(dice_list, dtype=np.float32)
    mean_dice = float(dice_arr.mean()) if len(dice_arr) > 0 else 0.0
    std_dice = float(dice_arr.std()) if len(dice_arr) > 0 else 0.0

    print("=" * 60)
    print(f"[Liver full-volume Dice] split='{args.split}', cases={len(dice_list)}")
    print(f"  Mean Dice = {mean_dice:.4f}, Std = {std_dice:.4f}")
    print("=" * 60)

    with (out_dir / "liver_eval_results.txt").open("w") as f:
        for cid, d in zip(case_ids, dice_list):
            f.write(f"{cid}\t{d:.6f}\n")
        f.write(f"\nMean Dice = {mean_dice:.6f}, Std = {std_dice:.6f}\n")


if __name__ == "__main__":
    main()
