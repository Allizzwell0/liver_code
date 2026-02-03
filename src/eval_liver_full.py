#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-volume liver evaluation (Dice) on nnUNetv2-preprocessed LiTS (.b2nd).

Supports:
- Single-stage:  --ckpt <liver_ckpt>
- Cascade:       --ckpt_coarse <coarse_ckpt> --ckpt_refine <refine_ckpt>
  (coarse -> bbox -> refine in ROI -> paste back)

This version is compatible with the provided UNet3D that can internally append coords
(use_coords flag stored in checkpoint).
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


# ----------------- postprocess ----------------- #

def _try_import_scipy():
    try:
        from scipy.ndimage import label as ndi_label, binary_fill_holes
        return ndi_label, binary_fill_holes
    except Exception:
        return None, None


def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    ndi_label, _ = _try_import_scipy()
    if ndi_label is None:
        return mask.astype(np.uint8)
    lab, n = ndi_label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return (lab == sizes.argmax()).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    _, binary_fill_holes = _try_import_scipy()
    if binary_fill_holes is None:
        return mask.astype(np.uint8)
    return binary_fill_holes(mask.astype(bool)).astype(np.uint8)


def postprocess_liver(mask01_zyx: np.ndarray) -> np.ndarray:
    m = (mask01_zyx > 0).astype(np.uint8)
    m = keep_largest_cc(m)
    m = fill_holes(m)
    m = keep_largest_cc(m)
    return m


def bbox_from_mask(mask: np.ndarray, margin: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    idx = np.where(mask > 0)
    if idx[0].size == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    x0, x1 = int(idx[2].min()), int(idx[2].max())
    Z, Y, X = mask.shape
    z0 = max(0, z0 - margin); z1 = min(Z - 1, z1 + margin)
    y0 = max(0, y0 - margin); y1 = min(Y - 1, y1 + margin)
    x0 = max(0, x0 - margin); x1 = min(X - 1, x1 + margin)
    return z0, z1, y0, y1, x0, x1


# ----------------- io ----------------- #

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


# ----------------- model / sliding window ----------------- #

def get_logits(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


def pad_to_min(volume: torch.Tensor, patch: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    # volume: (1,C,Z,Y,X)
    _, _, Z, Y, X = volume.shape
    pz, py, px = patch
    pad_z = max(0, pz - Z)
    pad_y = max(0, py - Y)
    pad_x = max(0, px - X)
    if pad_z or pad_y or pad_x:
        volume = F.pad(volume, (0, pad_x, 0, pad_y, 0, pad_z))
    return volume, (Z, Y, X)


def sliding_window_prob(
    volume: torch.Tensor,  # (1,C,Z,Y,X)
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int,
) -> torch.Tensor:
    model.eval()
    volume, orig = pad_to_min(volume, patch_size)
    _, _, Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    prob = torch.zeros((num_classes, Z, Y, X), device=volume.device, dtype=torch.float32)
    w = torch.zeros((1, Z, Y, X), device=volume.device, dtype=torch.float32)

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
                    patch = volume[:, :, z0:z0+pz, y0:y0+py, x0:x0+px]
                    logits = get_logits(model(patch))
                    probs = torch.softmax(logits, dim=1)[0]  # (C,Z,Y,X)
                    prob[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs
                    w[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0

    prob = prob / torch.clamp_min(w, 1.0)
    oZ, oY, oX = orig
    return prob[:, :oZ, :oY, :oX]


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_coords = bool(ckpt.get("use_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    use_attn_gate = bool(ckpt.get("use_attn_gate", False))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=float(ckpt.get("dropout_p", 0.0)),
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
        use_attn_gate=use_attn_gate,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = {"in_channels": in_channels, "num_classes": num_classes, "use_coords": use_coords, "use_sdf_head": use_sdf_head, "use_attn_gate": use_attn_gate}
    return model, meta


def dice_score(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-5) -> float:
    pred = (pred01 > 0).astype(np.float32)
    gt = (gt01 > 0).astype(np.float32)
    inter = float((pred * gt).sum())
    union = float(pred.sum() + gt.sum())
    return float((2.0 * inter + smooth) / (union + smooth))


# ----------------- main ----------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)

    p.add_argument("--ckpt", type=str, default=None, help="single-stage ckpt (optional)")
    p.add_argument("--ckpt_coarse", type=str, default=None, help="cascade coarse ckpt")
    p.add_argument("--ckpt_refine", type=str, default=None, help="cascade refine ckpt")

    p.add_argument("--out_dir", type=str, default="eval_liver_full")
    p.add_argument("--split", type=str, choices=["all", "train", "val"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cases", type=int, default=-1)

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64])

    p.add_argument("--bbox_margin", type=int, default=24, help="cascade: margin around coarse-pred bbox")
    p.add_argument("--thr", type=float, default=0.5, help="binarize prob threshold")
    p.add_argument("--use_gt_bbox", action="store_true", help="use GT liver bbox for refine (upper bound)")
    p.add_argument("--save_npy", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cascade = (args.ckpt_coarse is not None) and (args.ckpt_refine is not None)
    if (not use_cascade) and (args.ckpt is None):
        raise ValueError("Provide --ckpt (single) or --ckpt_coarse + --ckpt_refine (cascade).")

    model_c = meta_c = model_r = meta_r = None
    if use_cascade:
        model_c, meta_c = load_model(args.ckpt_coarse, device)
        model_r, meta_r = load_model(args.ckpt_refine, device)
        print("=> Cascade mode")
        print(f"   coarse={args.ckpt_coarse}")
        print(f"   refine={args.ckpt_refine} (use_coords={meta_r['use_coords']})")
    else:
        model, meta = load_model(args.ckpt, device)
        print("=> Single-stage mode")
        print(f"   ckpt={args.ckpt} (use_coords={meta['use_coords']})")

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    print(f"Found {len(case_ids)} case(s) for split='{args.split}'.")

    dices: List[float] = []

    for cid in tqdm(case_ids, desc="Eval-Liver", unit="case"):
        img_np, seg_np = load_case_b2nd(preproc_dir, cid)
        liver_gt = (seg_np > 0).astype(np.uint8)

        if use_cascade:
            vol = torch.from_numpy(img_np[None, ...]).to(device)
            prob_c = sliding_window_prob(vol, model_c, patch_size, stride, num_classes=meta_c["num_classes"])
            prob_liver_c = prob_c[1].detach().cpu().numpy()
            liver_c = postprocess_liver((prob_liver_c > args.thr).astype(np.uint8))

            if args.use_gt_bbox:
                b = bbox_from_mask(liver_gt, margin=int(args.bbox_margin))
            else:
                b = bbox_from_mask(liver_c, margin=int(args.bbox_margin))

            if b is None:
                liver_pred = np.zeros_like(liver_gt, dtype=np.uint8)
            else:
                z0, z1, y0, y1, x0, x1 = b
                img_roi = img_np[:, z0:z1+1, y0:y1+1, x0:x1+1]
                vol_roi = torch.from_numpy(img_roi[None, ...]).to(device)
                prob_r = sliding_window_prob(vol_roi, model_r, patch_size, stride, num_classes=meta_r["num_classes"])
                prob_liver_r = prob_r[1].detach().cpu().numpy()

                liver_pred = np.zeros_like(liver_gt, dtype=np.uint8)
                liver_pred[z0:z1+1, y0:y1+1, x0:x1+1] = (prob_liver_r > args.thr).astype(np.uint8)
                liver_pred = postprocess_liver(liver_pred)
        else:
            vol = torch.from_numpy(img_np[None, ...]).to(device)
            prob = sliding_window_prob(vol, model, patch_size, stride, num_classes=meta["num_classes"])
            prob_liver = prob[1].detach().cpu().numpy()
            liver_pred = postprocess_liver((prob_liver > args.thr).astype(np.uint8))

        d = dice_score(liver_pred, liver_gt)
        dices.append(d)
        print(f"  {cid}: Dice={d:.4f}")

        if args.save_npy:
            np.save(out_dir / f"{cid}_liver_pred.npy", liver_pred.astype(np.uint8))
            np.save(out_dir / f"{cid}_liver_gt.npy", liver_gt.astype(np.uint8))

    dices_np = np.array(dices, dtype=np.float32)
    print("=" * 60)
    print(f"[Liver Dice] split={args.split} cases={len(dices)}")
    print(f"  Mean={float(dices_np.mean()):.4f}  Std={float(dices_np.std()):.4f}")
    print("=" * 60)

    with (out_dir / "liver_eval_results.txt").open("w") as f:
        for cid, d in zip(case_ids, dices):
            f.write(f"{cid}\t{d:.6f}\n")
        f.write(f"\nMean\t{float(dices_np.mean()):.6f}\nStd\t{float(dices_np.std()):.6f}\n")


if __name__ == "__main__":
    main()
