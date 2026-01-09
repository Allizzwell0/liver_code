#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate liver segmentation on LiTS (.b2nd) with single or cascade liver model(s).

Examples:
  # Single model
  python eval_liver_full.py --preproc_dir <preproc> --ckpt_liver <ckpt> --split val

  # Cascade
  python eval_liver_full.py --preproc_dir <preproc> --ckpt_coarse <coarse> --ckpt_refine <refine> --split val --bbox_margin 24

Options:
  --use_gt_bbox : refine stage uses GT liver bbox (upper bound; for debugging)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import blosc2

from models.unet3D import UNet3D


def list_case_ids(preproc_dir: Path) -> List[str]:
    # prefer .pkl existence to match train/val split; fallback to .b2nd
    pkl = sorted(preproc_dir.glob("*.pkl"))
    if pkl:
        return [p.stem for p in pkl]
    files = sorted(preproc_dir.glob("*.b2nd"))
    ids = []
    for f in files:
        if f.name.endswith("_seg.b2nd"):
            continue
        ids.append(f.stem)
    if not ids:
        raise RuntimeError(f"No cases found in: {preproc_dir}")
    return ids


def split_ids(ids: List[str], split: str, train_ratio: float, seed: int) -> List[str]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_train = int(len(ids) * float(train_ratio))
    if split == "train":
        return ids[:n_train]
    if split == "val":
        return ids[n_train:]
    return ids


def load_b2nd(path: Path) -> np.ndarray:
    dparams = {"nthreads": 1}
    b = blosc2.open(urlpath=str(path), mode="r", dparams=dparams)
    return b[:]


def load_case(preproc_dir: Path, case_id: str):
    img = load_b2nd(preproc_dir / f"{case_id}.b2nd").astype(np.float32)  # (C,Z,Y,X)
    seg = load_b2nd(preproc_dir / f"{case_id}_seg.b2nd").astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return img, seg


def bbox_from_mask(mask01: np.ndarray, margin: int = 0):
    idx = np.argwhere(mask01 > 0)
    if idx.size == 0:
        return None
    z0, y0, x0 = idx.min(axis=0).tolist()
    z1, y1, x1 = idx.max(axis=0).tolist()
    Z, Y, X = mask01.shape
    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(Z - 1, z1 + margin)
    y1 = min(Y - 1, y1 + margin)
    x1 = min(X - 1, x1 + margin)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def keep_largest_cc(mask01: np.ndarray) -> np.ndarray:
    try:
        import scipy.ndimage as ndi
        lab, n = ndi.label(mask01.astype(np.uint8))
        if n <= 1:
            return mask01.astype(np.uint8)
        sizes = ndi.sum(mask01.astype(np.uint8), lab, index=np.arange(1, n + 1))
        k = int(np.argmax(sizes) + 1)
        return (lab == k).astype(np.uint8)
    except Exception:
        return mask01.astype(np.uint8)


def make_coords_zyx(shape_zyx: Tuple[int, int, int], device, dtype) -> torch.Tensor:
    Z, Y, X = shape_zyx
    zz = torch.linspace(-1, 1, Z, device=device, dtype=dtype)
    yy = torch.linspace(-1, 1, Y, device=device, dtype=dtype)
    xx = torch.linspace(-1, 1, X, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zz, yy, xx, indexing="ij")
    coords = torch.stack([z, y, x], dim=0)  # (3,Z,Y,X)
    return coords


def add_coords_if_needed(vol: torch.Tensor, need_coords: bool) -> torch.Tensor:
    if not need_coords:
        return vol
    _, _, Z, Y, X = vol.shape
    coords = make_coords_zyx((Z, Y, X), device=vol.device, dtype=vol.dtype).unsqueeze(0)
    return torch.cat([vol, coords], dim=1)


def _starts(dim: int, patch: int, stride: int) -> List[int]:
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def hann_window_3d(patch_size: Tuple[int, int, int], device, dtype) -> torch.Tensor:
    pz, py, px = patch_size
    wz = torch.hann_window(pz, periodic=False, device=device, dtype=dtype)
    wy = torch.hann_window(py, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(px, periodic=False, device=device, dtype=dtype)
    w = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    w = w / (w.max().clamp_min(1e-6))
    return w[None]  # (1,pz,py,px)


@torch.no_grad()
def sliding_window_prob(
    vol_czyx: np.ndarray,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int,
    need_coords: bool,
    device: torch.device,
) -> np.ndarray:
    C, Z0, Y0, X0 = vol_czyx.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    pad_z = max(0, pz - Z0)
    pad_y = max(0, py - Y0)
    pad_x = max(0, px - X0)
    if pad_z or pad_y or pad_x:
        vol_czyx = np.pad(vol_czyx, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant", constant_values=0)

    _, Z, Y, X = vol_czyx.shape
    prob_map = torch.zeros((num_classes, Z, Y, X), device=device, dtype=torch.float32)
    wsum = torch.zeros((1, Z, Y, X), device=device, dtype=torch.float32)
    w_patch = hann_window_3d(patch_size, device=device, dtype=torch.float32)

    z_starts = _starts(Z, pz, sz)
    y_starts = _starts(Y, py, sy)
    x_starts = _starts(X, px, sx)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = vol_czyx[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px]
                inp = torch.from_numpy(patch)[None].to(device=device, dtype=torch.float32)
                inp = add_coords_if_needed(inp, need_coords)
                out = model(inp)
                logits = out["logits"] if isinstance(out, dict) else out
                probs = torch.softmax(logits, dim=1)[0]
                prob_map[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += probs * w_patch
                wsum[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px] += w_patch

    prob_map = prob_map / wsum.clamp_min(1e-6)
    prob_map = prob_map[:, :Z0, :Y0, :X0]
    return prob_map.detach().cpu().numpy().astype(np.float32)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    in_ch = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    use_se = bool(ckpt.get("use_se", False))
    dropout_p = float(ckpt.get("dropout_p", 0.0))
    need_coords = bool(ckpt.get("add_coords", False))

    model = UNet3D(
        in_channels=in_ch,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=dropout_p,
        use_coords=False,
        use_sdf_head=use_sdf_head,
        use_se=use_se,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, {"num_classes": num_classes, "need_coords": need_coords}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)

    p.add_argument("--ckpt_liver", type=str, default=None)
    p.add_argument("--ckpt_coarse", type=str, default=None)
    p.add_argument("--ckpt_refine", type=str, default=None)

    p.add_argument("--split", type=str, choices=["train", "val", "all"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--bbox_margin", type=int, default=24)
    p.add_argument("--use_gt_bbox", action="store_true")
    return p.parse_args()


def dice01(pred01: np.ndarray, gt01: np.ndarray) -> float:
    inter = float((pred01 * gt01).sum())
    union = float(pred01.sum() + gt01.sum())
    return float((2 * inter + 1e-5) / (union + 1e-5))


def main():
    args = parse_args()
    preproc_dir = Path(args.preproc_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = tuple(int(x) for x in args.patch_size)
    stride = tuple(int(x) for x in args.stride)

    cascade = (args.ckpt_coarse is not None and args.ckpt_refine is not None)
    if cascade:
        model_c, meta_c = load_model(args.ckpt_coarse, device)
        model_r, meta_r = load_model(args.ckpt_refine, device)
    else:
        if args.ckpt_liver is None:
            raise ValueError("Provide either --ckpt_liver OR (--ckpt_coarse and --ckpt_refine).")
        model_s, meta_s = load_model(args.ckpt_liver, device)

    ids = split_ids(list_case_ids(preproc_dir), args.split, args.train_ratio, args.seed)
    print(f"[Eval] split={args.split} cases={len(ids)} device={device}")

    dices = []
    for cid in ids:
        img_czyx, seg_zyx = load_case(preproc_dir, cid)
        gt = (seg_zyx > 0).astype(np.uint8)

        if cascade:
            prob_c = sliding_window_prob(img_czyx, model_c, patch_size, stride, meta_c["num_classes"], meta_c["need_coords"], device)
            pred_c = (np.argmax(prob_c, axis=0) > 0).astype(np.uint8)
            bb = bbox_from_mask(gt if args.use_gt_bbox else pred_c, margin=int(args.bbox_margin))
            if bb is None:
                pred = keep_largest_cc(pred_c)
            else:
                z0, z1, y0, y1, x0, x1 = bb
                roi = img_czyx[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
                prob_r_roi = sliding_window_prob(roi, model_r, patch_size, stride, meta_r["num_classes"], meta_r["need_coords"], device)
                prob_r = np.zeros((2, *img_czyx.shape[1:]), dtype=np.float32)
                prob_r[:, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = prob_r_roi
                pred = keep_largest_cc((np.argmax(prob_r, axis=0) > 0).astype(np.uint8))
        else:
            prob = sliding_window_prob(img_czyx, model_s, patch_size, stride, meta_s["num_classes"], meta_s["need_coords"], device)
            pred = keep_largest_cc((np.argmax(prob, axis=0) > 0).astype(np.uint8))

        d = dice01(pred, gt)
        dices.append(d)
        print(f"{cid}: Dice={d:.4f}")

    if dices:
        print(f"Mean Dice = {float(np.mean(dices)):.4f} ± {float(np.std(dices)):.4f}")


if __name__ == "__main__":
    main()
