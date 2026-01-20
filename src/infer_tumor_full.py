#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tumor inference on full 3D volume (nnUNetv2-preprocessed .b2nd).

Inputs:
- preproc_dir: contains <case>.b2nd and *.pkl (for listing) OR use --case_id for single case.
- liver_dir: output of liver infer.py containing <case>_pred_liver.npy and optionally <case>_prob_liver.npy

Outputs:
- <out_dir>/<case>_pred_tumor.npy (uint8 0/1)
- <out_dir>/<case>_prob_tumor.npy (float32 0..1, optional with --save_prob)
- <out_dir>/<case>_pred_seg.npy   (uint8 0/1/2)  (optional with --save_seg)
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--ckpt_tumor", type=str, required=True)
    p.add_argument("--liver_dir", type=str, required=True)

    p.add_argument("--case_id", type=str, default=None)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 160, 160])
    p.add_argument("--stride", type=int, nargs=3, default=[48, 80, 80])
    p.add_argument("--bbox_margin", type=int, default=24)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_cc", type=int, default=20)

    p.add_argument("--save_prob", action="store_true")
    p.add_argument("--save_seg", action="store_true", help="save combined segmentation: 0 bg, 1 liver, 2 tumor")
    return p.parse_args()


def get_logits(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


def _try_import_scipy():
    try:
        from scipy.ndimage import label as ndi_label
        return ndi_label
    except Exception:
        return None


def remove_small_cc(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    ndi_label = _try_import_scipy()
    if ndi_label is None or min_size <= 0:
        return mask.astype(np.uint8)
    lab, n = ndi_label(mask.astype(np.uint8))
    if n == 0:
        return mask.astype(np.uint8)
    sizes = np.bincount(lab.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[1:] = sizes[1:] >= int(min_size)
    out = keep[lab].astype(np.uint8)
    return out


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


def pad_to_min(volume: torch.Tensor, patch: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
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
                    probs = torch.softmax(logits, dim=1)[0]
                    prob[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs
                    w[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0

    prob = prob / torch.clamp_min(w, 1.0)
    oZ, oY, oX = orig
    return prob[:, :oZ, :oY, :oX]


def load_case_b2nd(preproc_dir: Path, case_id: str) -> np.ndarray:
    dparams = {"nthreads": 1}
    data_file = preproc_dir / f"{case_id}.b2nd"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    return data_b[:].astype(np.float32)


def list_case_ids(preproc_dir: Path) -> List[str]:
    return [p.stem for p in sorted(preproc_dir.glob("*.pkl"))]


def _load_liver(liver_dir: Path, case_id: str):
    pred = None
    p = liver_dir / f"{case_id}_pred_liver.npy"
    if p.is_file():
        pred = np.load(p).astype(np.uint8)
    else:
        p2 = liver_dir / f"{case_id}_liver_pred.npy"
        if p2.is_file():
            pred = np.load(p2).astype(np.uint8)

    prob = None
    pp = liver_dir / f"{case_id}_prob_liver.npy"
    if pp.is_file():
        prob = np.load(pp).astype(np.float32)

    if pred is None:
        raise FileNotFoundError(f"Missing liver pred for case={case_id} in {liver_dir}")
    return pred, prob


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_coords = bool(ckpt.get("use_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=float(ckpt.get("dropout_p", 0.0)),
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = dict(ckpt)
    meta["in_channels"] = in_channels
    meta["num_classes"] = num_classes
    return model, meta


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    liver_dir = Path(args.liver_dir)

    model, meta = load_model(args.ckpt_tumor, device)
    in_ch = int(meta.get("in_channels", 1))
    prior_type = str(meta.get("tumor_prior_type", "mask"))
    need_prior = in_ch >= 2
    print(f"=> Tumor model: {args.ckpt_tumor}")
    print(f"   in_ch={in_ch}, need_prior={need_prior}, prior_type(from_ckpt)={prior_type}")

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = [args.case_id] if args.case_id else list_case_ids(preproc_dir)
    print(f"Found {len(case_ids)} case(s).")

    for cid in tqdm(case_ids, desc="Infer-Tumor", unit="case"):
        img_np = load_case_b2nd(preproc_dir, cid)  # (C,Z,Y,X)
        liver_pred, liver_prob = _load_liver(liver_dir, cid)
        liver_pred = (liver_pred > 0).astype(np.uint8)

        b = bbox_from_mask(liver_pred, margin=int(args.bbox_margin))
        if b is None:
            tumor_pred = np.zeros_like(liver_pred, dtype=np.uint8)
            prob_full = np.zeros_like(liver_pred, dtype=np.float32)
        else:
            z0,z1,y0,y1,x0,x1 = b
            img_roi = img_np[:, z0:z1+1, y0:y1+1, x0:x1+1]
            if in_ch == 1:
                in_roi = img_roi.astype(np.float32)
            else:
                if (prior_type == "prob") and (liver_prob is not None):
                    prior_roi = liver_prob[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
                else:
                    prior_roi = liver_pred[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
                in_roi = np.concatenate([img_roi, prior_roi], axis=0).astype(np.float32)

            vol_roi = torch.from_numpy(in_roi[None, ...]).to(device)
            prob = sliding_window_prob(vol_roi, model, patch_size, stride, num_classes=2)
            prob_tumor = prob[1].detach().cpu().numpy().astype(np.float32)

            tumor_roi = (prob_tumor > float(args.thr)).astype(np.uint8)
            liver_roi = liver_pred[z0:z1+1, y0:y1+1, x0:x1+1].astype(np.uint8)
            tumor_roi = (tumor_roi * liver_roi).astype(np.uint8)
            tumor_roi = remove_small_cc(tumor_roi, min_size=int(args.min_cc))

            tumor_pred = np.zeros_like(liver_pred, dtype=np.uint8)
            tumor_pred[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_roi

            prob_full = np.zeros_like(liver_pred, dtype=np.float32)
            prob_full[z0:z1+1, y0:y1+1, x0:x1+1] = prob_tumor

        np.save(out_dir / f"{cid}_pred_tumor.npy", tumor_pred.astype(np.uint8))
        if args.save_prob:
            np.save(out_dir / f"{cid}_prob_tumor.npy", prob_full.astype(np.float32))
        if args.save_seg:
            seg = liver_pred.astype(np.uint8)
            seg[tumor_pred > 0] = 2
            np.save(out_dir / f"{cid}_pred_seg.npy", seg.astype(np.uint8))

    print(f"[DONE] saved to: {out_dir}")


if __name__ == "__main__":
    main()
