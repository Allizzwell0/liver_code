#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-volume tumor inference on nnUNetv2-preprocessed LiTS (.b2nd) or your own preprocessed data.

Inputs:
  - preproc_dir: contains <case>.b2nd (and optionally <case>_seg.b2nd if labeled)
  - liver_dir: outputs from infer.py --save_prob (at least <case>_pred_liver.npy; optional <case>_prob_liver.npy)

Outputs (out_dir):
  - <case>_pred_tumor.npy  (Z,Y,X) uint8
  - <case>_prob_tumor.npy  (Z,Y,X) float32 if --save_prob
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import blosc2
from models.unet3D import UNet3D


# -------------------- io -------------------- #

def list_case_ids(preproc_dir: Path) -> List[str]:
    ids = []
    for p in sorted(preproc_dir.glob("*.b2nd")):
        name = p.stem
        if name.endswith("_seg"):
            continue
        ids.append(name)
    return ids


def load_case_img(preproc_dir: Path, case_id: str) -> np.ndarray:
    blosc2.set_nthreads(1)
    data_file = preproc_dir / f"{case_id}.b2nd"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams={"nthreads": 1})
    img = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    return img


def load_liver_mask(liver_dir: Path, case_id: str) -> np.ndarray:
    p = liver_dir / f"{case_id}_pred_liver.npy"
    if not p.is_file():
        # fallback name
        p2 = liver_dir / f"{case_id}_liver_pred.npy"
        if p2.is_file():
            p = p2
        else:
            raise FileNotFoundError(f"missing liver pred for {case_id}: {p}")
    m = np.load(p)
    return (m > 0).astype(np.uint8)


def load_liver_prob(liver_dir: Path, case_id: str) -> np.ndarray | None:
    for name in [f"{case_id}_prob_liver.npy", f"{case_id}_liver_prob.npy", f"{case_id}_pred_liver_prob.npy"]:
        p = liver_dir / name
        if p.is_file():
            return np.load(p).astype(np.float32)
    return None


# -------------------- misc -------------------- #

def bbox_from_mask(mask_zyx: np.ndarray, margin: int = 0) -> Tuple[slice, slice, slice]:
    zz, yy, xx = np.where(mask_zyx > 0)
    if len(zz) == 0:
        return slice(0, mask_zyx.shape[0]), slice(0, mask_zyx.shape[1]), slice(0, mask_zyx.shape[2])
    z0, z1 = zz.min(), zz.max() + 1
    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1

    z0 = max(0, z0 - margin)
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    z1 = min(mask_zyx.shape[0], z1 + margin)
    y1 = min(mask_zyx.shape[1], y1 + margin)
    x1 = min(mask_zyx.shape[2], x1 + margin)
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)


def remove_small_cc(mask_zyx: np.ndarray, min_size: int = 20) -> np.ndarray:
    try:
        from scipy.ndimage import label as ndi_label
    except Exception:
        return mask_zyx
    lab, n = ndi_label(mask_zyx.astype(np.uint8))
    if n == 0:
        return mask_zyx
    out = np.zeros_like(mask_zyx, dtype=np.uint8)
    for k in range(1, n + 1):
        comp = (lab == k)
        if int(comp.sum()) >= int(min_size):
            out[comp] = 1
    return out


def postprocess_tumor_mask(
    prob_tumor_zyx: np.ndarray,
    liver_roi_zyx: np.ndarray,
    thr: float = 0.5,
    min_cc: int = 20,
    cc_min_mean_prob: float = 0.0,
    filter_tubular: bool = True,
    tubular_aspect: float = 10.0,
    tubular_thickness: int = 3,
) -> np.ndarray:
    mask = (prob_tumor_zyx > float(thr)).astype(np.uint8)
    mask = (mask * (liver_roi_zyx > 0).astype(np.uint8)).astype(np.uint8)
    mask = remove_small_cc(mask, min_size=int(min_cc))

    if (cc_min_mean_prob > 0) or filter_tubular:
        try:
            from scipy.ndimage import label as ndi_label
        except Exception:
            return mask

        lab, n = ndi_label(mask.astype(np.uint8))
        if n <= 0:
            return mask

        keep = np.zeros(n + 1, dtype=np.uint8)
        for k in range(1, n + 1):
            comp = (lab == k)
            if comp.sum() == 0:
                continue

            if cc_min_mean_prob > 0:
                mp = float(prob_tumor_zyx[comp].mean())
                if mp < float(cc_min_mean_prob):
                    continue

            if filter_tubular:
                zs, ys, xs = np.where(comp)
                dz = int(zs.max() - zs.min() + 1)
                dy = int(ys.max() - ys.min() + 1)
                dx = int(xs.max() - xs.min() + 1)
                th = min(dz, dy, dx)
                asp = max(dz, dy, dx) / max(1, th)
                if (th <= int(tubular_thickness)) and (asp >= float(tubular_aspect)):
                    continue

            keep[k] = 1

        return (keep[lab] > 0).astype(np.uint8)

    return mask


# -------------------- model / sliding window -------------------- #

@torch.no_grad()
def sliding_window_prob(vol_czyx: np.ndarray, model: torch.nn.Module,
                        patch_size: Tuple[int, int, int],
                        stride: Tuple[int, int, int],
                        num_classes: int = 2,
                        device: torch.device | None = None) -> np.ndarray:
    """
    vol_czyx: (C,Z,Y,X) numpy float32
    Returns: prob_map (num_classes, Z, Y, X)
    """
    if device is None:
        device = next(model.parameters()).device

    C, Z, Y, X = vol_czyx.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    prob_map = np.zeros((num_classes, Z, Y, X), dtype=np.float32)
    count_map = np.zeros((Z, Y, X), dtype=np.float32)

    # gaussian-ish weights reduce seams (cheap separable)
    wz = np.hanning(pz) if pz > 1 else np.ones((1,), dtype=np.float32)
    wy = np.hanning(py) if py > 1 else np.ones((1,), dtype=np.float32)
    wx = np.hanning(px) if px > 1 else np.ones((1,), dtype=np.float32)
    w_patch = (wz[:, None, None] * wy[None, :, None] * wx[None, None, :]).astype(np.float32)
    w_patch = np.maximum(w_patch, 1e-3)  # avoid zero

    z_starts = list(range(0, max(Z - pz, 0) + 1, sz))
    y_starts = list(range(0, max(Y - py, 0) + 1, sy))
    x_starts = list(range(0, max(X - px, 0) + 1, sx))
    if len(z_starts) == 0:
        z_starts = [0]
    if z_starts[-1] != Z - pz:
        z_starts.append(max(Z - pz, 0))
    if len(y_starts) == 0:
        y_starts = [0]
    if y_starts[-1] != Y - py:
        y_starts.append(max(Y - py, 0))
    if len(x_starts) == 0:
        x_starts = [0]
    if x_starts[-1] != X - px:
        x_starts.append(max(X - px, 0))

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                z1, y1, x1 = z0 + pz, y0 + py, x0 + px
                patch = vol_czyx[:, z0:z1, y0:y1, x0:x1]
                # pad if needed
                pad_z = max(0, pz - patch.shape[1])
                pad_y = max(0, py - patch.shape[2])
                pad_x = max(0, px - patch.shape[3])
                if pad_z or pad_y or pad_x:
                    patch = np.pad(patch, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="edge")

                inp = torch.from_numpy(patch[None]).to(device, non_blocking=True)  # (1,C,pz,py,px)
                out = model(inp)
                logits = out["logits"] if isinstance(out, dict) else out
                probs = torch.softmax(logits, dim=1)[0].float().cpu().numpy()  # (K,pz,py,px)

                # crop back to original (if padded)
                probs = probs[:, :min(pz, Z - z0), :min(py, Y - y0), :min(px, X - x0)]
                w = w_patch[:probs.shape[1], :probs.shape[2], :probs.shape[3]]

                prob_map[:, z0:z0 + probs.shape[1], y0:y0 + probs.shape[2], x0:x0 + probs.shape[3]] += probs * w[None]
                count_map[z0:z0 + probs.shape[1], y0:y0 + probs.shape[2], x0:x0 + probs.shape[3]] += w

    prob_map = prob_map / np.clip(count_map[None], 1e-6, None)
    return prob_map


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    base_filters = int(ckpt.get("base_filters", 32))
    use_coords = bool(ckpt.get("use_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    use_attn_gate = bool(ckpt.get("use_attn_gate", False))
    use_se = bool(ckpt.get("use_se", False))
    use_cbam = bool(ckpt.get("use_cbam", False))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=base_filters,
        dropout_p=float(ckpt.get("dropout_p", 0.0)),
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
        use_attn_gate=use_attn_gate,
        use_se=use_se,
        use_cbam=use_cbam,
        deep_supervision=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = dict(ckpt)
    return model, meta


# -------------------- main -------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--ckpt_tumor", type=str, required=True)
    p.add_argument("--liver_dir", type=str, required=True)
    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 160, 160])
    p.add_argument("--stride", type=int, nargs=3, default=[48, 80, 80])
    p.add_argument("--bbox_margin", type=int, default=24)

    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_cc", type=int, default=20)
    p.add_argument("--cc_min_mean_prob", type=float, default=0.0)
    p.add_argument("--filter_tubular", type=int, default=1, choices=[0, 1])
    p.add_argument("--tubular_aspect", type=float, default=10.0)
    p.add_argument("--tubular_thickness", type=int, default=3)

    p.add_argument("--save_prob", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    liver_dir = Path(args.liver_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, meta = load_model(args.ckpt_tumor, device)
    tumor_add_liver_prior = bool(meta.get("tumor_add_liver_prior", False))
    tumor_prior_type = str(meta.get("tumor_prior_type", "mask"))

    case_ids = list_case_ids(preproc_dir)
    print(f"Found {len(case_ids)} case(s).")
    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    for cid in tqdm(case_ids, desc="Infer-Tumor"):
        img = load_case_img(preproc_dir, cid)  # (C,Z,Y,X)
        liver_mask = load_liver_mask(liver_dir, cid)  # (Z,Y,X)
        zsl, ysl, xsl = bbox_from_mask(liver_mask, margin=int(args.bbox_margin))

        img_roi = img[:, zsl, ysl, xsl]
        liver_roi = liver_mask[zsl, ysl, xsl].astype(np.uint8)

        # optional liver prior channel (must match training)
        if tumor_add_liver_prior:
            if tumor_prior_type == "prob":
                prob = load_liver_prob(liver_dir, cid)
                if prob is None:
                    prior = liver_mask.astype(np.float32)
                else:
                    prior = prob
            else:
                prior = liver_mask.astype(np.float32)
            prior_roi = prior[zsl, ysl, xsl].astype(np.float32)
            img_roi = np.concatenate([img_roi, prior_roi[None]], axis=0)

        prob_map = sliding_window_prob(img_roi, model, patch_size=patch_size, stride=stride,
                                       num_classes=int(meta.get("num_classes", 2)), device=device)
        prob_tumor_roi = prob_map[1]  # (Z,Y,X)

        # paste back to full size
        Z, Y, X = liver_mask.shape
        prob_tumor = np.zeros((Z, Y, X), dtype=np.float32)
        prob_tumor[zsl, ysl, xsl] = prob_tumor_roi

        pred_mask = np.zeros((Z, Y, X), dtype=np.uint8)
        pred_mask[zsl, ysl, xsl] = postprocess_tumor_mask(
            prob_tumor_roi,
            liver_roi,
            thr=float(args.thr),
            min_cc=int(args.min_cc),
            cc_min_mean_prob=float(args.cc_min_mean_prob),
            filter_tubular=bool(args.filter_tubular),
            tubular_aspect=float(args.tubular_aspect),
            tubular_thickness=int(args.tubular_thickness),
        )

        np.save(out_dir / f"{cid}_pred_tumor.npy", pred_mask)
        if args.save_prob:
            np.save(out_dir / f"{cid}_prob_tumor.npy", prob_tumor)

    print("Done. Output:", out_dir)


if __name__ == "__main__":
    main()
