#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference on nnUNetv2-preprocessed volumes (.b2nd) for LiTS / your own CT.

Supports robust liver cascade (recommended for LiTS->unlabeled CT transfer):
- Coarse liver model (full volume, no priors/coords) -> coarse liver mask -> bbox
- Refine liver model (ROI, ROI-global coords + priors optional) -> refined liver mask

Optional tumor model:
- Tumor model runs within predicted liver ROI and outputs tumor mask.

Outputs (.npy):
  <out_dir>/<case_id>_pred_liver.npy   (0/1) liver
  <out_dir>/<case_id>_pred_tumor.npy   (0/1) tumor (if --ckpt_tumor)
  <out_dir>/<case_id>_pred_3class.npy  (0/1/2) final (if --ckpt_tumor)

If GT seg exists (<case_id>_seg.b2nd), prints Dice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet3D import UNet3D


# -------------------- I/O -------------------- #

def load_b2nd(preproc_dir: Path, case_id: str):
    dparams = {"nthreads": 1}
    data_file = preproc_dir / f"{case_id}.b2nd"
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    img = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    return img


def load_seg_if_exists(preproc_dir: Path, case_id: str) -> Optional[np.ndarray]:
    dparams = {"nthreads": 1}
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    if not seg_file.is_file():
        return None
    seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)
    seg = seg_b[:].astype(np.int16)
    if seg.ndim == 4:
        seg = seg[0]
    return seg


def list_case_ids(preproc_dir: Path) -> List[str]:
    pkls = sorted(preproc_dir.glob("*.pkl"))
    if pkls:
        return [p.stem for p in pkls]
    b2 = sorted(preproc_dir.glob("*.b2nd"))
    ids = []
    for p in b2:
        if p.name.endswith("_seg.b2nd"):
            continue
        ids.append(p.stem)
    return ids


# -------------------- coords / bbox / postprocess -------------------- #

def make_coords_zyx(shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    Z, Y, X = shape_zyx
    zz = np.linspace(-1.0, 1.0, Z, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, Y, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, X, dtype=np.float32)[None, None, :]
    zc = np.broadcast_to(zz, (Z, Y, X))
    yc = np.broadcast_to(yy, (Z, Y, X))
    xc = np.broadcast_to(xx, (Z, Y, X))
    return np.stack([zc, yc, xc], axis=0)  # (3,Z,Y,X)


def add_coords_if_needed(img_czyx: np.ndarray, need_coords: bool) -> np.ndarray:
    if not need_coords:
        return img_czyx
    coords = make_coords_zyx(img_czyx.shape[1:])  # (3,Z,Y,X)
    return np.concatenate([img_czyx, coords], axis=0).astype(np.float32)


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


# -------------------- sliding window -------------------- #

def _hann_3d(patch_size: Tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """(1,1,pz,py,px) hann weight for smooth stitching."""
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
    volume: torch.Tensor,           # (1,C,Z,Y,X)
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

    w_patch = _hann_3d(patch_size, device=device)  # (1,1,pz,py,px)

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
                    out = model(patch)
                    logits = get_logits(out)
                    probs = F.softmax(logits, dim=1)  # (1,Cc,pz,py,px)
                    prob_map[:, z0:z1, y0:y1, x0:x1] += (probs[0] * w_patch[0])
                    weight_map[:, z0:z1, y0:y1, x0:x1] += w_patch[0, 0]

    prob_map = prob_map / torch.clamp_min(weight_map, 1e-6)
    return prob_map


def maybe_tta_prob(
    volume: torch.Tensor, model: torch.nn.Module, patch_size, stride, num_classes: int, tta: bool
) -> torch.Tensor:
    if not tta:
        return sliding_window_prob(volume, model, patch_size, stride, num_classes=num_classes)

    # 8 flips over (Z,Y,X)
    flips = [
        (),
        (2,),
        (3,),
        (4,),
        (2, 3),
        (2, 4),
        (3, 4),
        (2, 3, 4),
    ]
    probs_sum = None
    for dims in flips:
        v = volume
        if dims:
            v = torch.flip(v, dims=dims)
        p = sliding_window_prob(v, model, patch_size, stride, num_classes=num_classes)
        if dims:
            p = torch.flip(p, dims=dims)
        probs_sum = p if probs_sum is None else (probs_sum + p)
    return probs_sum / float(len(flips))


def dice_score(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-5) -> float:
    p = (pred01 > 0).astype(np.float32)
    g = (gt01 > 0).astype(np.float32)
    inter = (p * g).sum()
    union = p.sum() + g.sum()
    return float((2.0 * inter + smooth) / (union + smooth))


# -------------------- model loading -------------------- #

def load_model(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    use_se = bool(ckpt.get("use_se", False))
    need_coords = bool(ckpt.get("add_coords", False))

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
    meta = dict(ckpt)
    meta["need_coords"] = need_coords
    meta["patch_size"] = tuple(meta.get("patch_size", (128, 128, 128)))
    meta["num_classes"] = num_classes
    return model, meta


# -------------------- main -------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="infer_out")
    p.add_argument("--case_id", type=str, default=None, help="if None: run all cases in preproc_dir")

    # liver ckpts
    p.add_argument("--ckpt_liver", type=str, default=None, help="single-stage liver ckpt")
    p.add_argument("--ckpt_coarse", type=str, default=None, help="coarse liver ckpt (full volume)")
    p.add_argument("--ckpt_refine", type=str, default=None, help="refine liver ckpt (ROI)")

    # tumor ckpt (optional)
    p.add_argument("--ckpt_tumor", type=str, default=None)

    # override inference params (otherwise read from ckpt)
    p.add_argument("--patch_size", type=int, nargs=3, default=None)
    p.add_argument("--stride", type=int, nargs=3, default=None)

    p.add_argument("--bbox_margin", type=int, default=24)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--save_prob", action="store_true")
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
    if not use_cascade and args.ckpt_liver is None:
        raise ValueError("Provide either --ckpt_liver (single-stage) or (--ckpt_coarse and --ckpt_refine) for cascade.")

    if use_cascade:
        model_coarse, meta_c = load_model(Path(args.ckpt_coarse), device)
        model_refine, meta_r = load_model(Path(args.ckpt_refine), device)
    else:
        model_liver, meta_l = load_model(Path(args.ckpt_liver), device)

    model_tumor = None
    meta_t = None
    if args.ckpt_tumor:
        model_tumor, meta_t = load_model(Path(args.ckpt_tumor), device)

    case_ids = [args.case_id] if args.case_id else list_case_ids(preproc_dir)
    print(f"Found {len(case_ids)} case(s).")

    for case_id in tqdm(case_ids, desc="Infer", unit="case"):
        img_np = load_b2nd(preproc_dir, case_id)  # (C,Z,Y,X)
        seg_np = load_seg_if_exists(preproc_dir, case_id)  # (Z,Y,X) or None

        # ---------- Liver ----------
        if use_cascade:
            img_c = add_coords_if_needed(img_np, meta_c["need_coords"])
            vol = torch.from_numpy(img_c[None, ...]).to(device)
            patch_size = tuple(args.patch_size) if args.patch_size else tuple(meta_c["patch_size"])
            stride = tuple(args.stride) if args.stride else tuple(max(1, p // 2) for p in patch_size)
            prob_c = maybe_tta_prob(vol, model_coarse, patch_size, stride, meta_c["num_classes"], args.tta)
            pred_c = prob_c.argmax(dim=0).cpu().numpy().astype(np.uint8)
            liver_c = postprocess_liver(pred_c)

            bbox = get_bbox_from_mask(liver_c, margin=args.bbox_margin)
            if bbox is None:
                liver_pred = np.zeros_like(liver_c, dtype=np.uint8)
            else:
                img_roi = crop_with_bbox(img_np, bbox)
                img_r = add_coords_if_needed(img_roi, meta_r["need_coords"])
                vol_r = torch.from_numpy(img_r[None, ...]).to(device)

                patch_size_r = tuple(args.patch_size) if args.patch_size else tuple(meta_r["patch_size"])
                stride_r = tuple(args.stride) if args.stride else tuple(max(1, p // 2) for p in patch_size_r)
                prob_r = maybe_tta_prob(vol_r, model_refine, patch_size_r, stride_r, meta_r["num_classes"], args.tta)
                pred_r = prob_r.argmax(dim=0).cpu().numpy().astype(np.uint8)
                liver_roi = postprocess_liver(pred_r)

                liver_pred = np.zeros(img_np.shape[1:], dtype=np.uint8)  # (Z,Y,X)
                paste_roi(liver_pred, liver_roi, bbox)
                liver_pred = postprocess_liver(liver_pred)

            if args.save_prob:
                np.save(out_dir / f"{case_id}_prob_liver_coarse.npy", prob_c.cpu().numpy())
        else:
            img_l = add_coords_if_needed(img_np, meta_l["need_coords"])
            vol = torch.from_numpy(img_l[None, ...]).to(device)
            patch_size = tuple(args.patch_size) if args.patch_size else tuple(meta_l["patch_size"])
            stride = tuple(args.stride) if args.stride else tuple(max(1, p // 2) for p in patch_size)
            prob = maybe_tta_prob(vol, model_liver, patch_size, stride, meta_l["num_classes"], args.tta)
            pred = prob.argmax(dim=0).cpu().numpy().astype(np.uint8)
            liver_pred = postprocess_liver(pred)

        np.save(out_dir / f"{case_id}_pred_liver.npy", liver_pred)

        if seg_np is not None:
            liver_gt = (seg_np >= 1).astype(np.uint8)
            d = dice_score(liver_pred, liver_gt)
            print(f"  {case_id} Liver Dice={d:.4f}")

        # ---------- Tumor (optional) ----------
        if model_tumor is not None:
            bbox = get_bbox_from_mask(liver_pred, margin=10)
            if bbox is None:
                tumor_pred = np.zeros_like(liver_pred, dtype=np.uint8)
            else:
                img_roi = crop_with_bbox(img_np, bbox)
                img_t = add_coords_if_needed(img_roi, meta_t["need_coords"])
                vol_t = torch.from_numpy(img_t[None, ...]).to(device)

                patch_size_t = tuple(args.patch_size) if args.patch_size else tuple(meta_t["patch_size"])
                stride_t = tuple(args.stride) if args.stride else tuple(max(1, p // 2) for p in patch_size_t)
                prob_t = maybe_tta_prob(vol_t, model_tumor, patch_size_t, stride_t, meta_t["num_classes"], args.tta)
                pred_t = prob_t.argmax(dim=0).cpu().numpy().astype(np.uint8)

                tumor_roi = (pred_t > 0).astype(np.uint8)
                tumor_pred = np.zeros_like(liver_pred, dtype=np.uint8)
                paste_roi(tumor_pred, tumor_roi, bbox)
                tumor_pred = (tumor_pred & (liver_pred > 0)).astype(np.uint8)

            np.save(out_dir / f"{case_id}_pred_tumor.npy", tumor_pred)

            pred_3c = np.zeros_like(liver_pred, dtype=np.uint8)
            pred_3c[liver_pred > 0] = 1
            pred_3c[tumor_pred > 0] = 2
            np.save(out_dir / f"{case_id}_pred_3class.npy", pred_3c)

            if seg_np is not None:
                tumor_gt = (seg_np == 2).astype(np.uint8)
                d_t = dice_score(tumor_pred, tumor_gt)
                print(f"  {case_id} Tumor Dice={d_t:.4f}")


if __name__ == "__main__":
    main()
