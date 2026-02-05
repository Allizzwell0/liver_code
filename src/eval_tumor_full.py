#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-volume tumor evaluation (Dice) on nnUNetv2-preprocessed LiTS (.b2nd).

Workflow (recommended):
1) Run liver cascade inference first to save liver mask/prob:
   python infer.py --preproc_dir $PREPROC_DIR --out_dir pred_liver --ckpt_coarse ... --ckpt_refine ... --save_prob
2) Run this script for tumor full-volume Dice:
   python eval_tumor_full.py --preproc_dir $PREPROC_DIR --liver_dir pred_liver --ckpt_tumor ... --split val ...

If you don't provide --liver_dir, you can provide --ckpt_coarse/--ckpt_refine and it will compute liver on-the-fly.
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


# ----------------- helpers ----------------- #

def get_logits(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


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


def postprocess_liver(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    m = keep_largest_cc(m)
    m = fill_holes(m)
    m = keep_largest_cc(m)
    return m


def remove_small_cc(mask: np.ndarray, min_size: int = 20) -> np.ndarray:
    ndi_label, _ = _try_import_scipy()
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
    """
    Tumor postprocess inside liver ROI:
      1) thr -> binary
      2) mask *= liver_roi
      3) remove small CC
      4) (optional) remove low-mean-prob CC
      5) (optional) remove thin & elongated CC (duct-like)
    """
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
        keep[0] = 0
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


def dice_score(pred01: np.ndarray, gt01: np.ndarray, smooth: float = 1e-5) -> float:
    pred = (pred01 > 0).astype(np.float32)
    gt = (gt01 > 0).astype(np.float32)
    inter = float((pred * gt).sum())
    union = float(pred.sum() + gt.sum())
    return float((2.0 * inter + smooth) / (union + smooth))


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
    # deep_supervision is training-only; inference uses only the main logits
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
    meta.update({
        "in_channels": in_channels,
        "num_classes": num_classes,
        "base_filters": base_filters,
        "use_coords": use_coords,
        "use_sdf_head": use_sdf_head,
        "use_attn_gate": use_attn_gate,
        "use_se": use_se,
        "use_cbam": use_cbam,
    })
    return model, meta


# ----------------- main ----------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--ckpt_tumor", type=str, required=True)

    # liver source (recommended: precomputed from infer.py)
    p.add_argument("--liver_dir", type=str, default=None,
                   help="directory containing <case>_pred_liver.npy and optionally <case>_prob_liver.npy")
    p.add_argument("--ckpt_coarse", type=str, default=None, help="optional: compute liver on-the-fly")
    p.add_argument("--ckpt_refine", type=str, default=None, help="optional: compute liver on-the-fly")

    p.add_argument("--out_dir", type=str, default="eval_tumor_full")
    p.add_argument("--split", type=str, choices=["all", "train", "val"], default="val")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_cases", type=int, default=-1)

    p.add_argument("--patch_size", type=int, nargs=3, default=[96, 160, 160])
    p.add_argument("--stride", type=int, nargs=3, default=[48, 80, 80])

    p.add_argument("--bbox_margin", type=int, default=24)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--min_cc", type=int, default=20)
    p.add_argument("--cc_min_mean_prob", type=float, default=0.0,
                   help="remove CC whose mean(prob) < this (0 disables)")
    p.add_argument("--filter_tubular", type=int, default=1, choices=[0, 1],
                   help="heuristic: remove very thin & elongated CC (likely ducts)")
    p.add_argument("--tubular_aspect", type=float, default=10.0)
    p.add_argument("--tubular_thickness", type=int, default=3)

    p.add_argument("--use_gt_liver_bbox", action="store_true", help="use GT liver bbox (upper bound)")
    p.add_argument("--save_npy", action="store_true")
    return p.parse_args()


def _load_liver_from_dir(liver_dir: Path, case_id: str):
    m = None
    p = liver_dir / f"{case_id}_pred_liver.npy"
    if p.is_file():
        m = np.load(p).astype(np.uint8)
    else:
        p2 = liver_dir / f"{case_id}_liver_pred.npy"
        if p2.is_file():
            m = np.load(p2).astype(np.uint8)

    prob = None
    pprob = liver_dir / f"{case_id}_prob_liver.npy"
    if pprob.is_file():
        prob = np.load(pprob).astype(np.float32)
    return m, prob


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tumor_model, tumor_meta = load_model(args.ckpt_tumor, device)
    tumor_in_ch = int(tumor_meta.get("in_channels", 1))
    tumor_prior_type = str(tumor_meta.get("tumor_prior_type", "mask"))
    need_prior = tumor_in_ch >= 2
    print(f"=> Tumor ckpt={args.ckpt_tumor}")
    print(f"   tumor_in_ch={tumor_in_ch}, need_prior={need_prior}, prior_type(from_ckpt)={tumor_prior_type}")

    liver_dir = Path(args.liver_dir) if args.liver_dir else None
    liver_models = None
    if liver_dir is None:
        if not (args.ckpt_coarse and args.ckpt_refine):
            raise ValueError("Provide --liver_dir OR (--ckpt_coarse and --ckpt_refine) to get liver.")
        from infer import load_model as load_liver_model, sliding_window_prob as liver_sw_prob, postprocess_liver as liver_post, bbox_from_mask as liver_bbox
        liver_c, meta_c = load_liver_model(args.ckpt_coarse, device)
        liver_r, meta_r = load_liver_model(args.ckpt_refine, device)
        liver_models = (liver_c, meta_c, liver_r, meta_r, liver_sw_prob, liver_post, liver_bbox)

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)
    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]
    print(f"Found {len(case_ids)} case(s) for split='{args.split}'.")

    dices: List[float] = []

    for cid in tqdm(case_ids, desc="Eval-Tumor", unit="case"):
        img_np, seg_np = load_case_b2nd(preproc_dir, cid)
        liver_gt = (seg_np > 0).astype(np.uint8)
        tumor_gt = (seg_np == 2).astype(np.uint8)

        # ---- get liver pred + liver prob (optional) ----
        liver_pred = liver_prob = None
        if liver_dir is not None:
            liver_pred, liver_prob = _load_liver_from_dir(liver_dir, cid)

        if liver_pred is None:
            # on-the-fly liver cascade
            liver_c, meta_c, liver_r, meta_r, liver_sw_prob, liver_post, liver_bbox = liver_models
            vol = torch.from_numpy(img_np[None, ...]).to(device)
            prob_c = liver_sw_prob(vol, liver_c, tuple([128,128,128]), tuple([64,64,64]), num_classes=int(meta_c["num_classes"]))
            prob_liver_c = prob_c[1].detach().cpu().numpy()
            liver_c_mask = liver_post((prob_liver_c > 0.5).astype(np.uint8))
            b = liver_bbox(liver_c_mask, margin=int(args.bbox_margin), shape_zyx=liver_c_mask.shape)
            if b is None:
                liver_pred = np.zeros_like(liver_gt, dtype=np.uint8)
                liver_prob = np.zeros_like(liver_gt, dtype=np.float32)
            else:
                z0,z1,y0,y1,x0,x1 = b
                roi = img_np[:, z0:z1+1, y0:y1+1, x0:x1+1]
                prob_r = liver_sw_prob(torch.from_numpy(roi[None, ...]).to(device), liver_r, tuple([128,128,128]), tuple([64,64,64]), num_classes=int(meta_r["num_classes"]))
                prob_liver_r = prob_r[1].detach().cpu().numpy().astype(np.float32)
                liver_prob = np.zeros_like(liver_gt, dtype=np.float32)
                liver_prob[z0:z1+1, y0:y1+1, x0:x1+1] = prob_liver_r
                liver_pred = liver_post((liver_prob > 0.5).astype(np.uint8))

        liver_pred = (liver_pred > 0).astype(np.uint8)
        if liver_prob is not None:
            liver_prob = liver_prob.astype(np.float32)

        # ---- bbox for tumor ----
        if args.use_gt_liver_bbox:
            b = bbox_from_mask(liver_gt, margin=int(args.bbox_margin))
        else:
            b = bbox_from_mask(liver_pred, margin=int(args.bbox_margin))
            if b is None:
                b = bbox_from_mask(liver_gt, margin=int(args.bbox_margin))  # fallback

        if b is None:
            tumor_pred = np.zeros_like(tumor_gt, dtype=np.uint8)
        else:
            z0, z1, y0, y1, x0, x1 = b
            img_roi = img_np[:, z0:z1+1, y0:y1+1, x0:x1+1]  # (C,Z,Y,X)

            # build tumor input channels
            if tumor_in_ch == 1:
                in_roi = img_roi
            else:
                # prior channel
                if (tumor_prior_type == "prob") and (liver_prob is not None):
                    prior_full = liver_prob
                    prior_roi = prior_full[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
                else:
                    prior_roi = liver_pred[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
                in_roi = np.concatenate([img_roi, prior_roi], axis=0).astype(np.float32)

            vol_roi = torch.from_numpy(in_roi[None, ...]).to(device)
            prob_t = sliding_window_prob(vol_roi, tumor_model, patch_size, stride, num_classes=2)
            prob_tumor = prob_t[1].detach().cpu().numpy().astype(np.float32)
            tumor_roi = (prob_tumor > float(args.thr)).astype(np.uint8)

            # restrict to liver, remove small cc
            liver_roi = liver_pred[z0:z1+1, y0:y1+1, x0:x1+1].astype(np.uint8)
            tumor_roi = (tumor_roi * liver_roi).astype(np.uint8)
            tumor_roi = remove_small_cc(tumor_roi, min_size=int(args.min_cc))

            tumor_pred = np.zeros_like(tumor_gt, dtype=np.uint8)
            tumor_pred[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_roi

        d = dice_score(tumor_pred, tumor_gt)
        dices.append(d)
        print(f"  {cid}: Tumor Dice={d:.4f}")

        if args.save_npy:
            np.save(out_dir / f"{cid}_tumor_pred.npy", tumor_pred.astype(np.uint8))
            np.save(out_dir / f"{cid}_tumor_gt.npy", tumor_gt.astype(np.uint8))

    dices_np = np.array(dices, dtype=np.float32)
    print("=" * 60)
    print(f"[Tumor Dice] split={args.split} cases={len(dices)}")
    print(f"  Mean={float(dices_np.mean()):.4f}  Std={float(dices_np.std()):.4f}")
    print("=" * 60)

    with (out_dir / "tumor_eval_results.txt").open("w") as f:
        for cid, d in zip(case_ids, dices):
            f.write(f"{cid}\t{d:.6f}\n")
        f.write(f"\nMean\t{float(dices_np.mean()):.6f}\nStd\t{float(dices_np.std()):.6f}\n")


if __name__ == "__main__":
    main()
