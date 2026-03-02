#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 liver_coarse + liver_refine 做 cascade 推理，并导出：
  <out_dir>/<case_id>_pred_liver.npy   (uint8 0/1)
  <out_dir>/<case_id>_prob_liver.npy   (float32 0..1)

可用于：tumor 训练阶段的 ROI bbox / liver prior（prob/mask）。
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
    p.add_argument("--ckpt_coarse", type=str, required=True)
    p.add_argument("--ckpt_refine", type=str, default=None,
                   help="可选：liver_refine ckpt；不提供则只输出 coarse 结果")
    p.add_argument("--case_id", type=str, default=None, help="只推理单个 case；默认遍历全部")
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--bbox_margin", type=int, default=24)
    p.add_argument("--thr", type=float, default=0.5)
    # bbox seed for cascade (avoid full-body bbox on OOD cases)
    p.add_argument("--thr_bbox", type=float, default=0.98, help="seed threshold for bbox (use high) (default 0.98)")
    p.add_argument("--bbox_area_frac", type=float, default=0.03, help="keep z-slices with seed area >= max_area*frac (default 0.03)")
    p.add_argument("--bbox_min_area", type=int, default=500, help="min seed area per slice to be considered (default 500)")
    p.add_argument("--save_prob", action="store_true", help="保存 prob_liver.npy（float32）")
    return p.parse_args()


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


def bbox_from_mask(mask: np.ndarray, margin: int, shape_zyx: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int, int, int]]:
    idx = np.where(mask > 0)
    if idx[0].size == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    x0, x1 = int(idx[2].min()), int(idx[2].max())
    z0 -= margin; y0 -= margin; x0 -= margin
    z1 += margin; y1 += margin; x1 += margin
    Z, Y, X = shape_zyx
    z0 = max(0, min(z0, Z - 1)); z1 = max(0, min(z1, Z - 1))
    y0 = max(0, min(y0, Y - 1)); y1 = max(0, min(y1, Y - 1))
    x0 = max(0, min(x0, X - 1)); x1 = max(0, min(x1, X - 1))
    return z0, z1, y0, y1, x0, x1

def bbox_from_prob_seed(prob_zyx: np.ndarray,
                        thr_seed: float,
                        margin: int,
                        area_frac: float,
                        min_area: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    """
    Build a stable bbox from a HIGH-threshold seed.
    1) seed = prob > thr_seed
    2) slice-area gating: keep z slices where area >= max_area*area_frac and area >= min_area
    3) keep largest contiguous z-segment
    4) keep largest CC within that z-segment, then bbox_from_mask(+margin)
    """
    seed = (prob_zyx > float(thr_seed)).astype(np.uint8)
    if int(seed.sum()) == 0:
        return None

    area = seed.sum(axis=(1, 2)).astype(np.int64)  # per-z
    mx = int(area.max())
    thr_area = max(int(mx * float(area_frac)), int(min_area))
    keep_z = (area >= thr_area)

    # if too strict, relax to any non-zero area
    if keep_z.sum() == 0:
        keep_z = (area > 0)
        if keep_z.sum() == 0:
            return None

    # largest contiguous segment in z
    idx = np.where(keep_z)[0]
    segments = []
    st = int(idx[0]); prev = int(idx[0])
    for z in idx[1:]:
        z = int(z)
        if z == prev + 1:
            prev = z
        else:
            segments.append((st, prev))
            st = z; prev = z
    segments.append((st, prev))
    # pick segment with max total area
    best = max(segments, key=lambda ab: int(area[ab[0]:ab[1]+1].sum()))
    z0, z1 = best

    # restrict seed to this z-segment only
    seed2 = np.zeros_like(seed, dtype=np.uint8)
    seed2[z0:z1+1] = seed[z0:z1+1]

    # within segment, keep largest CC to avoid scattered noise
    seed2 = keep_largest_cc(seed2)
    seed2 = fill_holes(seed2)

    return bbox_from_mask(seed2, margin=int(margin), shape_zyx=seed2.shape)


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
    """
    返回 prob_map: (num_classes, Z, Y, X)（已裁回原尺寸）
    """
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
    if z_starts[-1] + pz < Z:
        z_starts.append(Z - pz)
    if y_starts[-1] + py < Y:
        y_starts.append(Y - py)
    if x_starts[-1] + px < X:
        x_starts.append(X - px)

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    patch = volume[:, :, z0:z0+pz, y0:y0+py, x0:x0+px]
                    out = model(patch)
                    logits = get_logits(out)
                    probs = torch.softmax(logits, dim=1)[0]  # (C,Z,Y,X)
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
    img = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    return img


def list_case_ids(preproc_dir: Path) -> List[str]:
    all_pkl = sorted(preproc_dir.glob("*.pkl"))
    return [p.stem for p in all_pkl]


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_coords = bool(ckpt.get("use_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))
    backbone = str(ckpt.get("backbone", "unet")).lower()
    mednext_k = int(ckpt.get("mednext_k", 7))
    mednext_expansion = int(ckpt.get("mednext_expansion", 4))
    mednext_blocks = int(ckpt.get("mednext_blocks", 2))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
        backbone=backbone,
        mednext_k=mednext_k,
        mednext_expansion=mednext_expansion,
        mednext_blocks=mednext_blocks,
    ).to(device)
    # --- robust state_dict loading (handle net./module. prefix mismatch) ---
    sd = ckpt["model_state"]
    # strip DDP prefix
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    model_keys = list(model.state_dict().keys())
    model_has_net = any(k.startswith("net.") for k in model_keys)
    sd_has_net = any(k.startswith("net.") for k in sd.keys())

    if model_has_net and (not sd_has_net):
        # ckpt: enc0.xxx  -> model: net.enc0.xxx
        sd = {("net." + k): v for k, v in sd.items()}
    elif (not model_has_net) and sd_has_net:
        # ckpt: net.enc0.xxx -> model: enc0.xxx
        sd = {k.replace("net.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model.eval()
    meta = dict(in_channels=in_channels, num_classes=num_classes, use_coords=use_coords, use_sdf_head=use_sdf_head,
                backbone=backbone, mednext_k=mednext_k, mednext_expansion=mednext_expansion, mednext_blocks=mednext_blocks)
    return model, meta


def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("=> Cascade mode")
    print(f"   coarse={args.ckpt_coarse}")
    print(f"   refine={args.ckpt_refine}") if args.ckpt_refine else print("   refine=None (coarse-only)")

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_c, meta_c = load_model(args.ckpt_coarse, device)
    model_r = meta_r = None
    if args.ckpt_refine:
        model_r, meta_r = load_model(args.ckpt_refine, device)

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = [args.case_id] if args.case_id else list_case_ids(preproc_dir)
    print(f"Found {len(case_ids)} case(s).")

    for case_id in tqdm(case_ids, desc="Infer", unit="case"):
        img_np = load_case_b2nd(preproc_dir, case_id)  # (C,Z,Y,X)
        vol = torch.from_numpy(img_np[None, ...]).to(device)

        # coarse prob full volume
        prob_c = sliding_window_prob(vol, model_c, patch_size, stride, num_classes=meta_c["num_classes"])
        prob_liver_c = prob_c[1].detach().cpu().numpy()  # (Z,Y,X)

        # --- robust bbox from HIGH-threshold seed + slice-area gating ---
        b = bbox_from_prob_seed(
            prob_liver_c,
            thr_seed=float(args.thr_bbox),
            margin=int(args.bbox_margin),
            area_frac=float(args.bbox_area_frac),
            min_area=int(args.bbox_min_area),
        )
        # fallback to old behavior if seed bbox fails
        if b is None:
            liver_c = postprocess_liver((prob_liver_c > args.thr).astype(np.uint8))
            b = bbox_from_mask(liver_c, margin=int(args.bbox_margin), shape_zyx=liver_c.shape)
        print(f"[BBox] {case_id} bbox={b} thr_bbox={args.thr_bbox} area_frac={args.bbox_area_frac} min_area={args.bbox_min_area}")

        if b is None:
            # fallback: no liver found -> output zeros
            prob_liver_full = np.zeros_like(prob_liver_c, dtype=np.float32)
            pred_liver = np.zeros_like(liver_c, dtype=np.uint8)
        else:
            z0, z1, y0, y1, x0, x1 = b
            img_roi = img_np[:, z0:z1+1, y0:y1+1, x0:x1+1]
            vol_roi = torch.from_numpy(img_roi[None, ...]).to(device)

            if model_r is None:
                # coarse-only: use coarse prob as final
                prob_liver_full = prob_liver_c.astype(np.float32)
            else:
                prob_r = sliding_window_prob(vol_roi, model_r, patch_size, stride, num_classes=meta_r["num_classes"])
                prob_liver_r = prob_r[1].detach().cpu().numpy().astype(np.float32)

                prob_liver_full = np.zeros_like(prob_liver_c, dtype=np.float32)
                prob_liver_full[z0:z1+1, y0:y1+1, x0:x1+1] = prob_liver_r

                raw = (prob_liver_full > args.thr).astype(np.uint8)
            if b is not None:
                z0, z1, y0, y1, x0, x1 = b
                raw2 = np.zeros_like(raw, dtype=np.uint8)
                raw2[z0:z1+1, y0:y1+1, x0:x1+1] = raw[z0:z1+1, y0:y1+1, x0:x1+1]
                raw = raw2
            pred_liver = postprocess_liver(raw)
        np.save(out_dir / f"{case_id}_pred_liver.npy", pred_liver.astype(np.uint8))
        if args.save_prob:
            np.save(out_dir / f"{case_id}_prob_liver.npy", prob_liver_full.astype(np.float32))

    print(f"[DONE] saved to: {out_dir}")


if __name__ == "__main__":
    main()
