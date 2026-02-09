#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-volume tumor inference with sliding window, using liver predictions to define ROI bbox.

Inputs:
  - preproc_dir: nnUNetv2-preprocessed LiTS-style .b2nd volumes
  - ckpt_tumor: tumor segmentation model
  - liver_dir: directory containing liver predictions from infer.py:
      <case>_pred_liver.npy (0/1) and optionally <case>_prob_liver.npy (0..1)

Outputs (per case):
  - <out_dir>/<case>_tumor_pred.npy  (uint8 0/1, Z,Y,X)
  - <out_dir>/<case>_tumor_prob.npy  (float32 0..1, Z,Y,X)  [if --save_prob]

Postprocess can be tuned by tune_tumor_postprocess.py (JSON) or set via CLI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, List, Optional

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet3D import UNet3D
from tumor_postprocess import TumorPostprocessConfig, postprocess_tumor_prob


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
    p.add_argument("--tumor_prior_type", type=str, default="prob", choices=["prob", "mask"],
                   help="if tumor model expects 2 channels, use liver prob or liver mask as prior channel")
    p.add_argument("--liver_bbox_from", type=str, default="mask", choices=["mask", "prob", "union"])
    p.add_argument("--liver_prob_thr", type=float, default=0.1)

    # postprocess (if postprocess_json provided, it overrides these)
    p.add_argument("--postprocess_json", type=str, default=None)
    p.add_argument("--thr", type=float, default=0.5, help="fixed threshold; set -1 to enable hysteresis mode")
    p.add_argument("--use_hysteresis", type=int, default=0, choices=[0, 1])
    p.add_argument("--q_high", type=float, default=99.5)
    p.add_argument("--seed_floor", type=float, default=0.5)
    p.add_argument("--low_ratio", type=float, default=0.5)
    p.add_argument("--low_floor", type=float, default=0.2)
    p.add_argument("--min_cc", type=int, default=20)
    p.add_argument("--tubular_aspect", type=float, default=10.0)
    p.add_argument("--tubular_thickness", type=int, default=5)

    p.add_argument("--save_prob", action="store_true")
    return p.parse_args()


def get_logits(model_out):
    if isinstance(model_out, dict):
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        return model_out[0]
    return model_out


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
                    probs = torch.softmax(logits, dim=1)[0]
                    prob[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs
                    w[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0

    prob = prob / torch.clamp_min(w, 1.0)
    oZ, oY, oX = orig
    return prob[:, :oZ, :oY, :oX]


def bbox_from_mask(mask: np.ndarray, margin: int) -> Optional[Tuple[int, int, int, int, int, int]]:
    idx = np.where(mask > 0)
    if idx[0].size == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    x0, x1 = int(idx[2].min()), int(idx[2].max())
    z0 -= margin; y0 -= margin; x0 -= margin
    z1 += margin; y1 += margin; x1 += margin
    Z, Y, X = mask.shape
    z0 = max(0, min(z0, Z - 1)); z1 = max(0, min(z1, Z - 1))
    y0 = max(0, min(y0, Y - 1)); y1 = max(0, min(y1, Y - 1))
    x0 = max(0, min(x0, X - 1)); x1 = max(0, min(x1, X - 1))
    return z0, z1, y0, y1, x0, x1


def load_case_b2nd(preproc_dir: Path, case_id: str) -> np.ndarray:
    dparams = {"nthreads": 1}
    data_file = preproc_dir / f"{case_id}.b2nd"
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    img = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    return img


def list_case_ids(preproc_dir: Path) -> List[str]:
    return [p.stem for p in sorted(preproc_dir.glob("*.pkl"))]


def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    use_coords = bool(ckpt.get("use_coords", False))
    use_sdf_head = bool(ckpt.get("use_sdf_head", False))

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    meta = dict(in_channels=in_channels, num_classes=num_classes, use_coords=use_coords, use_sdf_head=use_sdf_head)
    return model, meta


def load_liver_outputs(
    liver_dir: Path,
    case_id: str,
    shape_zyx: Tuple[int, int, int],
    src: str,
    thr: float
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      liver_bbox_mask (uint8 0/1, Z,Y,X), and liver_prob (float32, Z,Y,X) if exists.
    """
    mask = None
    for name in [f"{case_id}_pred_liver.npy", f"{case_id}_liver_pred.npy"]:
        p = liver_dir / name
        if p.is_file():
            mask = (np.load(p) > 0).astype(np.uint8)
            break
    prob = None
    for name in [f"{case_id}_prob_liver.npy", f"{case_id}_liver_prob.npy"]:
        p = liver_dir / name
        if p.is_file():
            prob = np.load(p).astype(np.float32)
            break

    Z, Y, X = shape_zyx
    if mask is not None:
        mask = mask[:Z, :Y, :X]
        if mask.shape != shape_zyx:
            pad = np.zeros(shape_zyx, dtype=np.uint8)
            zz, yy, xx = mask.shape
            pad[:zz, :yy, :xx] = mask
            mask = pad
    if prob is not None:
        prob = prob[:Z, :Y, :X]
        if prob.shape != shape_zyx:
            pad = np.zeros(shape_zyx, dtype=np.float32)
            zz, yy, xx = prob.shape
            pad[:zz, :yy, :xx] = prob
            prob = pad

    if mask is None and prob is None:
        return np.zeros(shape_zyx, dtype=np.uint8), None

    src = src.lower().strip()
    if src not in ["mask", "prob", "union"]:
        src = "mask"

    if src == "mask":
        bbox_mask = mask if mask is not None else (prob >= float(thr)).astype(np.uint8)
    elif src == "prob":
        if prob is None:
            bbox_mask = mask if mask is not None else np.zeros(shape_zyx, dtype=np.uint8)
        else:
            bbox_mask = (prob >= float(thr)).astype(np.uint8)
    else:
        m0 = mask if mask is not None else np.zeros(shape_zyx, dtype=np.uint8)
        if prob is None:
            bbox_mask = m0
        else:
            pm = (prob >= float(thr)).astype(np.uint8)
            bbox_mask = ((m0 > 0) | (pm > 0)).astype(np.uint8)

    return bbox_mask.astype(np.uint8), prob


def build_cfg(args) -> TumorPostprocessConfig:
    if args.postprocess_json:
        j = json.loads(Path(args.postprocess_json).read_text(encoding="utf-8"))
        cfg_dict = j.get("cfg", j)
        return TumorPostprocessConfig(**cfg_dict)

    cfg = TumorPostprocessConfig(
        thr=float(args.thr),
        use_hysteresis=bool(args.use_hysteresis),
        q_high=float(args.q_high),
        seed_floor=float(args.seed_floor),
        low_ratio=float(args.low_ratio),
        low_floor=float(args.low_floor),
        min_cc=int(args.min_cc),
        tubular_aspect=float(args.tubular_aspect),
        tubular_thickness=int(args.tubular_thickness),
    )
    return cfg


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
    cfg = build_cfg(args)
    print("[Tumor] postprocess cfg:", cfg)

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    case_ids = [args.case_id] if args.case_id else list_case_ids(preproc_dir)
    print(f"Found {len(case_ids)} case(s).")

    for case_id in tqdm(case_ids, desc="Infer-Tumor", unit="case"):
        img = load_case_b2nd(preproc_dir, case_id)  # (C,Z,Y,X)
        C, Z, Y, X = img.shape

        # liver bbox mask
        liver_bbox_mask, liver_prob = load_liver_outputs(liver_dir, case_id, (Z, Y, X), args.liver_bbox_from, args.liver_prob_thr)
        b = bbox_from_mask(liver_bbox_mask, margin=int(args.bbox_margin))
        if b is None:
            b = (0, Z - 1, 0, Y - 1, 0, X - 1)
        z0, z1, y0, y1, x0, x1 = b

        # crop ROI for inference
        roi = img[:, z0:z1+1, y0:y1+1, x0:x1+1]  # (C,z,y,x)

        in_ch = int(meta["in_channels"])
        if in_ch == 1:
            in_roi = roi
        else:
            # add liver prior channel
            if (args.tumor_prior_type == "prob") and (liver_prob is not None):
                prior_full = liver_prob
                prior_roi = prior_full[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
            else:
                prior_roi = liver_bbox_mask[z0:z1+1, y0:y1+1, x0:x1+1][None, ...].astype(np.float32)
            in_roi = np.concatenate([roi, prior_roi], axis=0).astype(np.float32)

        vol = torch.from_numpy(in_roi[None, ...]).to(device)

        prob_roi = sliding_window_prob(vol, model, patch_size, stride, num_classes=meta["num_classes"])
        tumor_prob_roi = prob_roi[1].detach().cpu().numpy().astype(np.float32)

        # paste back to full volume prob
        prob_full = np.zeros((Z, Y, X), dtype=np.float32)
        prob_full[z0:z1+1, y0:y1+1, x0:x1+1] = tumor_prob_roi

        pred_full = postprocess_tumor_prob(prob_full, liver_bbox_mask, cfg)

        np.save(out_dir / f"{case_id}_tumor_pred.npy", pred_full.astype(np.uint8))
        if args.save_prob:
            np.save(out_dir / f"{case_id}_tumor_prob.npy", prob_full.astype(np.float32))

    print("Done.")


if __name__ == "__main__":
    main()
