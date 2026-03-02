#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-network inference (NO postprocess):
- Liver: run coarse + refine on FULL volume, output argmax masks and prob maps
- Tumor: run tumor on FULL volume; if tumor_ckpt needs prior, use liver_prob (raw, no postprocess) as 2nd channel
Outputs:
  <out_dir>/<case>_prob_liver_coarse_raw.npy
  <out_dir>/<case>_pred_liver_coarse_raw.npy
  <out_dir>/<case>_prob_liver_refine_raw.npy
  <out_dir>/<case>_pred_liver_refine_raw.npy
  <out_dir>/<case>_prob_tumor_raw.npy
  <out_dir>/<case>_pred_tumor_raw.npy
  <out_dir>/<case>_seg012_raw.npy   (0=bg, 1=liver, 2=tumor)
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


def get_logits(out):
    if isinstance(out, dict):
        if "logits" in out:
            return out["logits"]
        if "out" in out:
            return out["out"]
    return out


def list_case_ids(preproc_dir: Path) -> List[str]:
    return [p.stem for p in sorted(preproc_dir.glob("*.pkl"))]


def load_b2nd(preproc_dir: Path, case_id: str) -> np.ndarray:
    data_file = preproc_dir / f"{case_id}.b2nd"
    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams={"nthreads": 1})
    arr = data_b[:].astype(np.float32)  # (C,Z,Y,X)
    if arr.ndim == 3:
        arr = arr[None, ...]
    return arr


def pad_to_min(volume: torch.Tensor, patch: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    _, _, Z, Y, X = volume.shape
    pz, py, px = patch
    pad_z = max(0, pz - Z)
    pad_y = max(0, py - Y)
    pad_x = max(0, px - X)
    if pad_z or pad_y or pad_x:
        volume = F.pad(volume, (0, pad_x, 0, pad_y, 0, pad_z))
    return volume, (Z, Y, X)


@torch.inference_mode()
def sliding_window_prob(
    volume: torch.Tensor,  # (1,C,Z,Y,X)
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int,
    use_amp: bool,
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
    if z_starts[-1] != Z - pz:
        z_starts.append(max(Z - pz, 0))
    if y_starts[-1] != Y - py:
        y_starts.append(max(Y - py, 0))
    if x_starts[-1] != X - px:
        x_starts.append(max(X - px, 0))

    total = len(z_starts) * len(y_starts) * len(x_starts)
    pbar = tqdm(total=total, desc="SW", leave=False)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = volume[:, :, z0:z0+pz, y0:y0+py, x0:x0+px]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(patch)
                    logits = get_logits(out)
                    probs = torch.softmax(logits, dim=1)[0]  # (C,pz,py,px)
                prob[:, z0:z0+pz, y0:y0+py, x0:x0+px] += probs
                w[:, z0:z0+pz, y0:y0+py, x0:x0+px] += 1.0
                pbar.update(1)

    pbar.close()
    prob = prob / torch.clamp(w, min=1.0)
    oZ, oY, oX = orig
    return prob[:, :oZ, :oY, :oX]


def load_model_robust(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
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

    sd = ckpt["model_state"]
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model_has_net = any(k.startswith("net.") for k in model.state_dict().keys())
    sd_has_net = any(k.startswith("net.") for k in sd.keys())
    if model_has_net and (not sd_has_net):
        sd = {("net." + k): v for k, v in sd.items()}
    elif (not model_has_net) and sd_has_net:
        sd = {k.replace("net.", "", 1): v for k, v in sd.items()}

    model.load_state_dict(sd, strict=True)
    model.eval()
    meta = dict(in_channels=in_channels, num_classes=num_classes, use_coords=use_coords, use_sdf_head=use_sdf_head,
                backbone=backbone, mednext_k=mednext_k, mednext_expansion=mednext_expansion, mednext_blocks=mednext_blocks)
    return model, meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--case_id", type=str, default=None)

    p.add_argument("--ckpt_coarse", type=str, required=True)
    p.add_argument("--ckpt_refine", type=str, required=True)
    p.add_argument("--ckpt_tumor", type=str, required=True)

    p.add_argument("--patch_liver", type=int, nargs=3, default=[96, 160, 160])
    p.add_argument("--stride_liver", type=int, nargs=3, default=[48, 80, 80])
    p.add_argument("--patch_tumor", type=int, nargs=3, default=[96, 160, 160])
    p.add_argument("--stride_tumor", type=int, nargs=3, default=[48, 80, 80])

    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_prob", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(args.use_amp)

    case_ids = [args.case_id] if args.case_id else list_case_ids(preproc_dir)
    print(f"[RAW-INFER] cases={len(case_ids)} device={device} use_amp={use_amp}")

    liver_coarse, meta_c = load_model_robust(args.ckpt_coarse, device)
    liver_refine, meta_r = load_model_robust(args.ckpt_refine, device)
    tumor_model, meta_t = load_model_robust(args.ckpt_tumor, device)

    for cid in tqdm(case_ids, desc="Cases"):
        arr = load_b2nd(preproc_dir, cid)      # (C,Z,Y,X)
        ct = arr[0:1]                          # (1,Z,Y,X)
        Z, Y, X = ct.shape[1:]

        # ---- Liver coarse (FULL VOL) ----
        vol = torch.from_numpy(ct[None]).to(device)  # (1,1,Z,Y,X)
        prob_c = sliding_window_prob(
            vol, liver_coarse,
            tuple(map(int, args.patch_liver)),
            tuple(map(int, args.stride_liver)),
            num_classes=int(meta_c["num_classes"]),
            use_amp=use_amp
        )
        prob_liver_c = prob_c[1].detach().cpu().numpy().astype(np.float32)  # (Z,Y,X)
        pred_liver_c = (prob_c.argmax(dim=0) == 1).detach().cpu().numpy().astype(np.uint8)

        np.save(out_dir / f"{cid}_prob_liver_coarse_raw.npy", prob_liver_c)
        np.save(out_dir / f"{cid}_pred_liver_coarse_raw.npy", pred_liver_c)

        # ---- Liver refine (FULL VOL) ----
        prob_r = sliding_window_prob(
            vol, liver_refine,
            tuple(map(int, args.patch_liver)),
            tuple(map(int, args.stride_liver)),
            num_classes=int(meta_r["num_classes"]),
            use_amp=use_amp
        )
        prob_liver_r = prob_r[1].detach().cpu().numpy().astype(np.float32)
        pred_liver_r = (prob_r.argmax(dim=0) == 1).detach().cpu().numpy().astype(np.uint8)

        np.save(out_dir / f"{cid}_prob_liver_refine_raw.npy", prob_liver_r)
        np.save(out_dir / f"{cid}_pred_liver_refine_raw.npy", pred_liver_r)

        # ---- Tumor (FULL VOL) ----
        in_ch_t = int(meta_t["in_channels"])
        if in_ch_t == 1:
            tumor_in = ct
        else:
            # tumor needs prior: use RAW liver prob (refine)
            prior = prob_liver_r[None, ...].astype(np.float32)  # (1,Z,Y,X)
            tumor_in = np.concatenate([ct, prior], axis=0)      # (2,Z,Y,X)

        vol_t = torch.from_numpy(tumor_in[None]).to(device)     # (1,C,Z,Y,X)
        prob_t = sliding_window_prob(
            vol_t, tumor_model,
            tuple(map(int, args.patch_tumor)),
            tuple(map(int, args.stride_tumor)),
            num_classes=int(meta_t["num_classes"]),
            use_amp=use_amp
        )
        prob_tumor = prob_t[1].detach().cpu().numpy().astype(np.float32)
        pred_tumor = (prob_t.argmax(dim=0) == 1).detach().cpu().numpy().astype(np.uint8)

        np.save(out_dir / f"{cid}_pred_tumor_raw.npy", pred_tumor)
        if int(args.save_prob):
            np.save(out_dir / f"{cid}_prob_tumor_raw.npy", prob_tumor)

        # ---- combined seg: 0 bg, 1 liver, 2 tumor (tumor overrides) ----
        seg012 = pred_liver_r.astype(np.uint8)
        seg012[pred_tumor > 0] = 2
        np.save(out_dir / f"{cid}_seg012_raw.npy", seg012.astype(np.uint8))

    print("Done.")


if __name__ == "__main__":
    main()
