#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tumor post-processing utilities.

Goals:
- reduce false positives (e.g., vessels/bile duct/cysts) with simple, controllable rules
- optionally use adaptive (per-case) hysteresis thresholds based on probability distribution
- keep dependencies minimal (scipy optional)

All masks/prob are expected in Z,Y,X order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np


def _try_import_scipy():
    try:
        from scipy.ndimage import label as ndi_label
        return ndi_label
    except Exception:
        return None


def _label_cc(mask01: np.ndarray):
    ndi_label = _try_import_scipy()
    if ndi_label is None:
        # fallback: treat the whole mask as one component
        lab = mask01.astype(np.int32)
        n = int(mask01.max() > 0)
        return lab, n, None
    lab, n = ndi_label(mask01.astype(np.uint8))
    return lab, int(n), ndi_label


def remove_small_cc(mask01: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask01.astype(np.uint8)
    lab, n, ndi_label = _label_cc(mask01)
    if n == 0:
        return mask01.astype(np.uint8)
    if ndi_label is None:
        # fallback: cannot separate CCs
        return mask01.astype(np.uint8) if mask01.sum() >= min_size else np.zeros_like(mask01, dtype=np.uint8)

    sizes = np.bincount(lab.ravel())
    keep = np.zeros_like(sizes, dtype=bool)
    keep[1:] = sizes[1:] >= int(min_size)
    return keep[lab].astype(np.uint8)


def keep_cc_intersect_seed(region01: np.ndarray, seed01: np.ndarray) -> np.ndarray:
    """Keep connected components of region that intersect seed."""
    lab, n, ndi_label = _label_cc(region01)
    if n == 0:
        return region01.astype(np.uint8)
    if ndi_label is None:
        return (region01 & (seed01 > 0)).astype(np.uint8)

    out = np.zeros_like(region01, dtype=np.uint8)
    seed_labels = np.unique(lab[seed01 > 0])
    for lb in seed_labels:
        if lb <= 0:
            continue
        out[lab == lb] = 1
    return out


def _bbox_dims(mask01: np.ndarray) -> Optional[Tuple[int, int, int]]:
    idx = np.where(mask01 > 0)
    if idx[0].size == 0:
        return None
    dz = int(idx[0].max() - idx[0].min() + 1)
    dy = int(idx[1].max() - idx[1].min() + 1)
    dx = int(idx[2].max() - idx[2].min() + 1)
    return dz, dy, dx


def remove_tubular_like(mask01: np.ndarray, tubular_aspect: float = 10.0, tubular_thickness: int = 5) -> np.ndarray:
    """
    Remove components that look very tubular: extreme aspect ratio AND small thickness.
    This is a crude heuristic to suppress vessels/bile ducts.

    tubular_aspect: max_dim / min_dim >= this -> tubular
    tubular_thickness: min_dim <= this -> thin
    """
    if tubular_aspect <= 0 or tubular_thickness <= 0:
        return mask01.astype(np.uint8)

    lab, n, ndi_label = _label_cc(mask01)
    if n == 0:
        return mask01.astype(np.uint8)
    if ndi_label is None:
        return mask01.astype(np.uint8)

    out = mask01.astype(np.uint8).copy()
    for lb in range(1, n + 1):
        comp = (lab == lb)
        dims = _bbox_dims(comp)
        if dims is None:
            continue
        mx = max(dims)
        mn = max(1, min(dims))
        aspect = float(mx) / float(mn)
        thickness = int(min(dims))
        if aspect >= float(tubular_aspect) and thickness <= int(tubular_thickness):
            out[comp] = 0
    return out


@dataclass
class TumorPostprocessConfig:
    # basic threshold
    thr: float = 0.5

    # adaptive hysteresis (if enabled)
    use_hysteresis: bool = False
    q_high: float = 99.5     # percentile on prob (in-liver) -> thr_high
    seed_floor: float = 0.5  # min thr_high
    low_ratio: float = 0.5   # thr_low = max(low_floor, thr_high * low_ratio)
    low_floor: float = 0.2   # min thr_low

    # CC filters
    min_cc: int = 20
    tubular_aspect: float = 10.0
    tubular_thickness: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def postprocess_tumor_prob(
    prob_zyx: np.ndarray,
    liver_mask_zyx: Optional[np.ndarray],
    cfg: TumorPostprocessConfig,
) -> np.ndarray:
    """
    Input:
      prob_zyx: float32 (Z,Y,X), tumor probability
      liver_mask_zyx: optional 0/1 (Z,Y,X) to restrict predictions

    Output:
      tumor mask 0/1 (Z,Y,X)
    """
    p = prob_zyx.astype(np.float32)
    if liver_mask_zyx is not None and liver_mask_zyx.shape == p.shape:
        p = p * (liver_mask_zyx > 0).astype(np.float32)

    if (cfg.use_hysteresis is False) or (cfg.thr >= 0):
        m = (p >= float(cfg.thr)).astype(np.uint8)
    else:
        # adaptive hysteresis: compute per-case thresholds based on prob distribution
        in_mask = (liver_mask_zyx > 0) if (liver_mask_zyx is not None and liver_mask_zyx.shape == p.shape) else (np.ones_like(p, dtype=bool))
        vals = p[in_mask]
        if vals.size == 0:
            m = np.zeros_like(p, dtype=np.uint8)
        else:
            thr_high = float(np.percentile(vals, float(cfg.q_high)))
            thr_high = max(float(cfg.seed_floor), thr_high)
            thr_low = max(float(cfg.low_floor), thr_high * float(cfg.low_ratio))

            seed = (p >= thr_high).astype(np.uint8)
            region = (p >= thr_low).astype(np.uint8)
            m = keep_cc_intersect_seed(region, seed).astype(np.uint8)

    m = remove_small_cc(m, int(cfg.min_cc))
    m = remove_tubular_like(m, tubular_aspect=float(cfg.tubular_aspect), tubular_thickness=int(cfg.tubular_thickness))
    m = remove_small_cc(m, int(cfg.min_cc))
    return m.astype(np.uint8)
