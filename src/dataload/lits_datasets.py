# dataload/lits_datasets.py
# -*- coding: utf-8 -*-
"""
Dataset for nnUNetv2-preprocessed LiTS (MSD Task03_Liver) stored as:
  - <case_id>.b2nd        image (C,Z,Y,X) float32
  - <case_id>_seg.b2nd    label (Z,Y,X) or (1,Z,Y,X) int16
  - <case_id>.pkl         properties (only used for listing case ids)

Key updates (to support LiTS->unlabeled CT transfer + fix prior issues):
- Optional liver ROI sampling for liver stage (liver_use_bbox): coarse vs refine training.
- Optional ROI-global coords channels (add_coords): coords are created on ROI/full volume first,
  then patch is cropped from that ROI -> coords are consistent (NOT per-patch local coords).
- Deterministic center-crop for val (train=False) for more stable model selection.
- SDF target generation uses SimpleITK if available; otherwise falls back to scipy distance transform.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, Optional, List

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---- optional deps for SDF ----
_HAS_SITK = False
try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    sitk = None  # type: ignore

_HAS_SCIPY = False
try:
    from scipy.ndimage import distance_transform_edt
    _HAS_SCIPY = True
except Exception:
    distance_transform_edt = None  # type: ignore


def _coords_zyx(shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    """Return (3,Z,Y,X) coords normalized to [-1,1] over the given volume."""
    Z, Y, X = shape_zyx
    zz = np.linspace(-1.0, 1.0, Z, dtype=np.float32)[:, None, None]
    yy = np.linspace(-1.0, 1.0, Y, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, X, dtype=np.float32)[None, None, :]
    zc = np.broadcast_to(zz, (Z, Y, X))
    yc = np.broadcast_to(yy, (Z, Y, X))
    xc = np.broadcast_to(xx, (Z, Y, X))
    return np.stack([zc, yc, xc], axis=0)  # (3,Z,Y,X)


class LITSDatasetB2ND(Dataset):
    """
    stage="liver":  0=background, 1=liver (including tumor)
    stage="tumor":  0=non-tumor,  1=tumor

    Tumor stage always samples inside liver ROI (from GT seg).
    Liver stage can be coarse (full-volume sampling) or refine (liver ROI sampling).
    """
    def __init__(
        self,
        preproc_dir: str,
        stage: str = "liver",
        patch_size: Tuple[int, int, int] = (96, 160, 160),
        train: bool = True,
        train_ratio: float = 0.8,
        seed: int = 0,
        return_sdf: bool = False,
        sdf_clip: float = 20.0,
        sdf_margin: int = 0,
        liver_use_bbox: bool = False,
        liver_bbox_margin: int = 16,
        add_coords: bool = False,
        tumor_bbox_margin: int = 10,
    ):
        assert stage in ("liver", "tumor")
        self.stage = stage
        self.patch_size = tuple(int(x) for x in patch_size)
        self.preproc_dir = Path(preproc_dir)

        self.train = bool(train)
        self.return_sdf = bool(return_sdf)
        self.sdf_clip = float(sdf_clip)
        self.sdf_margin = int(sdf_margin)

        self.liver_use_bbox = bool(liver_use_bbox)
        self.liver_bbox_margin = int(liver_bbox_margin)
        self.tumor_bbox_margin = int(tumor_bbox_margin)
        self.add_coords = bool(add_coords)

        all_pkl = sorted(self.preproc_dir.glob("*.pkl"))
        case_ids = [p.stem for p in all_pkl]
        if not case_ids:
            raise RuntimeError(f"No .pkl files found in {preproc_dir}")

        rng = random.Random(seed)
        rng.shuffle(case_ids)
        n_train = int(len(case_ids) * float(train_ratio))
        self.case_ids: List[str] = case_ids[:n_train] if self.train else case_ids[n_train:]

        print(
            f"[LITSDatasetB2ND] stage={stage}, train={self.train}, "
            f"num_cases={len(self.case_ids)}, patch_size={self.patch_size}, "
            f"liver_use_bbox={self.liver_use_bbox}, add_coords={self.add_coords}, return_sdf={self.return_sdf}"
        )

    def __len__(self) -> int:
        return len(self.case_ids)

    def _load_case_np(self, case_id: str):
        dparams = {"nthreads": 1}
        data_file = self.preproc_dir / f"{case_id}.b2nd"
        seg_file = self.preproc_dir / f"{case_id}_seg.b2nd"

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

    @staticmethod
    def _remap_labels(seg_zyx: np.ndarray, stage: str) -> np.ndarray:
        if stage == "liver":
            return (seg_zyx >= 1).astype(np.int64)
        return (seg_zyx == 2).astype(np.int64)

    @staticmethod
    def _get_bbox_from_mask(mask_zyx: np.ndarray, margin: int = 0) -> Optional[Tuple[int, int, int, int, int, int]]:
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

    @staticmethod
    def _crop_zyx(img_czyx: np.ndarray, seg_zyx: np.ndarray, bbox) -> Tuple[np.ndarray, np.ndarray]:
        z0, z1, y0, y1, x0, x1 = bbox
        img = img_czyx[:, z0:z1+1, y0:y1+1, x0:x1+1]
        seg = seg_zyx[z0:z1+1, y0:y1+1, x0:x1+1]
        return img, seg

    @staticmethod
    def _pad_to_at_least(img_czyx: np.ndarray, seg_zyx: np.ndarray, target_zyx: Tuple[int, int, int]):
        _, Z, Y, X = img_czyx.shape
        pz, py, px = target_zyx
        pad_z = max(0, pz - Z)
        pad_y = max(0, py - Y)
        pad_x = max(0, px - X)
        if pad_z or pad_y or pad_x:
            img_czyx = np.pad(img_czyx, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
            seg_zyx = np.pad(seg_zyx, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        return img_czyx, seg_zyx

    def _random_crop(self, img_czyx: np.ndarray, seg_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_czyx, seg_zyx = self._pad_to_at_least(img_czyx, seg_zyx, self.patch_size)
        _, Z, Y, X = img_czyx.shape
        pz, py, px = self.patch_size
        z0 = random.randint(0, max(0, Z - pz))
        y0 = random.randint(0, max(0, Y - py))
        x0 = random.randint(0, max(0, X - px))
        return (
            img_czyx[:, z0:z0+pz, y0:y0+py, x0:x0+px],
            seg_zyx[z0:z0+pz, y0:y0+py, x0:x0+px],
        )

    def _center_crop(self, img_czyx: np.ndarray, seg_zyx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        img_czyx, seg_zyx = self._pad_to_at_least(img_czyx, seg_zyx, self.patch_size)
        _, Z, Y, X = img_czyx.shape
        pz, py, px = self.patch_size
        z0 = max(0, (Z - pz) // 2)
        y0 = max(0, (Y - py) // 2)
        x0 = max(0, (X - px) // 2)
        return (
            img_czyx[:, z0:z0+pz, y0:y0+py, x0:x0+px],
            seg_zyx[z0:z0+pz, y0:y0+py, x0:x0+px],
        )

    @staticmethod
    def _compute_sdf_from_binary(mask_zyx: np.ndarray, clip: float = 50.0, margin: int = 0) -> np.ndarray:
        """
        Signed distance field in voxel units:
          sdf > 0 inside foreground, sdf < 0 outside.
        Output is clipped and normalized to ~[-1,1].
        """
        m = (mask_zyx > 0).astype(np.uint8)
        if margin > 0:
            m = np.pad(m, ((margin, margin), (margin, margin), (margin, margin)), mode="constant")

        if _HAS_SITK:
            img = sitk.GetImageFromArray(m)  # type: ignore
            dist = sitk.SignedMaurerDistanceMap(img, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)  # type: ignore
            sdf = sitk.GetArrayFromImage(dist).astype(np.float32)  # type: ignore
        else:
            if not _HAS_SCIPY:
                sdf = np.zeros_like(m, dtype=np.float32)
            else:
                inside = distance_transform_edt(m.astype(bool)).astype(np.float32)  # type: ignore
                outside = distance_transform_edt((1 - m).astype(bool)).astype(np.float32)  # type: ignore
                sdf = inside - outside

        if margin > 0:
            sdf = sdf[margin:-margin, margin:-margin, margin:-margin]

        if clip is not None and clip > 0:
            sdf = np.clip(sdf, -clip, clip) / clip
        return sdf.astype(np.float32)

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        img, seg = self._load_case_np(case_id)     # img:(C,Z,Y,X), seg:(Z,Y,X)

        # ---- ROI selection (before coords) ----
        if self.stage == "tumor":
            liver_mask = (seg >= 1).astype(np.uint8)
            bbox = self._get_bbox_from_mask(liver_mask, margin=self.tumor_bbox_margin)
            if bbox is not None:
                img, seg = self._crop_zyx(img, seg, bbox)
        elif self.stage == "liver" and self.liver_use_bbox:
            liver_mask = (seg >= 1).astype(np.uint8)
            bbox = self._get_bbox_from_mask(liver_mask, margin=self.liver_bbox_margin)
            if bbox is not None:
                img, seg = self._crop_zyx(img, seg, bbox)

        # ---- coords on ROI/full volume (global in this ROI) ----
        if self.add_coords:
            coords = _coords_zyx(seg.shape).astype(np.float32)  # (3,Z,Y,X)
            img = np.concatenate([img, coords], axis=0)

        # ---- patch crop ----
        if self.train:
            img, seg = self._random_crop(img, seg)
        else:
            img, seg = self._center_crop(img, seg)

        tgt = self._remap_labels(seg, self.stage)  # (Z,Y,X) 0/1

        img_t = torch.from_numpy(img).float()                # (C,Z,Y,X)
        tgt_t = torch.from_numpy(tgt).long()                 # (Z,Y,X)

        if self.return_sdf and self.stage == "liver":
            sdf = self._compute_sdf_from_binary(tgt, clip=self.sdf_clip, margin=self.sdf_margin)
        else:
            sdf = np.zeros_like(tgt, dtype=np.float32)
        sdf_t = torch.from_numpy(sdf).float().unsqueeze(0)   # (1,Z,Y,X)

        return img_t, tgt_t, sdf_t, case_id
