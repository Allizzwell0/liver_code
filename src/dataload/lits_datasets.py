# dataload/lits_datasets.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, Optional, Dict

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk


def _bbox_from_mask(mask_zyx: np.ndarray, margin: int = 0) -> Optional[Tuple[int, int, int, int, int, int]]:
    """Return (z0,z1,y0,y1,x0,x1) inclusive bounds. None if empty."""
    idx = np.where(mask_zyx > 0)
    if idx[0].size == 0:
        return None
    z0, z1 = int(idx[0].min()), int(idx[0].max())
    y0, y1 = int(idx[1].min()), int(idx[1].max())
    x0, x1 = int(idx[2].min()), int(idx[2].max())
    z0 -= margin; y0 -= margin; x0 -= margin
    z1 += margin; y1 += margin; x1 += margin
    return z0, z1, y0, y1, x0, x1


def _clamp_bbox(b: Tuple[int, int, int, int, int, int], Z: int, Y: int, X: int) -> Tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = b
    z0 = max(0, min(z0, Z - 1)); z1 = max(0, min(z1, Z - 1))
    y0 = max(0, min(y0, Y - 1)); y1 = max(0, min(y1, Y - 1))
    x0 = max(0, min(x0, X - 1)); x1 = max(0, min(x1, X - 1))
    if z1 < z0: z0, z1 = z1, z0
    if y1 < y0: y0, y1 = y1, y0
    if x1 < x0: x0, x1 = x1, x0
    return z0, z1, y0, y1, x0, x1


def _pad_end_img_seg(img_czyx: np.ndarray, seg_zyx: np.ndarray, patch_size: Tuple[int, int, int]):
    """Pad at the END so that (Z,Y,X) >= patch_size."""
    _, Z, Y, X = img_czyx.shape
    pz, py, px = patch_size
    pad_z = max(0, pz - Z)
    pad_y = max(0, py - Y)
    pad_x = max(0, px - X)
    if pad_z or pad_y or pad_x:
        img_czyx = np.pad(img_czyx, ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
        seg_zyx = np.pad(seg_zyx, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    return img_czyx, seg_zyx


def _pad_end_vol(vol_zyx: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
    Z, Y, X = vol_zyx.shape
    pz, py, px = patch_size
    pad_z = max(0, pz - Z)
    pad_y = max(0, py - Y)
    pad_x = max(0, px - X)
    if pad_z or pad_y or pad_x:
        vol_zyx = np.pad(vol_zyx, ((0, pad_z), (0, pad_y), (0, pad_x)), mode="constant")
    return vol_zyx


def _randint(lo: int, hi: int) -> int:
    # inclusive bounds
    if hi <= lo:
        return int(lo)
    return int(np.random.randint(lo, hi + 1))


def _start_range_for_center(center: int, p: int, d0: int, d1: int, D: int) -> Optional[Tuple[int, int]]:
    """
    Choose start s such that:
      s <= center <= s+p-1
      d0 <= s <= d1-p+1  (keep inside bbox)
      0 <= s <= D-p      (keep inside volume)
    Return (low, high) inclusive. None if impossible.
    """
    s_low = max(0, center - p + 1, d0)
    s_high = min(D - p, center, d1 - p + 1)
    if s_low > s_high:
        return None
    return int(s_low), int(s_high)


class LITSDatasetB2ND(Dataset):
    """
    nnUNetv2 预处理后的 LiTS (.b2nd + .pkl)

    - liver:
        * liver_use_bbox=0: 全体积随机裁剪
        * liver_use_bbox=1: 在 bbox(来自 GT 或 pred) 内随机裁剪
        * return_sdf=True: 额外返回 liver mask 的 SDF（用于 sdf head）

    - tumor:
        * ROI bbox 来自 GT liver 或 pred liver（可按 tumor_pred_bbox_ratio 混合）
        * patch 采样：pos / hardneg / random（由 tumor_pos_ratio / tumor_hardneg_ratio 控制）
        * 可追加 liver prior 通道（mask/prob）
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

        # liver
        liver_use_bbox: bool = True,
        liver_bbox_margin: int = 16,
        liver_use_pred_bbox: bool = False,
        liver_pred_dir: Optional[str] = None,
        liver_add_pred_prior: bool = False,

        # tumor
        tumor_use_pred_liver: bool = False,
        tumor_pred_liver_dir: Optional[str] = None,
        tumor_pred_bbox_ratio: float = 1.0,
        tumor_add_liver_prior: bool = False,
        tumor_prior_type: str = "mask",  # mask|prob
        tumor_bbox_margin: int = 24,
        tumor_pos_ratio: float = 0.6,
        tumor_hardneg_ratio: float = 0.3,
    ):
        self.preproc_dir = Path(preproc_dir)
        self.stage = stage
        assert stage in ["liver", "tumor"]
        self.train = bool(train)
        self.patch_size = tuple(patch_size)
        self.return_sdf = bool(return_sdf) and (stage == "liver")
        self.sdf_clip = float(sdf_clip)
        self.sdf_margin = int(sdf_margin)

        # liver options
        self.liver_use_bbox = bool(liver_use_bbox) and (stage == "liver")
        self.liver_bbox_margin = int(liver_bbox_margin)
        self.liver_use_pred_bbox = bool(liver_use_pred_bbox) and (stage == "liver")
        self.liver_pred_dir = Path(liver_pred_dir) if liver_pred_dir else None
        self.liver_add_pred_prior = bool(liver_add_pred_prior) and (stage == "liver")

        # tumor options
        self.tumor_use_pred_liver = bool(tumor_use_pred_liver) and (stage == "tumor")
        self.tumor_pred_liver_dir = Path(tumor_pred_liver_dir) if tumor_pred_liver_dir else None
        self.tumor_pred_bbox_ratio = float(tumor_pred_bbox_ratio)
        self.tumor_add_liver_prior = bool(tumor_add_liver_prior) and (stage == "tumor")
        self.tumor_prior_type = str(tumor_prior_type)
        self.tumor_bbox_margin = int(tumor_bbox_margin)
        self.tumor_pos_ratio = float(tumor_pos_ratio)
        self.tumor_hardneg_ratio = float(tumor_hardneg_ratio)

        all_pkl = sorted(self.preproc_dir.glob("*.pkl"))
        case_ids = [p.stem for p in all_pkl]
        if len(case_ids) == 0:
            raise RuntimeError(f"No .pkl files found in {preproc_dir}")

        rng = random.Random(seed)
        rng.shuffle(case_ids)
        n_train = int(len(case_ids) * train_ratio)
        self.case_ids = case_ids[:n_train] if train else case_ids[n_train:]

        blosc2.set_nthreads(1)

        msg = f"[LITSDatasetB2ND] stage={stage}, train={train}, num_cases={len(self.case_ids)}, patch_size={self.patch_size}"
        if stage == "liver":
            msg += f", liver_use_bbox={self.liver_use_bbox}, liver_use_pred_bbox={self.liver_use_pred_bbox}, return_sdf={self.return_sdf}"
            if self.liver_add_pred_prior:
                msg += ", liver_add_pred_prior=True"
        else:
            msg += f", tumor_use_pred_liver={self.tumor_use_pred_liver}, tumor_pred_bbox_ratio={self.tumor_pred_bbox_ratio:.2f}"
            if self.tumor_add_liver_prior:
                msg += f", tumor_prior={self.tumor_prior_type}"
        print(msg)

    def __len__(self):
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
            seg = seg[0]  # (Z,Y,X)
        return img, seg

    def _load_pred_liver(self, case_id: str) -> Optional[np.ndarray]:
        """Load pred liver mask (Z,Y,X) 0/1 if exists."""
        if not self.liver_pred_dir and not self.tumor_pred_liver_dir:
            return None
        dirs = []
        if self.stage == "liver" and self.liver_pred_dir:
            dirs.append(self.liver_pred_dir)
        if self.stage == "tumor" and self.tumor_pred_liver_dir:
            dirs.append(self.tumor_pred_liver_dir)

        for d in dirs:
            p = d / f"{case_id}_pred_liver.npy"
            if p.is_file():
                m = np.load(p)
                m = (m > 0).astype(np.uint8)
                m = _pad_end_vol(m, self.patch_size)
                return m
            # fallback names (older scripts)
            p2 = d / f"{case_id}_liver_pred.npy"
            if p2.is_file():
                m = np.load(p2)
                m = (m > 0).astype(np.uint8)
                m = _pad_end_vol(m, self.patch_size)
                return m
        return None

    def _load_pred_liver_prob(self, case_id: str) -> Optional[np.ndarray]:
        """Load pred liver prob (Z,Y,X) float if exists."""
        if self.stage != "tumor" or (not self.tumor_pred_liver_dir):
            return None
        d = self.tumor_pred_liver_dir
        for name in [f"{case_id}_prob_liver.npy", f"{case_id}_liver_prob.npy", f"{case_id}_pred_liver_prob.npy"]:
            p = d / name
            if p.is_file():
                prob = np.load(p).astype(np.float32)
                prob = _pad_end_vol(prob, self.patch_size)
                return prob
        return None

    @staticmethod
    def _compute_sdf(binary_mask_zyx: np.ndarray, clip: float = 20.0) -> np.ndarray:
        """Signed distance field in [-1,1] (after clip & normalize)."""
        mask = (binary_mask_zyx > 0).astype(np.uint8)
        itk = sitk.GetImageFromArray(mask)
        dist_out = sitk.SignedMaurerDistanceMap(itk, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        sdf = sitk.GetArrayFromImage(dist_out).astype(np.float32)
        if clip > 0:
            sdf = np.clip(sdf, -clip, clip) / clip
        return sdf

    # --------- light data augmentation (no extra deps) ---------
    def _augment_patch(self, img_czyx: np.ndarray, seg_zyx: np.ndarray, strong: bool = False):
        """Augment a cropped patch (numpy).

        - flips: always enabled
        - intensity/noise: only when strong=True (recommended for tumor)
        """
        # random flips on spatial axes (Z,Y,X)
        if np.random.rand() < 0.5:
            img_czyx = img_czyx[:, ::-1, :, :]
            seg_zyx = seg_zyx[::-1, :, :]
        if np.random.rand() < 0.5:
            img_czyx = img_czyx[:, :, ::-1, :]
            seg_zyx = seg_zyx[:, ::-1, :]
        if np.random.rand() < 0.5:
            img_czyx = img_czyx[:, :, :, ::-1]
            seg_zyx = seg_zyx[:, :, ::-1]

        if strong:
            # mild scale/shift (assumes input already normalized by preprocessing)
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.1, 0.1)
            img_czyx = img_czyx * scale + shift
            # gaussian noise
            if np.random.rand() < 0.5:
                img_czyx = img_czyx + np.random.normal(0.0, 0.03, size=img_czyx.shape).astype(np.float32)

        return img_czyx, seg_zyx

    def _random_crop_in_bbox(self, img: np.ndarray, seg: np.ndarray, bbox: Tuple[int, int, int, int, int, int]):
        """Random crop fully inside bbox (or best-effort if bbox smaller)."""
        _, Z, Y, X = img.shape
        pz, py, px = self.patch_size
        z0, z1, y0, y1, x0, x1 = _clamp_bbox(bbox, Z, Y, X)

        def pick_start(d0, d1, p, D):
            if (d1 - d0 + 1) <= p:
                return max(0, min(d0, D - p))
            return int(np.random.randint(d0, d1 - p + 2))

        sz = pick_start(z0, z1, pz, Z)
        sy = pick_start(y0, y1, py, Y)
        sx = pick_start(x0, x1, px, X)
        return img[:, sz:sz+pz, sy:sy+py, sx:sx+px], seg[sz:sz+pz, sy:sy+py, sx:sx+px], (sz, sy, sx)

    def _crop_by_start(self, img: np.ndarray, seg: np.ndarray, start_zyx: Tuple[int, int, int]):
        pz, py, px = self.patch_size
        z0, y0, x0 = start_zyx
        return img[:, z0:z0+pz, y0:y0+py, x0:x0+px], seg[z0:z0+pz, y0:y0+py, x0:x0+px]

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        img, seg = self._load_case_np(case_id)

        # pad end so sliding/cropping never produces smaller-than-patch blocks
        img, seg = _pad_end_img_seg(img, seg, self.patch_size)

        # --------- build GT masks ---------
        gt_liver = (seg > 0).astype(np.uint8)
        gt_tumor = (seg == 2).astype(np.uint8)

        # --------- liver stage ---------
        if self.stage == "liver":
            # bbox source: GT or pred
            bbox_mask = gt_liver
            pred_liver = None
            if self.liver_use_bbox and self.liver_use_pred_bbox:
                pred_liver = self._load_pred_liver(case_id)
                if pred_liver is not None and pred_liver.shape == gt_liver.shape:
                    bbox_mask = pred_liver

            if self.liver_use_bbox:
                b = _bbox_from_mask(bbox_mask, margin=self.liver_bbox_margin)
                if b is None:
                    # fallback: full-volume crop
                    img_p, seg_p, start = self._random_crop_full(img, seg)
                else:
                    img_p, seg_p, start = self._random_crop_in_bbox(img, seg, b)
            else:
                img_p, seg_p, start = self._random_crop_full(img, seg)

            # optional: append pred prior channel (mask)
            if self.liver_add_pred_prior:
                if pred_liver is None:
                    pred_liver = self._load_pred_liver(case_id)
                if pred_liver is None or pred_liver.shape != gt_liver.shape:
                    prior = gt_liver
                else:
                    prior = pred_liver
                prior = _pad_end_vol(prior, self.patch_size)
                z0, y0, x0 = start
                pz, py, px = self.patch_size
                prior_p = prior[z0:z0+pz, y0:y0+py, x0:x0+px].astype(np.float32)[None, ...]
                img_p = np.concatenate([img_p, prior_p], axis=0)

            # augment (train only). liver: flips only
            if self.train:
                img_p, seg_p = self._augment_patch(img_p, seg_p, strong=False)

            target = (seg_p > 0).astype(np.int64)

            if self.return_sdf:
                sdf = self._compute_sdf(target.astype(np.uint8), clip=self.sdf_clip)
                sdf = sdf[None, ...].astype(np.float32)
            else:
                sdf = np.zeros((1,) + self.patch_size, dtype=np.float32)

            return (
                torch.from_numpy(img_p.copy()),
                torch.from_numpy(target.copy()),
                torch.from_numpy(sdf.copy()),
                case_id,
            )

        # --------- tumor stage ---------
        # bbox source mixing: pred or GT
        pred_liver = self._load_pred_liver(case_id) if self.tumor_use_pred_liver else None
        use_pred_bbox = False
        if self.tumor_use_pred_liver and pred_liver is not None and pred_liver.shape == gt_liver.shape:
            if np.random.rand() < np.clip(self.tumor_pred_bbox_ratio, 0.0, 1.0):
                use_pred_bbox = True

        bbox_mask = pred_liver if use_pred_bbox else gt_liver
        b = _bbox_from_mask(bbox_mask, margin=self.tumor_bbox_margin)
        if b is None:
            # fallback: full volume bbox
            _, Z, Y, X = img.shape
            b = (0, Z - 1, 0, Y - 1, 0, X - 1)
        b = _clamp_bbox(b, img.shape[1], img.shape[2], img.shape[3])
        z0, z1, y0, y1, x0, x1 = b

        # candidates in bbox
        roi_seg = seg[z0:z1+1, y0:y1+1, x0:x1+1]
        roi_liver = (roi_seg > 0)
        roi_tumor = (roi_seg == 2)

        r = float(np.random.rand())
        pos_r = max(0.0, self.tumor_pos_ratio)
        hn_r = max(0.0, self.tumor_hardneg_ratio)
        if pos_r + hn_r > 1.0:
            s = pos_r + hn_r
            pos_r /= s
            hn_r /= s

        mode = "random"
        if r < pos_r:
            mode = "pos"
        elif r < pos_r + hn_r:
            mode = "hardneg"

        pz, py, px = self.patch_size
        _, Z, Y, X = img.shape

        def pick_start_for_center(cz, cy, cx):
            rz = _start_range_for_center(cz, pz, z0, z1, Z)
            ry = _start_range_for_center(cy, py, y0, y1, Y)
            rx = _start_range_for_center(cx, px, x0, x1, X)
            if (rz is None) or (ry is None) or (rx is None):
                return None
            return (_randint(rz[0], rz[1]), _randint(ry[0], ry[1]), _randint(rx[0], rx[1]))

        start = None
        if mode == "pos":
            cand = np.where(roi_tumor)
            if cand[0].size > 0:
                k = int(np.random.randint(0, cand[0].size))
                cz, cy, cx = int(cand[0][k] + z0), int(cand[1][k] + y0), int(cand[2][k] + x0)
                start = pick_start_for_center(cz, cy, cx)

        if start is None and mode == "hardneg":
            cand = np.where(roi_liver & (~roi_tumor))
            if cand[0].size > 0:
                k = int(np.random.randint(0, cand[0].size))
                cz, cy, cx = int(cand[0][k] + z0), int(cand[1][k] + y0), int(cand[2][k] + x0)
                start = pick_start_for_center(cz, cy, cx)

        if start is None:
            # random crop inside bbox
            img_p, seg_p, start = self._random_crop_in_bbox(img, seg, b)
        else:
            img_p, seg_p = self._crop_by_start(img, seg, start)

        # optional: append liver prior channel
        if self.tumor_add_liver_prior:
            prior = None
            if self.tumor_prior_type == "prob":
                prior = self._load_pred_liver_prob(case_id)
                if prior is not None and prior.shape != gt_liver.shape:
                    prior = None
            if prior is None:
                if pred_liver is not None and pred_liver.shape == gt_liver.shape:
                    prior = pred_liver.astype(np.float32)
                else:
                    prior = gt_liver.astype(np.float32)

            prior = _pad_end_vol(prior, self.patch_size)
            zz, yy, xx = start
            prior_p = prior[zz:zz+pz, yy:yy+py, xx:xx+px][None, ...].astype(np.float32)
            img_p = np.concatenate([img_p, prior_p], axis=0)

        # augment (train only). tumor: flips + mild intensity/noise
        if self.train:
            img_p, seg_p = self._augment_patch(img_p, seg_p, strong=True)

        # tumor label: 2 -> 1, else 0
        target = (seg_p == 2).astype(np.int64)

        sdf = np.zeros((1,) + self.patch_size, dtype=np.float32)
        return (
            torch.from_numpy(img_p.copy()),
            torch.from_numpy(target.copy()),
            torch.from_numpy(sdf.copy()),
            case_id,
        )

    def _random_crop_full(self, img: np.ndarray, seg: np.ndarray):
        """Random crop anywhere in volume (after pad_end)."""
        _, Z, Y, X = img.shape
        pz, py, px = self.patch_size
        sz = int(np.random.randint(0, max(Z - pz + 1, 1)))
        sy = int(np.random.randint(0, max(Y - py + 1, 1)))
        sx = int(np.random.randint(0, max(X - px + 1, 1)))
        return img[:, sz:sz+pz, sy:sy+py, sx:sx+px], seg[sz:sz+pz, sy:sy+py, sx:sx+px], (sz, sy, sx)
