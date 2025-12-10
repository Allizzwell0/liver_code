# datasets/lits_datasets.py
import random
from pathlib import Path
from typing import Tuple

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset


class LITSDatasetB2ND(Dataset):
    """
    使用 nnUNetv2 预处理后的 LiTS (MSD Task03_Liver) 数据 (.b2nd + .pkl)

    - stage="liver":  背景 vs 肝脏(含肿瘤)，标签: 0 / 1
    - stage="tumor":  背景/肝 vs 肿瘤，      标签: 0 / 1
    - train_ratio: 简单按病例划分 train/val
    """

    def __init__(
        self,
        preproc_dir: str,
        stage: str = "liver",           # "liver" or "tumor"
        patch_size: Tuple[int, int, int] = (96, 160, 160),
        train: bool = True,
        train_ratio: float = 0.8,
        seed: int = 0,
    ):
        assert stage in ("liver", "tumor")
        self.stage = stage
        self.patch_size = patch_size
        self.preproc_dir = Path(preproc_dir)

        # 所有病例的 id：用 .pkl 名称作为 case_id
        all_pkl = sorted(self.preproc_dir.glob("*.pkl"))
        case_ids = [p.stem for p in all_pkl]
        if len(case_ids) == 0:
            raise RuntimeError(f"No .pkl files found in {preproc_dir}")

        # 按病例划分 train / val
        rng = random.Random(seed)
        rng.shuffle(case_ids)
        n_train = int(len(case_ids) * train_ratio)
        if train:
            self.case_ids = case_ids[:n_train]
        else:
            self.case_ids = case_ids[n_train:]

        # 建议用 1 线程读取 blosc（nnUNet 里也有类似设置）
        blosc2.set_nthreads(1)

        print(
            f"[LITSDatasetB2ND] stage={stage}, train={train}, "
            f"num_cases={len(self.case_ids)}, patch_size={patch_size}"
        )

    def __len__(self):
        return len(self.case_ids)

    def _load_case_np(self, case_id: str):
        """
        读取一个病例的 image / seg
        返回:
            img: float32, (C, Z, Y, X)
            seg: int16,   (Z, Y, X)
        """
        dparams = {"nthreads": 1}

        data_file = self.preproc_dir / f"{case_id}.b2nd"
        seg_file = self.preproc_dir / f"{case_id}_seg.b2nd"
        prop_file = self.preproc_dir / f"{case_id}.pkl"

        if not data_file.is_file():
            raise FileNotFoundError(data_file)
        if not seg_file.is_file():
            raise FileNotFoundError(seg_file)

        data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
        seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)

        img = data_b[:].astype(np.float32)  # (C, Z, Y, X)
        seg = seg_b[:].astype(np.int16)     # (1, Z, Y, X) or (Z, Y, X)

        if seg.ndim == 4:
            seg = seg[0]

        # properties 先读出来备用（这里暂不使用）
        import pickle as pkl
        with open(prop_file, "rb") as f:
            properties = pkl.load(f)

        return img, seg, properties

    @staticmethod
    def _random_crop_with_padding(
        img: np.ndarray,
        seg: np.ndarray,
        patch_size: Tuple[int, int, int],
    ):
        """
        简单随机裁剪 + 正向 padding：
        img: (C, Z, Y, X)
        seg: (Z, Y, X)
        """
        _, Z, Y, X = img.shape
        pz, py, px = patch_size

        pad_z = max(0, pz - Z)
        pad_y = max(0, py - Y)
        pad_x = max(0, px - X)

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            img = np.pad(img,
                         ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
                         mode="constant")
            seg = np.pad(seg,
                         ((0, pad_z), (0, pad_y), (0, pad_x)),
                         mode="constant")
            _, Z, Y, X = img.shape

        max_z = Z - pz
        max_y = Y - py
        max_x = X - px

        z0 = np.random.randint(0, max_z + 1) if max_z > 0 else 0
        y0 = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x0 = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        img_patch = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
        seg_patch = seg[z0:z0+pz, y0:y0+py, x0:x0+px]

        return img_patch, seg_patch

    def _remap_labels(self, seg: np.ndarray) -> np.ndarray:
        """
        MSD Task03_Liver (LiTS) 标签约定:
          0: 背景
          1: 肝脏
          2: 肿瘤

        转二分类:
          stage="liver":  0=背景, 1=肝脏(1或2)
          stage="tumor":  0=非肿瘤, 1=肿瘤(=2)
        """
        if self.stage == "liver":
            tgt = (seg >= 1).astype(np.int64)
        else:
            tgt = (seg == 2).astype(np.int64)
        return tgt

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        img, seg, _ = self._load_case_np(case_id)

        # 随机 3D patch
        img, seg = self._random_crop_with_padding(img, seg, self.patch_size)

        # 标签映射到 0/1
        tgt = self._remap_labels(seg)  # (Z, Y, X)

        img_t = torch.from_numpy(img)        # (C, Z, Y, X), float32
        tgt_t = torch.from_numpy(tgt).long() # (Z, Y, X),   int64

        return img_t, tgt_t, case_id
