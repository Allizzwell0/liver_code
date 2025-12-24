# dataload/lits_datasets.py
import random
from pathlib import Path
from typing import Tuple, Optional

import blosc2
import numpy as np
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk



class LITSDatasetB2ND(Dataset):
    """
    使用 nnUNetv2 预处理后的 LiTS (MSD Task03_Liver) 数据 (.b2nd + .pkl)

    - stage="liver":  背景 vs 肝脏(含肿瘤)，标签: 0 / 1
    - stage="tumor":  背景/肝 vs 肿瘤，      标签: 0 / 1

    肿瘤阶段 (stage="tumor")：
        随机 patch 在『肝脏 ROI』内部采样，而不是整幅 volume 乱裁，
        提高肿瘤样本比例，训练更稳定。
    """

    def __init__(
        self,
        preproc_dir: str,
        stage: str = "liver",           # "liver" or "tumor"
        patch_size: Tuple[int, int, int] = (96, 160, 160),
        train: bool = True,
        train_ratio: float = 0.8,
        seed: int = 0,
        return_sdf: bool = False, 
        sdf_clip: float = 50.0, 
        sdf_margin: int = 0
    ):
        assert stage in ("liver", "tumor")
        self.stage = stage
        self.patch_size = patch_size
        self.preproc_dir = Path(preproc_dir)
        self.return_sdf = return_sdf
        self.sdf_clip = float(sdf_clip)
        self.sdf_margin = int(sdf_margin)

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

    # -------------------- 读一个病例 -------------------- #
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
    
    # -------------------- 计算 SDF -------------------- #
    @staticmethod
    def _compute_sdf_from_binary(mask_zyx: np.ndarray, clip: float = 50.0, margin: int = 0) -> np.ndarray:
        """
        mask_zyx: (Z,Y,X) 0/1，前景=1
        返回: sdf_zyx float32，范围约 [-1,1]，inside positive, outside negative
        """
        m = (mask_zyx > 0).astype(np.uint8)

        # 可选：加 margin，减少 patch 边缘截断影响
        if margin > 0:
            m_pad = np.pad(m, ((margin, margin), (margin, margin), (margin, margin)), mode="constant")
        else:
            m_pad = m

        img = sitk.GetImageFromArray(m_pad)  # SITK: array is (Z,Y,X)
        # insideIsPositive=True => inside 为正；useImageSpacing=False => 按 voxel 距离
        dist = sitk.SignedMaurerDistanceMap(
            img, insideIsPositive=True, squaredDistance=False, useImageSpacing=False
        )
        sdf = sitk.GetArrayFromImage(dist).astype(np.float32)

        # 去掉 margin
        if margin > 0:
            sdf = sdf[margin:-margin, margin:-margin, margin:-margin]

        # 截断并归一化到 [-1,1]
        if clip is not None and clip > 0:
            sdf = np.clip(sdf, -clip, clip) / clip

        return sdf


    # -------------------- 全图随机裁剪 -------------------- #
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
            img = np.pad(
                img,
                ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
            seg = np.pad(
                seg,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
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

    # -------------------- 肝脏 ROI 包围盒 -------------------- #
    @staticmethod
    def _get_liver_bbox(seg: np.ndarray, margin: int = 10) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        从 seg 中提取肝脏 (seg>=1) 的 3D 包围盒，并加上一点 margin。
        seg: (Z, Y, X)

        返回:
            z_min, z_max, y_min, y_max, x_min, x_max
        若没有肝脏（极少见），返回 None。
        """
        liver_mask = seg >= 1          # 肝脏 + 肿瘤
        if not liver_mask.any():
            return None

        zz, yy, xx = np.where(liver_mask)
        z_min, z_max = int(zz.min()), int(zz.max())
        y_min, y_max = int(yy.min()), int(yy.max())
        x_min, x_max = int(xx.min()), int(xx.max())

        # 加 margin 再裁剪到图像范围内
        z_min = max(0, z_min - margin)
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)

        Z, Y, X = seg.shape
        z_max = min(Z - 1, z_max + margin)
        y_max = min(Y - 1, y_max + margin)
        x_max = min(X - 1, x_max + margin)

        return z_min, z_max, y_min, y_max, x_min, x_max

    # -------------------- 肝脏 ROI 内随机裁剪 -------------------- #
    @staticmethod
    def _random_crop_in_roi(
        img: np.ndarray,
        seg: np.ndarray,
        patch_size: Tuple[int, int, int],
        roi_bbox: Tuple[int, int, int, int, int, int],
    ):
        """
        在给定的 ROI 包围盒内部随机裁剪 patch。
        img: (C, Z, Y, X)
        seg: (Z, Y, X)
        roi_bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
        """
        (z_min, z_max, y_min, y_max, x_min, x_max) = roi_bbox
        _, Z, Y, X = img.shape
        pz, py, px = patch_size

        # 1) 先看 ROI 自身是否比 patch 小，如是则整幅 pad
        roi_Z = z_max - z_min + 1
        roi_Y = y_max - y_min + 1
        roi_X = x_max - x_min + 1

        pad_z = max(0, pz - roi_Z)
        pad_y = max(0, py - roi_Y)
        pad_x = max(0, px - roi_X)

        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            img = np.pad(
                img,
                ((0, 0), (0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
            seg = np.pad(
                seg,
                ((0, pad_z), (0, pad_y), (0, pad_x)),
                mode="constant",
            )
            Z += pad_z
            Y += pad_y
            X += pad_x
            z_max += pad_z
            y_max += pad_y
            x_max += pad_x

        # 2) 在 ROI 内随机确定起点 (z0, y0, x0)
        #   保证 patch 完全落在 [z_min, z_max] 等范围内
        max_z0 = max(z_min, z_max - pz + 1)
        max_y0 = max(y_min, y_max - py + 1)
        max_x0 = max(x_min, x_max - px + 1)

        if max_z0 <= z_min:
            z0 = z_min
        else:
            z0 = np.random.randint(z_min, max_z0 + 1)

        if max_y0 <= y_min:
            y0 = y_min
        else:
            y0 = np.random.randint(y_min, max_y0 + 1)

        if max_x0 <= x_min:
            x0 = x_min
        else:
            x0 = np.random.randint(x_min, max_x0 + 1)

        img_patch = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
        seg_patch = seg[z0:z0+pz, y0:y0+py, x0:x0+px]

        return img_patch, seg_patch

    # -------------------- 标签重映射 -------------------- #
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

    # -------------------- 取一个样本 -------------------- #
    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        img, seg, _ = self._load_case_np(case_id)

        # === 关键逻辑：肿瘤阶段在肝脏 ROI 内采样 ===
        if self.stage == "tumor":
            bbox = self._get_liver_bbox(seg, margin=10)
            if bbox is not None:
                img, seg = self._random_crop_in_roi(img, seg, self.patch_size, bbox)
            else:
                # 极少数没有肝脏标签的情况，退回到全图随机裁剪
                img, seg = self._random_crop_with_padding(img, seg, self.patch_size)
        else:
            # liver 阶段：整幅 volume 上随机采 patch（原逻辑不变）
            img, seg = self._random_crop_with_padding(img, seg, self.patch_size)

        # 标签映射到 0/1
        tgt = self._remap_labels(seg)  # (Z, Y, X)

        img_t = torch.from_numpy(img)              # (C,Z,Y,X) float32
        tgt_t = torch.from_numpy(tgt).long()       # (Z,Y,X)   int64

        # ---- 新增：SDF ----
        if self.return_sdf and self.stage == "liver":
            sdf = self._compute_sdf_from_binary(tgt, clip=self.sdf_clip, margin=self.sdf_margin)  # (Z,Y,X)
        else:
            # tumor 或不开启 sdf：返回全 0，保证两任务接口一致
            sdf = np.zeros_like(tgt, dtype=np.float32)

        sdf_t = torch.from_numpy(sdf).float().unsqueeze(0)  # (1,Z,Y,X)

        return img_t, tgt_t, sdf_t, case_id

