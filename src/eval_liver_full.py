#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对 liver_best.pth 在整幅 3D 体积上做推理和 Dice 评估。

用法示例：

  PREPROC_DIR=/home/my/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres

  # 只评估验证集（和训练时一样的划分）
  python eval_liver_full.py \
    --preproc_dir $PREPROC_DIR \
    --ckpt train_logs/lits_3dfullres_like/liver_best.pth \
    --out_dir eval_liver_full_val \
    --split val \
    --patch_size 128 128 128 \
    --stride 64 64 64 \
    --save_npy

  # 评估所有病例
  python eval_liver_full.py \
    --preproc_dir $PREPROC_DIR \
    --ckpt train_logs/lits_3dfullres_like/liver_best.pth \
    --out_dir eval_liver_full_all \
    --split all \
    --patch_size 128 128 128 \
    --stride 64 64 64
"""

import argparse
from pathlib import Path
from typing import Tuple, List

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.unet3D import UNet3D


# ----------------- 参数解析 ----------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preproc_dir",
        type=str,
        required=True,
        help="nnUNetv2 预处理目录，例如 Dataset003_Liver/nnUNetPlans_3d_fullres",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="liver 模型 checkpoint 路径，例如 train_logs/.../liver_best.pth",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="eval_liver_full",
        help="保存评估结果的目录",
    )
    p.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help="滑动窗口 patch 大小 (Z Y X)",
    )
    p.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="滑动窗口步长 (Z Y X)",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=["all", "train", "val"],
        default="val",
        help="评估全部病例还是训练/验证划分",
    )
    p.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="和训练时一致，用于 train/val 划分",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="train/val 划分所用随机种子（需和训练时一致）",
    )
    p.add_argument(
        "--max_cases",
        type=int,
        default=-1,
        help="最多评估多少个 case（-1 = 不限制）",
    )
    p.add_argument(
        "--save_npy",
        action="store_true",
        help="是否保存预测 mask & GT 为 .npy",
    )
    return p.parse_args()


# ----------------- 数据加载（.b2nd + .pkl） ----------------- #

def list_case_ids(preproc_dir: Path, split: str, train_ratio: float, seed: int) -> List[str]:
    """
    和 LITSDatasetB2ND 一致：用 .pkl 名字做 case_id 再划分 train/val
    """
    all_pkl = sorted(preproc_dir.glob("*.pkl"))
    case_ids = [p.stem for p in all_pkl]
    if len(case_ids) == 0:
        raise RuntimeError(f"No .pkl files found in {preproc_dir}")

    import random
    rng = random.Random(seed)
    rng.shuffle(case_ids)
    n_train = int(len(case_ids) * train_ratio)

    if split == "all":
        return case_ids
    elif split == "train":
        return case_ids[:n_train]
    else:  # "val"
        return case_ids[n_train:]


def load_case_b2nd(preproc_dir: Path, case_id: str):
    """
    仿照 LITSDatasetB2ND._load_case_np:
      - data_file: {case_id}.b2nd  -> img: (C, Z, Y, X)
      - seg_file:  {case_id}_seg.b2nd -> seg: (Z, Y, X) or (1, Z, Y, X)
    properties 目前不使用，但可以按需返回。
    """
    dparams = {"nthreads": 1}

    data_file = preproc_dir / f"{case_id}.b2nd"
    seg_file = preproc_dir / f"{case_id}_seg.b2nd"
    prop_file = preproc_dir / f"{case_id}.pkl"

    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    if not seg_file.is_file():
        raise FileNotFoundError(seg_file)

    data_b = blosc2.open(urlpath=str(data_file), mode="r", dparams=dparams)
    seg_b = blosc2.open(urlpath=str(seg_file), mode="r", dparams=dparams)

    img = data_b[:].astype(np.float32)  # (C, Z, Y, X)
    seg = seg_b[:].astype(np.int16)     # (1, Z, Y, X) or (Z, Y, X)

    if seg.ndim == 4:
        seg = seg[0]  # -> (Z, Y, X)

    # 如果以后要用 spacing 等信息，可以在这里加载 properties
    # import pickle as pkl
    # with open(prop_file, "rb") as f:
    #     properties = pkl.load(f)

    return img, seg


# ----------------- 滑动窗口推理 ----------------- #

def sliding_window_inference(
    volume: torch.Tensor,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int = 2,
) -> torch.Tensor:
    """
    对单个体积 (1, C, Z, Y, X) 做滑动窗口推理，返回：
      (num_classes, Z, Y, X) 的 prob map (softmax 后)。
    """
    model.eval()
    _, C, Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    prob_map = torch.zeros((num_classes, Z, Y, X), dtype=torch.float32, device=volume.device)
    weight_map = torch.zeros((1, Z, Y, X), dtype=torch.float32, device=volume.device)

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
                    z1 = z0 + pz
                    y1 = y0 + py
                    x1 = x0 + px

                    patch = volume[:, :, z0:z1, y0:y1, x0:x1]  # (1, C, pz, py, px)
                    logits = model(patch)                      # (1, num_classes, pz, py, px)
                    probs = F.softmax(logits, dim=1)[0]       # (num_classes, pz, py, px)

                    prob_map[:, z0:z1, y0:y1, x0:x1] += probs
                    weight_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    prob_map = prob_map / torch.clamp_min(weight_map, 1.0)
    return prob_map


# ----------------- Dice 计算 ----------------- #

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    pred, target: (Z, Y, X), 二值 {0,1}，只算前景 dice。
    """
    assert pred.shape == target.shape
    pred_fg = (pred > 0).astype(np.float32)
    tgt_fg = (target > 0).astype(np.float32)

    intersection = (pred_fg * tgt_fg).sum()
    union = pred_fg.sum() + tgt_fg.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


# ----------------- 主流程 ----------------- #

def main():
    args = parse_args()

    # 和你训练时一样，建议用单线程读取 blosc
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载 checkpoint，构建 UNet3D
    ckpt = torch.load(args.ckpt, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)

    print(f"=> Loading model from {args.ckpt}")
    print(f"   in_channels={in_channels}, num_classes={num_classes}")

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,        # 要和训练时保持一致
        dropout_p=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    # 2) 获取要评估的 case_id 列表（train/val/all）
    case_ids = list_case_ids(
        preproc_dir=preproc_dir,
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if args.max_cases > 0:
        case_ids = case_ids[: args.max_cases]

    print(f"Found {len(case_ids)} case(s) for split='{args.split}'.")

    dice_list: List[float] = []

    # 3) 逐个 case 做整幅推理 + Dice
    for case_id in tqdm(case_ids, desc="Evaluating", unit="case"):
        img_np, seg_np = load_case_b2nd(preproc_dir, case_id)   # img: (C,Z,Y,X), seg: (Z,Y,X)

        # liver 的 GT: seg>0 视为前景
        liver_gt = (seg_np > 0).astype(np.int16)

        # 转 torch，增加 batch 维
        vol = torch.from_numpy(img_np[None, ...]).to(device)  # (1, C, Z, Y, X)

        prob_map = sliding_window_inference(
            volume=vol,
            model=model,
            patch_size=patch_size,
            stride=stride,
            num_classes=num_classes,
        )  # (num_classes, Z, Y, X)

        pred_label = prob_map.argmax(dim=0).cpu().numpy().astype(np.int16)  # (Z,Y,X)

        dice = dice_score(pred_label, liver_gt)
        dice_list.append(dice)

        print(f"  Case {case_id}: Dice={dice:.4f}")

        if args.save_npy:
            np.save(out_dir / f"{case_id}_liver_pred.npy", pred_label)
            np.save(out_dir / f"{case_id}_liver_gt.npy", liver_gt)

    # 4) 汇总结果
    dice_arr = np.array(dice_list, dtype=np.float32)
    mean_dice = float(dice_arr.mean()) if len(dice_arr) > 0 else 0.0
    std_dice = float(dice_arr.std()) if len(dice_arr) > 0 else 0.0

    print("=" * 60)
    print(f"[Liver full-volume Dice] split='{args.split}', cases={len(dice_list)}")
    print(f"  Mean Dice = {mean_dice:.4f}, Std = {std_dice:.4f}")
    print("=" * 60)

    # 写入 txt
    with (out_dir / "liver_eval_results.txt").open("w") as f:
        for cid, d in zip(case_ids, dice_list):
            f.write(f"{cid}\t{d:.6f}\n")
        f.write(f"\nMean Dice = {mean_dice:.6f}, Std = {std_dice:.6f}\n")


if __name__ == "__main__":
    main()
