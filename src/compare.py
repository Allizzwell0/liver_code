#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比评估：你的 UNet3D(liver_best.pth) vs TotalSegmentator(roi_subset liver)
在 LiTS 的 nnUNet 预处理数据（.b2nd + _seg.b2nd 或 gt_segmentations/*.nii.gz）上做全体积推理与 Dice 统计。

用法示例：

PREPROC_DIR=/home/my/data/liver_data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres
python compare.py \
  --preproc_dir $PREPROC_DIR \
  --ckpt train_logs/lits_3dfullres_like/liver_best.pth \
  --out_dir eval_compare_val \
  --split val \
  --patch_size 128 128 128 \
  --stride 64 64 64 \
  --save_npy \
  --run_totalseg \
  --totalseg_fast

可选：
- 若你想用 gt_segmentations/*.nii.gz 做 GT，而不是 _seg.b2nd：
    --gt_mode nii --gt_nii_dir /home/my/.../gt_segmentations
"""

import argparse
import csv
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import blosc2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import SimpleITK as sitk

from models.unet3D import UNet3D


# ----------------- 参数解析 ----------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True,
                   help="nnUNetv2 预处理目录，例如 .../Dataset003_Liver/nnUNetPlans_3d_fullres")
    p.add_argument("--ckpt", type=str, required=True,
                   help="你的 liver checkpoint 路径，例如 train_logs/.../liver_best.pth")
    p.add_argument("--out_dir", type=str, default="eval_compare_liver",
                   help="保存评估结果的目录")

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128],
                   help="滑动窗口 patch 大小 (Z Y X)")
    p.add_argument("--stride", type=int, nargs=3, default=[64, 64, 64],
                   help="滑动窗口步长 (Z Y X)")

    p.add_argument("--split", type=str, choices=["all", "train", "val"], default="val",
                   help="评估全部病例还是训练/验证划分")
    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="和训练时一致，用于 train/val 划分")
    p.add_argument("--seed", type=int, default=0,
                   help="train/val 划分所用随机种子（需和训练时一致）")
    p.add_argument("--max_cases", type=int, default=-1,
                   help="最多评估多少个 case（-1 = 不限制）")

    p.add_argument("--save_npy", action="store_true",
                   help="是否保存预测 mask & GT 为 .npy")
    p.add_argument("--save_nii", action="store_true",
                   help="是否保存预测 mask 为 .nii.gz（便于 3D Slicer 查看）")

    # GT 来源：b2nd（默认）或 gt_segmentations 的 nii.gz
    p.add_argument("--gt_mode", type=str, choices=["b2nd", "nii"], default="b2nd",
                   help="GT 来源：b2nd=使用 {case}_seg.b2nd；nii=使用 gt_segmentations/*.nii.gz")
    p.add_argument("--gt_nii_dir", type=str, default="",
                   help="当 --gt_mode nii 时，GT NIfTI 所在目录（例如 .../gt_segmentations）")

    # TotalSegmentator
    p.add_argument("--run_totalseg", action="store_true",
                   help="是否运行 TotalSegmentator 做对比")
    p.add_argument("--totalseg_bin", type=str, default="TotalSegmentator",
                   help="TotalSegmentator 可执行命令名（默认 TotalSegmentator）")
    p.add_argument("--totalseg_fast", action="store_true",
                   help="给 TotalSegmentator 加 --fast（更快但可能略降精度）")
    p.add_argument("--totalseg_robust_crop", action="store_true",
                   help="给 TotalSegmentator 加 --robust_crop（避免 ROI crop 裁断，较慢）")
    p.add_argument("--totalseg_body_seg", action="store_true",
                   help="给 TotalSegmentator 加 --body_seg（先裁剪 body 区域）")
    p.add_argument("--totalseg_device", type=str, choices=["auto", "cpu", "gpu"], default="auto",
                   help="如果你的 TotalSegmentator 版本支持 --device，可选 cpu/gpu；auto=不传")
    p.add_argument("--keep_totalseg_tmp", action="store_true",
                   help="保留每个 case 的 totalseg 临时目录（调试用）")

    return p.parse_args()


# ----------------- split / 列 case ----------------- #

def list_case_ids(preproc_dir: Path, split: str, train_ratio: float, seed: int) -> List[str]:
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
    else:
        return case_ids[n_train:]


# ----------------- 读取 b2nd / properties ----------------- #

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

    img = data_b[:].astype(np.float32)     # (C, Z, Y, X)
    seg = seg_b[:].astype(np.int16)        # (1, Z, Y, X) or (Z, Y, X)
    if seg.ndim == 4:
        seg = seg[0]
    return img, seg


def load_properties(preproc_dir: Path, case_id: str) -> Dict:
    pkl_file = preproc_dir / f"{case_id}.pkl"
    if not pkl_file.is_file():
        return {}
    try:
        with open(pkl_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


# ----------------- GT 读取（nii 模式） ----------------- #

def load_gt_from_nii(gt_nii_dir: Path, case_id: str) -> np.ndarray:
    # 常见：gt_segmentations/{case_id}.nii.gz
    cand = [
        gt_nii_dir / f"{case_id}.nii.gz",
        gt_nii_dir / f"{case_id}.nii",
    ]
    for p in cand:
        if p.is_file():
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z,Y,X)
            return arr
    raise FileNotFoundError(f"GT nii not found for case_id={case_id} in {gt_nii_dir}")


# ----------------- 你的模型：滑动窗口推理 ----------------- #

def sliding_window_inference(
    volume: torch.Tensor,
    model: torch.nn.Module,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int = 2,
) -> torch.Tensor:
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

                    patch = volume[:, :, z0:z1, y0:y1, x0:x1]      # (1,C,pz,py,px)
                    logits = model(patch)                          # (1,K,pz,py,px)
                    probs = F.softmax(logits, dim=1)[0]            # (K,pz,py,px)

                    prob_map[:, z0:z1, y0:y1, x0:x1] += probs
                    weight_map[:, z0:z1, y0:y1, x0:x1] += 1.0

    prob_map = prob_map / torch.clamp_min(weight_map, 1.0)
    return prob_map


# ----------------- Dice ----------------- #

def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    assert pred.shape == target.shape, (pred.shape, target.shape)
    pred_fg = (pred > 0).astype(np.float32)
    tgt_fg = (target > 0).astype(np.float32)
    inter = (pred_fg * tgt_fg).sum()
    union = pred_fg.sum() + tgt_fg.sum()
    return float((2.0 * inter + smooth) / (union + smooth))


# ----------------- TotalSegmentator（只分 liver） ----------------- #

def _guess_spacing_xyz(props: Dict) -> Optional[Tuple[float, float, float]]:
    """
    尝试从 properties 里取 spacing，并转成 SimpleITK 需要的 (x,y,z)。
    nnUNet 常用 zyx；这里做一个启发式转换。
    """
    spacing = None
    for k in ["spacing", "current_spacing", "spacing_after_resampling", "target_spacing"]:
        if k in props and props[k] is not None:
            spacing = props[k]
            break
    if spacing is None:
        return None
    try:
        s = tuple(float(x) for x in spacing)
        if len(s) != 3:
            return None
        # 启发式：若第一个最大，可能是 (z,y,x)
        if s[0] >= s[1] and s[0] >= s[2]:
            return (s[2], s[1], s[0])
        return (s[0], s[1], s[2])
    except Exception:
        return None


def write_ct_as_nifti_for_totalseg(ct_zyx: np.ndarray, out_path: Path, props: Dict):
    """
    输入 ct_zyx: (Z,Y,X) float32
    写出 nii.gz，尽量写 spacing（方向设为 identity，origin=0）
    """
    img = sitk.GetImageFromArray(ct_zyx.astype(np.float32))  # SITK expects z,y,x array
    sp = _guess_spacing_xyz(props)
    if sp is not None:
        img.SetSpacing(sp)  # (x,y,z)
    # 方向矩阵/原点对 totalseg 的内部 resample 有影响，但这里用 preproc 数据对比，简化为 identity
    dim = img.GetDimension()
    img.SetDirection(tuple([1.0 if i % (dim + 1) == 0 else 0.0 for i in range(dim * dim)]))
    img.SetOrigin(tuple([0.0] * dim))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def run_totalseg_liver(
    totalseg_bin: str,
    input_nii: Path,
    out_dir: Path,
    fast: bool,
    robust_crop: bool,
    body_seg: bool,
    device: str,
):
    cmd = [totalseg_bin, "-i", str(input_nii), "-o", str(out_dir), "--roi_subset", "liver"]
    if fast:
        cmd.append("--fast")
    if robust_crop:
        cmd.append("--robust_crop")
    if body_seg:
        cmd.append("--body_seg")
    # 不是所有版本都有 --device，auto 就不传；否则尝试传一下
    if device in ["cpu", "gpu"]:
        cmd += ["--device", device]

    subprocess.run(cmd, check=True)


def load_totalseg_liver_mask(out_dir: Path) -> np.ndarray:
    # 通常：out_dir/liver.nii.gz
    p = out_dir / "liver.nii.gz"
    if not p.is_file():
        # 兜底：找一个包含 liver 的 nii
        cand = sorted(out_dir.glob("*liver*.nii*"))
        if len(cand) == 1:
            p = cand[0]
        else:
            raise FileNotFoundError(f"Cannot find liver mask in {out_dir}")
    img = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(img).astype(np.int16)  # (Z,Y,X)
    return arr


def save_mask_as_nii(mask_zyx: np.ndarray, out_path: Path, ref_props: Dict):
    img = sitk.GetImageFromArray(mask_zyx.astype(np.uint8))
    sp = _guess_spacing_xyz(ref_props)
    if sp is not None:
        img.SetSpacing(sp)
    dim = img.GetDimension()
    img.SetDirection(tuple([1.0 if i % (dim + 1) == 0 else 0.0 for i in range(dim * dim)]))
    img.SetOrigin(tuple([0.0] * dim))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


# ----------------- 主流程 ----------------- #

def main():
    args = parse_args()
    blosc2.set_nthreads(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using torch device:", device)

    preproc_dir = Path(args.preproc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 加载你的模型
    ckpt = torch.load(args.ckpt, map_location=device)
    in_channels = ckpt.get("in_channels", 1)
    num_classes = ckpt.get("num_classes", 2)

    print(f"=> Loading model from {args.ckpt}")
    print(f"   in_channels={in_channels}, num_classes={num_classes}")

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    # 2) case 列表
    case_ids = list_case_ids(preproc_dir, args.split, args.train_ratio, args.seed)
    if args.max_cases > 0:
        case_ids = case_ids[:args.max_cases]
    print(f"Found {len(case_ids)} case(s) for split='{args.split}'.")

    # 3) GT nii dir（可选）
    gt_nii_dir = None
    if args.gt_mode == "nii":
        if args.gt_nii_dir:
            gt_nii_dir = Path(args.gt_nii_dir)
        else:
            # 自动猜：preproc_dir/../gt_segmentations
            guess = preproc_dir.parent / "gt_segmentations"
            gt_nii_dir = guess
        if not gt_nii_dir.is_dir():
            raise FileNotFoundError(f"--gt_mode nii but gt_nii_dir not found: {gt_nii_dir}")
        print(f"GT from NIfTI dir: {gt_nii_dir}")

    # 输出文件夹
    if args.save_npy:
        (out_dir / "npy").mkdir(exist_ok=True, parents=True)
    if args.save_nii:
        (out_dir / "nii_ours").mkdir(exist_ok=True, parents=True)
        (out_dir / "nii_totalseg").mkdir(exist_ok=True, parents=True)

    rows = []
    dice_ours_list = []
    dice_totalseg_list = []

    # 4) 逐 case 评估
    for case_id in tqdm(case_ids, desc="Evaluating", unit="case"):
        img_np, seg_np = load_case_b2nd(preproc_dir, case_id)   # img: (C,Z,Y,X), seg: (Z,Y,X)
        props = load_properties(preproc_dir, case_id)

        # GT
        if args.gt_mode == "b2nd":
            liver_gt = (seg_np > 0).astype(np.int16)
        else:
            gt_arr = load_gt_from_nii(gt_nii_dir, case_id)       # (Z,Y,X)
            liver_gt = (gt_arr > 0).astype(np.int16)

        # ---------- 你的模型 ----------
        vol = torch.from_numpy(img_np[None, ...]).to(device)    # (1,C,Z,Y,X)
        prob_map = sliding_window_inference(vol, model, patch_size, stride, num_classes=num_classes)
        pred_ours = prob_map.argmax(dim=0).cpu().numpy().astype(np.int16)  # (Z,Y,X)

        dice_ours = dice_score(pred_ours, liver_gt)
        dice_ours_list.append(dice_ours)

        # 保存（可选）
        if args.save_npy:
            np.save(out_dir / "npy" / f"{case_id}_ours_pred.npy", pred_ours)
            np.save(out_dir / "npy" / f"{case_id}_gt.npy", liver_gt)

        if args.save_nii:
            save_mask_as_nii(pred_ours, out_dir / "nii_ours" / f"{case_id}_liver_pred_ours.nii.gz", props)
            save_mask_as_nii(liver_gt, out_dir / "nii_ours" / f"{case_id}_liver_gt.nii.gz", props)

        # ---------- TotalSegmentator ----------
        dice_tot = float("nan")
        if args.run_totalseg:
            tmp_root = Path(tempfile.mkdtemp(prefix=f"totalseg_{case_id}_", dir=str(out_dir)))
            try:
                tmp_ct = tmp_root / "ct.nii.gz"
                tmp_out = tmp_root / "out"

                # 用 preproc 的 channel0 作为 totalseg 输入（保持同一输入空间）
                ct_zyx = img_np[0] if img_np.shape[0] >= 1 else img_np.squeeze(0)
                write_ct_as_nifti_for_totalseg(ct_zyx, tmp_ct, props)

                run_totalseg_liver(
                    totalseg_bin=args.totalseg_bin,
                    input_nii=tmp_ct,
                    out_dir=tmp_out,
                    fast=args.totalseg_fast,
                    robust_crop=args.totalseg_robust_crop,
                    body_seg=args.totalseg_body_seg,
                    device=args.totalseg_device,
                )

                pred_tot = load_totalseg_liver_mask(tmp_out)      # (Z,Y,X)
                # 有些情况下 shape 可能不一致（极少见），这里做严格检查
                if pred_tot.shape != liver_gt.shape:
                    raise RuntimeError(f"TotalSeg mask shape {pred_tot.shape} != GT shape {liver_gt.shape}")

                dice_tot = dice_score(pred_tot, liver_gt)
                dice_totalseg_list.append(dice_tot)

                if args.save_npy:
                    np.save(out_dir / "npy" / f"{case_id}_totalseg_pred.npy", pred_tot)

                if args.save_nii:
                    save_mask_as_nii(pred_tot, out_dir / "nii_totalseg" / f"{case_id}_liver_pred_totalseg.nii.gz", props)

            except Exception as e:
                print(f"[ERROR][TotalSeg] case={case_id}: {e}")
            finally:
                if args.keep_totalseg_tmp:
                    print(f"[KEEP] tmp at {tmp_root}")
                else:
                    shutil.rmtree(tmp_root, ignore_errors=True)

        rows.append({
            "case_id": case_id,
            "dice_ours": dice_ours,
            "dice_totalseg": dice_tot,
            "delta(ours-totalseg)": (dice_ours - dice_tot) if np.isfinite(dice_tot) else float("nan"),
        })

    # 5) 汇总
    def _mean_std(x: List[float]) -> Tuple[float, float]:
        if len(x) == 0:
            return 0.0, 0.0
        arr = np.array(x, dtype=np.float32)
        return float(arr.mean()), float(arr.std())

    ours_mean, ours_std = _mean_std(dice_ours_list)
    tot_mean, tot_std = _mean_std([d for d in dice_totalseg_list if np.isfinite(d)])

    summary = {
        "split": args.split,
        "num_cases": len(case_ids),
        "ours": {"mean": ours_mean, "std": ours_std},
        "totalseg": {"mean": tot_mean, "std": tot_std, "num_valid": int(np.isfinite(np.array(dice_totalseg_list)).sum())},
        "note": "TotalSegmentator here runs on preprocessed volume (channel0) by default. For best TotalSeg performance, use raw HU CT.",
    }

    print("=" * 70)
    print("[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 70)

    # 6) 写 CSV / TXT / JSON
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "dice_ours", "dice_totalseg", "delta(ours-totalseg)"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "results.txt", "w") as f:
        for r in rows:
            f.write(f"{r['case_id']}\t{r['dice_ours']:.6f}\t{r['dice_totalseg']}\t{r['delta(ours-totalseg)']}\n")
        f.write("\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote: {csv_path}")


if __name__ == "__main__":
    main()
