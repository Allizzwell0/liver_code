#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 nnUNet 预处理得到的 .b2nd CT + seg.npy 生成叠加彩色 GIF。

示例：
 python make_seg_gif.py \
    --img_b2nd /home/my/data/liver_data/self_data/prepocessed_data/my_case001.b2nd \
    --seg_npy  /home/my/data/liver_data/eval_outputs/my_data/my_case001_pred_3class.npy \
    --out_gif  /home/my/data/liver_data/eval_outputs/gif/my_case_001_z.gif \
    --axis z --fps 8
"""

import argparse
from pathlib import Path

import blosc2
import imageio.v2 as imageio
import numpy as np


def load_ct_from_b2nd(img_b2nd_path: Path) -> np.ndarray:
    """
    读取 nnUNet 预处理后的 CT:
      返回 ct_vol: (Z, Y, X), float32
    """
    dparams = {"nthreads": 1}
    arr = blosc2.open(urlpath=str(img_b2nd_path), mode="r", dparams=dparams)[:]
    print(f"[INFO] raw img_b2nd shape={arr.shape}, dtype={arr.dtype}")

    # nnUNet 格式：通常是 (C, Z, Y, X)
    if arr.ndim == 4:
        # 默认只用第 0 个通道（LiTS 即单通道 CT）
        ct = arr[0].astype(np.float32)
    elif arr.ndim == 3:
        ct = arr.astype(np.float32)
    else:
        raise ValueError(f"Unexpected img array shape: {arr.shape}")

    print(f"[INFO] CT volume shape={ct.shape}")
    return ct


def load_seg_from_npy(seg_path: Path) -> np.ndarray:
    """
    读取 seg.npy，并整理成 (Z, Y, X) 的 int16 标签体积。
    支持：
      - (Z, Y, X)
      - (1, Z, Y, X)
      - (C, Z, Y, X) -> argmax(C)
    """
    arr = np.load(seg_path)
    print(f"[INFO] raw seg shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim == 3:
        seg = arr
    elif arr.ndim == 4:
        if arr.shape[0] == 1:
            seg = arr[0]
        else:
            seg = np.argmax(arr, axis=0)
    else:
        raise ValueError(f"Unexpected seg array shape: {arr.shape}")

    seg = seg.astype(np.int16)
    print(f"[INFO] parsed seg shape={seg.shape}, unique={np.unique(seg)}")
    return seg


def normalize_ct_for_display(ct: np.ndarray) -> np.ndarray:
    """
    把 nnUNet 预处理后的 CT 做一个简单可视化归一化：
      - 用 [1%, 99%] 分位数做裁剪
      - 映射到 [0, 255] uint8 灰度
    """
    vmin, vmax = np.percentile(ct, [1, 99])
    print(f"[INFO] CT intensity window: [{vmin:.2f}, {vmax:.2f}] (1%~99%)")

    ct_clip = np.clip(ct, vmin, vmax)
    ct_norm = (ct_clip - vmin) / (vmax - vmin + 1e-6)
    ct_gray = (ct_norm * 255.0).astype(np.uint8)
    return ct_gray


def make_overlay_frames(
    ct_gray: np.ndarray,  # (Z, Y, X), uint8
    seg: np.ndarray,      # (Z, Y, X), int
    axis: str = "z",
    alpha: float = 0.6,
):
    """
    生成叠加 CT+seg 的彩色帧列表。

    label 颜色可根据需要调整：
      0: 背景（不覆盖）
      1: 肝脏（红色）
      2: 肿瘤（绿色）
    """
    # label -> RGB
    label_colors = {
        1: np.array([255, 0, 0], dtype=np.uint8),   # 肝脏：红色
        2: np.array([0, 255, 0], dtype=np.uint8),   # 肿瘤：绿色
        # 如果有别的结构，继续往下加
        # 3: np.array([0, 0, 255], dtype=np.uint8),
    }

    frames = []

    # 选择切片方向
    if axis == "z":
        num_slices = ct_gray.shape[0]
        ct_indexer = lambda i: ct_gray[i]        # (Y, X)
        seg_indexer = lambda i: seg[i]
    elif axis == "y":
        num_slices = ct_gray.shape[1]
        ct_indexer = lambda i: ct_gray[:, i, :]  # (Z, X)
        seg_indexer = lambda i: seg[:, i, :]
    elif axis == "x":
        num_slices = ct_gray.shape[2]
        ct_indexer = lambda i: ct_gray[:, :, i]  # (Z, Y)
        seg_indexer = lambda i: seg[:, :, i]
    else:
        raise ValueError("axis must be one of ['z', 'y', 'x']")

    for i in range(num_slices):
        ct_slice = ct_indexer(i)   # 2D
        seg_slice = seg_indexer(i) # 2D

        h, w = ct_slice.shape
        # 灰度转 3 通道
        base_rgb = np.stack([ct_slice] * 3, axis=-1)  # (H, W, 3), uint8

        overlay = base_rgb.copy()

        # 对每个 label 做 alpha 融合
        for label, color in label_colors.items():
            mask = (seg_slice == label)
            if not np.any(mask):
                continue

            # 取出对应位置像素
            overlay_label = overlay[mask].astype(np.float32)
            color_f = color.astype(np.float32)

            # alpha blend: out = (1-a)*gray + a*color
            blended = (1 - alpha) * overlay_label + alpha * color_f
            overlay[mask] = blended.astype(np.uint8)

        frames.append(overlay)

    return frames


def save_gif(frames, out_path: Path, fps: int = 8):
    duration = 1.0 / max(1, fps)
    imageio.mimsave(out_path, frames, duration=duration)
    print(f"[INFO] GIF saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_b2nd", type=str, required=True, help="预处理后的 CT .b2nd")
    parser.add_argument("--seg_npy", type=str, required=True, help="推理得到的 seg.npy")
    parser.add_argument("--out_gif", type=str, required=True, help="输出 GIF 路径")
    parser.add_argument(
        "--axis",
        type=str,
        default="z",
        choices=["z", "y", "x"],
        help="沿哪个轴做动画（默认 z）"
    )
    parser.add_argument("--fps", type=int, default=8, help="GIF 帧率（默认 8）")
    parser.add_argument("--alpha", type=float, default=0.6, help="覆盖透明度 (0~1)")
    return parser.parse_args()


def main():
    args = parse_args()
    img_b2nd_path = Path(args.img_b2nd)
    seg_npy_path = Path(args.seg_npy)
    out_path = Path(args.out_gif)

    ct = load_ct_from_b2nd(img_b2nd_path)      # (Z, Y, X), float32
    seg = load_seg_from_npy(seg_npy_path)      # (Z, Y, X), int

    # 简单检查一下几何形状是否匹配
    if ct.shape != seg.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape} vs seg {seg.shape}")

    ct_gray = normalize_ct_for_display(ct)     # (Z, Y, X), uint8
    frames = make_overlay_frames(
        ct_gray,
        seg,
        axis=args.axis,
        alpha=args.alpha,
    )
    save_gif(frames, out_path, fps=args.fps)


if __name__ == "__main__":
    main()
