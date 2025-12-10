#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 nnUNetv2 预处理后的 LiTS (MSD Task03_Liver) 数据训练自建 3D UNet。

示例：
  # 肝脏分割
  python train.py \
    --preproc_dir /home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres \
    --stage liver

  # 肿瘤分割
  python train.py \
    --preproc_dir ... \
    --stage tumor
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.lits_datasets import LITSDatasetB2ND
from models.unet3D import UNet3D


def dice_ce_loss(logits, target, num_classes=2, smooth=1e-5):
    """
    组合 CrossEntropy + 全类 Dice loss
    logits: (B, C, Z, Y, X)
    target: (B, Z, Y, X)
    """
    ce = F.cross_entropy(logits, target)

    probs = F.softmax(logits, dim=1)  # (B, C, Z, Y, X)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target_oh, dims)
    union = torch.sum(probs + target_oh, dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    return ce + dice_loss


def foreground_dice(logits, target, smooth=1e-5):
    """
    只看前景（类别=1）的 Dice。
    logits: (B, 2, Z, Y, X)
    target: (B, Z, Y, X)
    """
    preds = torch.argmax(logits, dim=1)  # (B, Z, Y, X)
    pred_fg = (preds == 1).float()
    tgt_fg = (target == 1).float()

    intersection = torch.sum(pred_fg * tgt_fg)
    union = torch.sum(pred_fg) + torch.sum(tgt_fg)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preproc_dir",
        type=str,
        required=True,
        help="nnUNetv2 预处理目录，例如 Dataset003_Liver/nnUNetPlans_3d_fullres",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="liver",
        choices=["liver", "tumor"],
        help="liver: 肝脏分割; tumor: 肿瘤分割",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 160, 160])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="train_logs/lits_b2nd")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    patch_size: Tuple[int, int, int] = tuple(args.patch_size)

    # 1) Dataset & DataLoader
    train_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=True,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    val_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=False,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 2) 用一个 batch 推断 in_channels
    sample_imgs, _, _ = next(iter(train_loader))
    in_channels = sample_imgs.shape[1]
    num_classes = 2

    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base_filters=32)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0

    # 3) 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        n_train = 0

        for imgs, targets, _ids in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = dice_ce_loss(logits, targets, num_classes=num_classes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
            optimizer.step()

            train_loss += loss.item()
            train_dice += foreground_dice(logits.detach(), targets)
            n_train += 1

        train_loss /= max(1, n_train)
        train_dice /= max(1, n_train)

        # 验证
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        n_val = 0

        with torch.no_grad():
            for imgs, targets, _ids in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(imgs)

                loss = dice_ce_loss(logits, targets, num_classes=num_classes)
                val_loss += loss.item()
                val_dice += foreground_dice(logits, targets)
                n_val += 1

        val_loss /= max(1, n_val)
        val_dice /= max(1, n_val)

        print(
            f"[{args.stage}] Epoch {epoch:03d} | "
            f"Train loss={train_loss:.4f}, dice={train_dice:.4f} | "
            f"Val loss={val_loss:.4f}, dice={val_dice:.4f}"
        )

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            ckpt_path = save_dir / f"{args.stage}_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "stage": args.stage,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "patch_size": patch_size,
                },
                ckpt_path,
            )
            print(f"  >> New best val dice={best_val_dice:.4f}, saved to {ckpt_path}")


if __name__ == "__main__":
    main()
