#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 nnUNetv2 预处理后的 LiTS (MSD Task03_Liver) 数据训练自建 3D UNet。

功能：
  - CrossEntropy + 全类 Dice 的组合 loss
  - 训练 / 验证前景 Dice 指标
  - 每个 epoch 自动保存:
      <stage>_last.pth   : 最近一次 epoch 的 checkpoint（用于继续训练）
      <stage>_best.pth   : 验证 Dice 最好的 checkpoint（用于最终推理）
  - 支持 --resume 从 checkpoint 继续训练
  - 支持 Ctrl+C 中断时保存 <stage>_interrupt.pth
  - 使用 TensorBoard 记录 loss/dice 曲线，实时查看训练过程

示例：
  # 肝脏分割从头训练
  python train.py \
    --preproc_dir /home/mayue/WorldModel/Dreamer/liver/data/nnUNet_data/preprocessed/Dataset003_Liver/nnUNetPlans_3d_fullres \
    --stage liver

  # 肿瘤分割从头训练
  python train.py \
    --preproc_dir ... \
    --stage tumor

  # 从上一次训练的 last checkpoint 继续训练（推荐）
  python train.py \
    --preproc_dir ... \
    --stage liver \
    --resume train_logs/lits_b2nd/liver_last.pth
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataload.lits_datasets import LITSDatasetB2ND
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

def dice_focal_loss(logits, target, alpha=0.75, gamma=2.0, smooth=1e-5):
    """
    Focal + Dice，用于肿瘤分割（类别极不均衡时更稳）。
    logits: (B, C, Z, Y, X), C=2
    target: (B, Z, Y, X)，值为 0/1
    """
    # ----- Focal part -----
    # 普通 CE，先不做 mean，保留每个 voxel 的 loss
    ce_voxel = F.cross_entropy(logits, target, reduction="none")  # (B,Z,Y,X)
    pt = torch.exp(-ce_voxel)          # pt 越小，说明越难分
    focal = (alpha * (1 - pt) ** gamma * ce_voxel).mean()

    # ----- Dice part（和原来的一样） -----
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = torch.sum(probs * target_oh, dims)
    union = torch.sum(probs + target_oh, dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    return focal + dice_loss


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


def soft_dilate(p, k=5):
    return F.max_pool3d(p, kernel_size=k, stride=1, padding=k//2)

def soft_erode(p, k=5):
    return -F.max_pool3d(-p, kernel_size=k, stride=1, padding=k//2)

def soft_closing(p, k=5):
    return soft_erode(soft_dilate(p, k=k), k=k)

def unpack_out(out):
    """
    现在的 model 输出：dict {"logits":..., "sdf":...optional}
    """
    if isinstance(out, dict):
        return out["logits"], out.get("sdf", None)
    elif isinstance(out, (tuple, list)):
        logits = out[0]
        sdf = out[1] if len(out) > 1 else None
        return logits, sdf
    else:
        return out, None



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
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="train_logs/lits_b2nd")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从已有 checkpoint 恢复训练，例如 train_logs/lits_b2nd/liver_last.pth",
    )
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
        return_sdf=(args.stage=="liver"), 
        sdf_margin=16
    )
    val_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=False,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=(args.stage=="liver"), 
        sdf_margin=16
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
    sample_imgs, _, _, _ = next(iter(train_loader))
    in_channels = sample_imgs.shape[1]
    num_classes = 2

    use_priors = (args.stage == "liver")
    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=use_priors,
        use_sdf_head=use_priors,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",      
    factor=0.9,      # 每次降低到 0.9 倍
    patience=30,     # 连续 30 个 epoch 没有提升就降 lr
    min_lr=1e-5,     # 不要小于 1e-5
    # verbose=True,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard 日志目录
    tb_dir = save_dir / "tb_logs" / args.stage
    tb_dir.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    start_epoch = 1
    writer: SummaryWriter | None = None

    # 2.5) 可选：从 checkpoint 恢复
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            print(f"=> Resuming training from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"[WARN] Failed to load optimizer_state from ckpt: {e}")

            best_val_dice = float(ckpt.get("best_val_dice", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1

            ckpt_in_ch = ckpt.get("in_channels", None)
            ckpt_patch = tuple(ckpt.get("patch_size", patch_size))
            if ckpt_in_ch is not None and ckpt_in_ch != in_channels:
                print(f"[WARN] in_channels mismatch: ckpt={ckpt_in_ch}, current={in_channels}")
            if ckpt_patch != patch_size:
                print(f"[WARN] patch_size mismatch: ckpt={ckpt_patch}, current={patch_size}")

            print(
                f"   start_epoch={start_epoch}, "
                f"best_val_dice={best_val_dice:.4f}"
            )
        else:
            print(f"[WARN] --resume 指定的文件不存在: {ckpt_path}，将从头训练。")

    current_epoch = start_epoch - 1

    try:
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"[INFO] TensorBoard logging to: {tb_dir}")

        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            n_train = 0

            for imgs, targets, sdf_gt, _ids in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                sdf_gt = sdf_gt.to(device, non_blocking=True)

                optimizer.zero_grad()
                out = model(imgs)
                logits, sdf_pred = unpack_out(out)
                if args.stage == "tumor":
                    loss = dice_focal_loss(logits, targets)
                else:
                    loss = dice_ce_loss(logits, targets, num_classes=num_classes)

                    # ---- 新增：liver 才加先验约束 ----
                    # 1) 软 closing 约束
                    prob = torch.softmax(logits, dim=1)[:, 1:2]     # (B,1,Z,Y,X)
                    prob_close = soft_closing(prob, k=5)
                    loss_hole = F.l1_loss(prob, prob_close)

                    # 2) SDF 监督
                    loss_sdf = 0.0
                    if sdf_pred is not None:
                        loss_sdf = F.l1_loss(torch.tanh(sdf_pred), sdf_gt)

                    loss = loss + 0.05 * loss_hole + 0.2 * loss_sdf
                    # loss = loss + 0.2 * loss_sdf


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
                for imgs, targets, sdf_gt, _ids in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    sdf_gt = sdf_gt.to(device, non_blocking=True)
                    out = model(imgs)
                    logits, sdf_pred = unpack_out(out)

                    if args.stage == "tumor":
                        loss = dice_focal_loss(logits, targets)
                    else:
                        loss = dice_ce_loss(logits, targets, num_classes=num_classes)
                        # val 阶段一般不必加 sdf/hole 正则
                        # 也可以加，但会让 val_loss 不再可比
                    val_loss += loss.item()
                    val_dice += foreground_dice(logits, targets)
                    n_val += 1


            val_loss /= max(1, n_val)
            val_dice /= max(1, n_val)
            # scheduler.step(val_dice)
            print(
                f"[{args.stage}] Epoch {epoch:03d} | "
                f"Train loss={train_loss:.4f}, dice={train_dice:.4f} | "
                f"Val loss={val_loss:.4f}, dice={val_dice:.4f}"
            )
            scheduler.step(val_dice)

            # === 写入 TensorBoard 标量 ===
            if writer is not None:
                writer.add_scalar(f"{args.stage}/Loss/train", train_loss, epoch)
                writer.add_scalar(f"{args.stage}/Loss/val",   val_loss,   epoch)
                writer.add_scalar(f"{args.stage}/Dice/train", train_dice, epoch)
                writer.add_scalar(f"{args.stage}/Dice/val",   val_dice,   epoch)
                writer.add_scalar(f"{args.stage}/LR", optimizer.param_groups[0]["lr"], epoch)

            # 统一 checkpoint 字典
            ckpt_common = {
                "epoch": epoch,
                "stage": args.stage,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "in_channels": in_channels,
                "num_classes": num_classes,
                "patch_size": patch_size,
                "use_coords": use_priors,
                "use_sdf_head": use_priors,

            }

            # 每个 epoch 保存最近的 last checkpoint
            last_ckpt_path = save_dir / f"{args.stage}_last.pth"
            torch.save(ckpt_common, last_ckpt_path)

            # 如有更好 val_dice，则更新 best checkpoint
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                ckpt_common["best_val_dice"] = best_val_dice
                best_ckpt_path = save_dir / f"{args.stage}_best.pth"
                torch.save(ckpt_common, best_ckpt_path)
                print(f"  >> New best val dice={best_val_dice:.4f}, saved to {best_ckpt_path}")

    except KeyboardInterrupt:
        print("\n[INFO] 捕获到 KeyboardInterrupt，中断训练。保存当前 checkpoint...")
        interrupt_ckpt = {
            "epoch": current_epoch,
            "stage": args.stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "patch_size": patch_size,
            "use_coords": use_priors,
            "use_sdf_head": use_priors,

        }
        interrupt_path = save_dir / f"{args.stage}_interrupt.pth"
        torch.save(interrupt_ckpt, interrupt_path)
        print(f"  >> Interrupt checkpoint saved to {interrupt_path}")

    finally:
        if writer is not None:
            writer.close()
            print("[INFO] TensorBoard writer closed.")


if __name__ == "__main__":
    main()
