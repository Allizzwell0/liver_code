#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练自建 3D UNet（nnUNetv2 预处理 LiTS .b2nd）：
- liver / tumor 两阶段（可做三阶段：liver_coarse -> liver_refine -> tumor）
- TensorBoard
- checkpoint 保存与 resume
- tumor：支持“部分用 pred liver bbox、部分用 GT liver bbox”混合裁剪 + 可追加 liver prior 通道
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataload.lits_datasets import LITSDatasetB2ND
from models.unet3D import UNet3D


# -------------------- losses -------------------- #

def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 2, smooth: float = 1e-5) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * target_oh, dims)
    union = torch.sum(probs + target_oh, dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return ce + (1.0 - dice.mean())


def dice_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1e-5) -> torch.Tensor:
    """
    tumor（极度类别不平衡）：focal(CE) + multi-class soft dice
    target: (B,Z,Y,X) with class {0,1}
    """
    ce_voxel = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce_voxel)
    focal = (alpha * (1.0 - pt).pow(gamma) * ce_voxel).mean()

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * target_oh, dims)
    union = torch.sum(probs + target_oh, dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return focal + (1.0 - dice.mean())


def foreground_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    preds = torch.argmax(logits, dim=1)
    pred_fg = (preds == 1).float()
    tgt_fg = (target == 1).float()
    inter = torch.sum(pred_fg * tgt_fg)
    union = torch.sum(pred_fg) + torch.sum(tgt_fg)
    return float((2.0 * inter + smooth) / (union + smooth))


# soft morphology for liver priors (encourage hole-free) ---- keep it minimal
def soft_dilate(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.max_pool3d(p, kernel_size=k, stride=1, padding=k // 2)


def soft_erode(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return -F.max_pool3d(-p, kernel_size=k, stride=1, padding=k // 2)


def soft_closing(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return soft_erode(soft_dilate(p, k=k), k=k)


def unpack_out(out):
    if isinstance(out, dict):
        return out["logits"], out.get("sdf", None)
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


# -------------------- utils -------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, ckpt: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def load_pretrained(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.is_file():
        print(f"[WARN] pretrained ckpt not found: {p}")
        return
    ckpt = torch.load(str(p), map_location=device)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"=> Loaded pretrained weights: {p}")
    if missing:
        print(f"   [pretrained] missing keys: {len(missing)}")
    if unexpected:
        print(f"   [pretrained] unexpected keys: {len(unexpected)}")


# -------------------- args -------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="liver", choices=["liver", "tumor"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="train_logs/lits_b2nd")
    p.add_argument("--resume", type=str, default=None)

    # 可选：用上一阶段权重初始化（不带 optimizer）
    p.add_argument("--pretrained", type=str, default=None)

    # --------- liver bbox 采样开关（0/1）---------
    p.add_argument("--liver_use_bbox", type=int, default=1, choices=[0, 1])
    p.add_argument("--liver_bbox_margin", type=int, default=16)

    # 二阶段用一阶段结果（pred liver）做 bbox / prior（可选）
    p.add_argument("--liver_use_pred_bbox", type=int, default=0, choices=[0, 1])
    p.add_argument("--liver_pred_dir", type=str, default=None)
    p.add_argument("--liver_add_pred_prior", type=int, default=0, choices=[0, 1])

    # --------- tumor: 使用 pred liver（部分）做 bbox + 追加 prior ---------
    p.add_argument("--tumor_use_pred_liver", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_pred_liver_dir", type=str, default=None)
    p.add_argument("--tumor_pred_bbox_ratio", type=float, default=1.0,
                   help="tumor 采样时 bbox 来源：pred liver 的比例（其余用 GT liver）。建议 0.3~0.7")
    p.add_argument("--tumor_pred_bbox_from", type=str, default="mask", choices=["mask", "prob", "union"],
                   help="当使用 pred liver bbox 时，bbox_mask 的来源：mask=pred_liver；prob=pred_liver_prob>thr；union=二者并集")
    p.add_argument("--tumor_pred_prob_thr", type=float, default=0.10,
                   help="tumor_pred_bbox_from=prob/union 时，使用 pred_liver_prob 的阈值")
    p.add_argument("--tumor_add_liver_prior", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_prior_type", type=str, default="mask", choices=["mask", "prob"])
    p.add_argument("--tumor_bbox_margin", type=int, default=24)

    # tumor patch 采样策略：pos / hardneg / random
    p.add_argument("--tumor_pos_ratio", type=float, default=0.6)
    p.add_argument("--tumor_hardneg_ratio", type=float, default=0.3)

    # tumor focal 超参
    p.add_argument("--tumor_alpha", type=float, default=0.75)
    p.add_argument("--tumor_gamma", type=float, default=2.0)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    patch_size: Tuple[int, int, int] = tuple(args.patch_size)
    liver_use_bbox = bool(args.liver_use_bbox)

    # --------- Dataset ---------
    train_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=True,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=(args.stage == "liver"),
        sdf_margin=16,

        liver_use_bbox=liver_use_bbox,
        liver_bbox_margin=args.liver_bbox_margin,
        liver_use_pred_bbox=bool(args.liver_use_pred_bbox),
        liver_pred_dir=args.liver_pred_dir,
        liver_add_pred_prior=bool(args.liver_add_pred_prior),

        tumor_use_pred_liver=bool(args.tumor_use_pred_liver),
        tumor_pred_liver_dir=args.tumor_pred_liver_dir,
        tumor_pred_bbox_ratio=float(args.tumor_pred_bbox_ratio),
            tumor_pred_bbox_from=str(args.tumor_pred_bbox_from),
            tumor_pred_prob_thr=float(args.tumor_pred_prob_thr),
        tumor_add_liver_prior=bool(args.tumor_add_liver_prior),
        tumor_prior_type=args.tumor_prior_type,
        tumor_bbox_margin=int(args.tumor_bbox_margin),
        tumor_pos_ratio=float(args.tumor_pos_ratio),
        tumor_hardneg_ratio=float(args.tumor_hardneg_ratio),
    )

    val_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=False,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=(args.stage == "liver"),
        sdf_margin=16,

        liver_use_bbox=liver_use_bbox,
        liver_bbox_margin=args.liver_bbox_margin,
        liver_use_pred_bbox=bool(args.liver_use_pred_bbox),
        liver_pred_dir=args.liver_pred_dir,
        liver_add_pred_prior=bool(args.liver_add_pred_prior),

        tumor_use_pred_liver=bool(args.tumor_use_pred_liver),
        tumor_pred_liver_dir=args.tumor_pred_liver_dir,
        tumor_pred_bbox_ratio=float(args.tumor_pred_bbox_ratio),
            tumor_pred_bbox_from=str(args.tumor_pred_bbox_from),
            tumor_pred_prob_thr=float(args.tumor_pred_prob_thr),
        tumor_add_liver_prior=bool(args.tumor_add_liver_prior),
        tumor_prior_type=args.tumor_prior_type,
        tumor_bbox_margin=int(args.tumor_bbox_margin),
        tumor_pos_ratio=float(args.tumor_pos_ratio),
        tumor_hardneg_ratio=float(args.tumor_hardneg_ratio),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(1, args.num_workers // 2), pin_memory=True
    )

    # --------- Model ---------
    sample_imgs, _, _, _ = next(iter(train_loader))
    in_channels = int(sample_imgs.shape[1])
    num_classes = 2

    # coords/sdf head：仅在 liver 且 bbox ROI 采样开启时启用（与你原脚本一致）
    use_priors = (args.stage == "liver") and liver_use_bbox

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=0.0,
        use_coords=use_priors,
        use_sdf_head=use_priors,
    ).to(device)

    if args.pretrained and (args.resume is None):
        load_pretrained(model, args.pretrained, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.9, patience=30, min_lr=1e-5
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = save_dir / "tb_logs" / args.stage
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[INFO] TensorBoard: {tb_dir}")

    best_val_dice = 0.0
    start_epoch = 1

    # --------- resume ---------
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            print(f"=> Resuming from: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"[WARN] optimizer_state load failed: {e}")
            best_val_dice = float(ckpt.get("best_val_dice", 0.0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"   start_epoch={start_epoch}, best_val_dice={best_val_dice:.4f}")
        else:
            print(f"[WARN] resume not found: {ckpt_path}")

    last_path = save_dir / f"{args.stage}_last.pth"
    best_path = save_dir / f"{args.stage}_best.pth"

    def pack_ckpt(epoch: int) -> dict:
        return {
            "epoch": int(epoch),
            "stage": args.stage,
            "best_val_dice": float(best_val_dice),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "in_channels": int(in_channels),
            "num_classes": int(num_classes),
            "patch_size": tuple(patch_size),
            "use_coords": bool(use_priors),
            "use_sdf_head": bool(use_priors),
            # liver
            "liver_use_bbox": bool(liver_use_bbox),
            "liver_bbox_margin": int(args.liver_bbox_margin),
            "liver_use_pred_bbox": bool(args.liver_use_pred_bbox),
            "liver_pred_dir": args.liver_pred_dir,
            "liver_add_pred_prior": bool(args.liver_add_pred_prior),
            # tumor
            "tumor_use_pred_liver": bool(args.tumor_use_pred_liver),
            "tumor_pred_liver_dir": args.tumor_pred_liver_dir,
            "tumor_pred_bbox_ratio": float(args.tumor_pred_bbox_ratio),
            "tumor_add_liver_prior": bool(args.tumor_add_liver_prior),
            "tumor_prior_type": args.tumor_prior_type,
            "tumor_bbox_margin": int(args.tumor_bbox_margin),
            "tumor_pos_ratio": float(args.tumor_pos_ratio),
            "tumor_hardneg_ratio": float(args.tumor_hardneg_ratio),
            "tumor_alpha": float(args.tumor_alpha),
            "tumor_gamma": float(args.tumor_gamma),
        }

    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch

            # --------- train ---------
            model.train()
            tr_loss = 0.0
            tr_dice = 0.0
            n_tr = 0

            for imgs, targets, sdf_gt, _ids in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                sdf_gt = sdf_gt.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(imgs)
                logits, sdf_pred = unpack_out(out)

                if args.stage == "tumor":
                    loss = dice_focal_loss(logits, targets, alpha=float(args.tumor_alpha), gamma=float(args.tumor_gamma))
                else:
                    loss = dice_ce_loss(logits, targets, num_classes=num_classes)
                    if use_priors:
                        prob = torch.softmax(logits, dim=1)[:, 1:2]  # (B,1,Z,Y,X)
                        prob_close = soft_closing(prob, k=7)
                        loss_hole = F.l1_loss(prob, prob_close)
                        loss_sdf = 0.0
                        if sdf_pred is not None:
                            loss_sdf = F.l1_loss(torch.tanh(sdf_pred), sdf_gt)
                        loss = loss + 0.3 * loss_hole + 0.8 * loss_sdf

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()

                tr_loss += float(loss.item())
                tr_dice += foreground_dice(logits.detach(), targets)
                n_tr += 1

            tr_loss /= max(1, n_tr)
            tr_dice /= max(1, n_tr)

            # --------- val ---------
            model.eval()
            va_loss = 0.0
            va_dice = 0.0
            n_va = 0
            with torch.no_grad():
                for imgs, targets, sdf_gt, _ids in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    sdf_gt = sdf_gt.to(device, non_blocking=True)

                    out = model(imgs)
                    logits, sdf_pred = unpack_out(out)

                    if args.stage == "tumor":
                        loss = dice_focal_loss(logits, targets, alpha=float(args.tumor_alpha), gamma=float(args.tumor_gamma))
                    else:
                        loss = dice_ce_loss(logits, targets, num_classes=num_classes)
                        if use_priors:
                            prob = torch.softmax(logits, dim=1)[:, 1:2]
                            prob_close = soft_closing(prob, k=7)
                            loss_hole = F.l1_loss(prob, prob_close)
                            loss_sdf = 0.0
                            if sdf_pred is not None:
                                loss_sdf = F.l1_loss(torch.tanh(sdf_pred), sdf_gt)
                            loss = loss + 0.3 * loss_hole + 0.8 * loss_sdf

                    va_loss += float(loss.item())
                    va_dice += foreground_dice(logits, targets)
                    n_va += 1

            va_loss /= max(1, n_va)
            va_dice /= max(1, n_va)

            scheduler.step(va_dice)

            print(
                f"[{args.stage}] epoch {epoch:03d}/{args.epochs} | "
                f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
                f"val loss {va_loss:.4f} dice {va_dice:.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.2e}"
            )

            writer.add_scalar(f"{args.stage}/loss_train", tr_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_train", tr_dice, epoch)
            writer.add_scalar(f"{args.stage}/loss_val", va_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_val", va_dice, epoch)
            writer.add_scalar(f"{args.stage}/lr", optimizer.param_groups[0]["lr"], epoch)

            # save last
            save_ckpt(last_path, pack_ckpt(epoch))

            # save best
            if va_dice > best_val_dice:
                best_val_dice = float(va_dice)
                ckpt_best = pack_ckpt(epoch)
                ckpt_best["best_val_dice"] = float(best_val_dice)
                save_ckpt(best_path, ckpt_best)
                print(f"  => new best_val_dice={best_val_dice:.4f} saved: {best_path}")

    except KeyboardInterrupt:
        interrupt_path = save_dir / f"{args.stage}_interrupt.pth"
        save_ckpt(interrupt_path, pack_ckpt(current_epoch))
        print(f"\n[Interrupted] checkpoint saved: {interrupt_path}")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
