#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 3D U-Net on nnUNetv2-preprocessed LiTS (.b2nd).

Recommended usage for LiTS -> unlabeled CT transfer (robust cascade):
1) Liver coarse (full-volume sampling, no priors/coords):
   python train.py --preproc_dir <LiTS_preproc> --stage liver --save_dir train_logs/liver_coarse --use_priors 0 --liver_use_bbox 0
2) Liver refine (liver ROI sampling, ROI-global coords + priors):
   python train.py --preproc_dir <LiTS_preproc> --stage liver --save_dir train_logs/liver_refine --use_priors 1 --liver_use_bbox 1

Tumor stage (optional):
   python train.py --preproc_dir <LiTS_preproc> --stage tumor --save_dir train_logs/tumor

This script saves:
  <save_dir>/<stage>_last.pth
  <save_dir>/<stage>_best.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

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
    For tumor (extreme imbalance): focal (binary) + dice on foreground.
    """
    probs = torch.softmax(logits, dim=1)[:, 1]  # (B,Z,Y,X)
    tgt = target.float()
    pt = probs * tgt + (1 - probs) * (1 - tgt)
    w = alpha * tgt + (1 - alpha) * (1 - tgt)
    focal = -w * (1 - pt).pow(gamma) * torch.log(pt.clamp_min(1e-6))
    focal = focal.mean()
    inter = (probs * tgt).sum()
    union = probs.sum() + tgt.sum()
    dice = (2 * inter + smooth) / (union + smooth)
    return focal + (1.0 - dice)


def foreground_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    pred = torch.argmax(logits, dim=1)
    pred = (pred > 0).float()
    tgt = (target > 0).float()
    inter = (pred * tgt).sum().item()
    union = pred.sum().item() + tgt.sum().item()
    return float((2.0 * inter + smooth) / (union + smooth))


# soft morphology for liver priors (encourage hole-free)
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
        return out[0], out[1] if len(out) > 1 else None
    return out, None


# -------------------- utils -------------------- #

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
              best_val_dice: float, meta: dict):
    ckpt = {
        "epoch": int(epoch),
        "best_val_dice": float(best_val_dice),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        **meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


# -------------------- args -------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--stage", type=str, choices=["liver", "tumor"], default="liver")
    p.add_argument("--save_dir", type=str, default="train_logs/lits")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.8)

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])

    # liver-specific switches
    p.add_argument("--use_priors", type=int, default=1, help="liver priors: ROI-global coords + sdf head + soft closing loss")
    p.add_argument("--liver_use_bbox", type=int, default=0, help="1: sample liver patches inside GT liver bbox (refine); 0: full-volume sampling (coarse)")
    p.add_argument("--liver_bbox_margin", type=int, default=16)

    # model tweak
    p.add_argument("--use_se", type=int, default=0)
    p.add_argument("--dropout_p", type=float, default=0.0)

    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "runs"))

    patch_size = tuple(args.patch_size)

    use_priors = bool(args.use_priors) and (args.stage == "liver")
    add_coords = use_priors  # coords are generated by dataset/infer, not by model

    train_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=True,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=use_priors,
        sdf_clip=20.0,
        sdf_margin=0,
        liver_use_bbox=bool(args.liver_use_bbox) if args.stage == "liver" else False,
        liver_bbox_margin=args.liver_bbox_margin,
        add_coords=add_coords if args.stage == "liver" else False,
    )
    val_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=False,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=use_priors,
        sdf_clip=20.0,
        sdf_margin=0,
        liver_use_bbox=bool(args.liver_use_bbox) if args.stage == "liver" else False,
        liver_bbox_margin=args.liver_bbox_margin,
        add_coords=add_coords if args.stage == "liver" else False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=max(1, args.num_workers // 2), pin_memory=True)

    sample_imgs, _, _, _ = next(iter(train_loader))
    in_channels = int(sample_imgs.shape[1])
    num_classes = 2

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=float(args.dropout_p),
        use_sdf_head=use_priors,   # liver can choose; tumor always False
        use_se=bool(args.use_se),
        use_coords=False,          # coords handled externally
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    start_epoch = 1
    best_val_dice = 0.0

    # resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            print(f"=> Resuming from {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print("[WARN] optimizer_state load failed:", e)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_dice = float(ckpt.get("best_val_dice", 0.0))
            print(f"   start_epoch={start_epoch}, best_val_dice={best_val_dice:.4f}")

    meta = {
        "stage": args.stage,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "use_priors": bool(use_priors),
        "add_coords": bool(add_coords),
        "use_sdf_head": bool(use_priors),
        "liver_use_bbox": bool(args.liver_use_bbox) if args.stage == "liver" else False,
        "liver_bbox_margin": int(args.liver_bbox_margin),
        "use_se": bool(args.use_se),
        "dropout_p": float(args.dropout_p),
    }

    last_path = save_dir / f"{args.stage}_last.pth"
    best_path = save_dir / f"{args.stage}_best.pth"

    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            current_epoch = epoch
            model.train()
            train_loss = 0.0
            train_dice = 0.0
            n_train = 0

            for imgs, targets, sdf_gt, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                sdf_gt = sdf_gt.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(imgs)
                logits, sdf_pred = unpack_out(out)

                if args.stage == "tumor":
                    loss = dice_focal_loss(logits, targets)
                else:
                    loss = dice_ce_loss(logits, targets, num_classes=num_classes)
                    if use_priors:
                        # 1) soft closing consistency (hole-free prior)
                        prob = torch.softmax(logits, dim=1)[:, 1:2]  # (B,1,Z,Y,X)
                        prob_close = soft_closing(prob, k=7)
                        loss_hole = F.l1_loss(prob, prob_close)

                        # 2) SDF supervision
                        loss_sdf = 0.0
                        if sdf_pred is not None:
                            loss_sdf = F.l1_loss(torch.tanh(sdf_pred), sdf_gt)

                        loss = loss + 0.3 * loss_hole + 0.8 * loss_sdf

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()

                train_loss += float(loss.item())
                train_dice += foreground_dice(logits.detach(), targets)
                n_train += 1

            train_loss /= max(1, n_train)
            train_dice /= max(1, n_train)

            # val
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            n_val = 0
            with torch.no_grad():
                for imgs, targets, sdf_gt, _ in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    sdf_gt = sdf_gt.to(device, non_blocking=True)

                    out = model(imgs)
                    logits, sdf_pred = unpack_out(out)

                    if args.stage == "tumor":
                        loss = dice_focal_loss(logits, targets)
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

                    val_loss += float(loss.item())
                    val_dice += foreground_dice(logits, targets)
                    n_val += 1

            val_loss /= max(1, n_val)
            val_dice /= max(1, n_val)

            # log
            writer.add_scalar(f"{args.stage}/loss_train", train_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_train", train_dice, epoch)
            writer.add_scalar(f"{args.stage}/loss_val", val_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_val", val_dice, epoch)

            print(
                f"[{args.stage}] epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
                f"val loss {val_loss:.4f} dice {val_dice:.4f}"
            )

            # save last
            save_ckpt(last_path, model, optimizer, epoch, best_val_dice, meta)

            # save best
            if val_dice > best_val_dice:
                best_val_dice = float(val_dice)
                save_ckpt(best_path, model, optimizer, epoch, best_val_dice, meta)
                print(f"  => new best_val_dice={best_val_dice:.4f} saved: {best_path}")

    except KeyboardInterrupt:
        interrupt_path = save_dir / f"{args.stage}_interrupt.pth"
        save_ckpt(interrupt_path, model, optimizer, current_epoch, best_val_dice, meta)
        print(f"\n[Interrupted] checkpoint saved: {interrupt_path}")

    writer.close()


if __name__ == "__main__":
    main()
