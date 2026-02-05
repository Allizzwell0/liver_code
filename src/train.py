#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train 3D U-Net on nnUNetv2-preprocessed LiTS (.b2nd).

This repo supports a 3-stage pipeline:
  1) liver_coarse  (full volume, no priors/coords)
  2) liver_refine  (liver ROI, coords + sdf head + priors)
  3) tumor         (tumor ROI inside liver, optional pred-liver mix + liver prior)

Tumor improvements in this version:
  - optional CBAM/SE + AttentionGate on skips
  - optional deep supervision (aux logits)
  - focal-dice / focal-tversky loss
  - optional AMP for faster training
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

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
    """Focal CE + soft dice (multi-class)."""
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


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 1.33,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Focal Tversky (foreground) + CE (more stable for tiny tumors)."""
    ce = F.cross_entropy(logits, target)
    probs = torch.softmax(logits, dim=1)[:, 1]  # (B,Z,Y,X)
    tgt = (target == 1).float()
    dims = (1, 2, 3)
    tp = (probs * tgt).sum(dims)
    fp = (probs * (1 - tgt)).sum(dims)
    fn = ((1 - probs) * tgt).sum(dims)
    tversky = (tp + smooth) / (tp + alpha * fp + (1 - alpha) * fn + smooth)
    return ce + (1.0 - tversky).clamp_min(0.0).pow(gamma).mean()


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
    """Return logits, sdf(optional), aux_logits(optional)."""
    if isinstance(out, dict):
        return out["logits"], out.get("sdf", None), out.get("aux_logits", None)
    if isinstance(out, (tuple, list)):
        # (logits, sdf, aux?) is not used in this project, keep compatibility
        logits = out[0]
        sdf = out[1] if len(out) > 1 else None
        aux = out[2] if len(out) > 2 else None
        return logits, sdf, aux
    return out, None, None


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

    # optional: init with previous stage weights
    p.add_argument("--pretrained", type=str, default=None)

    # model switches
    p.add_argument("--base_filters", type=int, default=32)
    p.add_argument("--use_attn_gate", type=int, default=0, choices=[0, 1], help="AttentionGate on skip (recommended for tumor)")
    p.add_argument("--use_se", type=int, default=0, choices=[0, 1])
    p.add_argument("--use_cbam", type=int, default=0, choices=[0, 1], help="CBAM = channel+spatial attention (stronger but slower)")
    p.add_argument("--deep_supervision", type=int, default=0, choices=[0, 1], help="aux logits at decoder (tumor recommended)")

    # AMP (faster, lower memory)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])

    # --------- liver bbox sampling ---------
    p.add_argument("--liver_use_bbox", type=int, default=1, choices=[0, 1])
    p.add_argument("--liver_bbox_margin", type=int, default=16)

    # stage2: use stage1 predicted liver for bbox/prior (optional)
    p.add_argument("--liver_use_pred_bbox", type=int, default=0, choices=[0, 1])
    p.add_argument("--liver_pred_dir", type=str, default=None)
    p.add_argument("--liver_add_pred_prior", type=int, default=0, choices=[0, 1])

    # --------- tumor: mix pred liver bbox + GT liver bbox + optional liver prior ---------
    p.add_argument("--tumor_use_pred_liver", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_pred_liver_dir", type=str, default=None)
    p.add_argument("--tumor_pred_bbox_ratio", type=float, default=1.0,
                   help="bbox source ratio from pred liver (rest uses GT liver). e.g. 0.3~0.7")
    p.add_argument("--tumor_add_liver_prior", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_prior_type", type=str, default="mask", choices=["mask", "prob"])
    p.add_argument("--tumor_bbox_margin", type=int, default=24)

    # tumor sampling ratios
    p.add_argument("--tumor_pos_ratio", type=float, default=0.6)
    p.add_argument("--tumor_hardneg_ratio", type=float, default=0.3)

    # tumor loss
    p.add_argument("--tumor_alpha", type=float, default=0.75)
    p.add_argument("--tumor_gamma", type=float, default=1.33)
    p.add_argument("--tumor_loss", type=str, default="focal_tversky", choices=["focal_dice", "focal_tversky"])

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
        tumor_add_liver_prior=bool(args.tumor_add_liver_prior),
        tumor_prior_type=args.tumor_prior_type,
        tumor_bbox_margin=int(args.tumor_bbox_margin),
        tumor_pos_ratio=float(args.tumor_pos_ratio),
        tumor_hardneg_ratio=float(args.tumor_hardneg_ratio),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        timeout=120 if args.num_workers > 0 else 0
    )
    # tumor val is easy to get stuck by I/O/workers -> force single-process
    val_workers = 0 if args.stage == "tumor" else max(1, args.num_workers // 2)
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=val_workers, pin_memory=True,
        timeout=120 if val_workers > 0 else 0
    )

    # --------- Model ---------
    sample_imgs, _, _, _ = next(iter(train_loader))
    in_channels = int(sample_imgs.shape[1])
    num_classes = 2

    # coords/sdf head: only for liver refine (bbox ROI sampling)
    use_priors = (args.stage == "liver") and liver_use_bbox

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=int(args.base_filters),
        dropout_p=0.0,
        use_coords=use_priors,
        use_sdf_head=use_priors,
        use_attn_gate=bool(args.use_attn_gate),
        use_se=bool(args.use_se),
        use_cbam=bool(args.use_cbam),
        deep_supervision=bool(args.deep_supervision) if args.stage == "tumor" else False,
    ).to(device)

    if args.pretrained and (args.resume is None):
        load_pretrained(model, args.pretrained, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.9, patience=30, min_lr=1e-5
    )

    # AMP
    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
            "base_filters": int(args.base_filters),
            "use_coords": bool(use_priors),
            "use_sdf_head": bool(use_priors),
            "use_attn_gate": bool(args.use_attn_gate),
            "use_se": bool(args.use_se),
            "use_cbam": bool(args.use_cbam),
            "deep_supervision": bool(args.deep_supervision) if args.stage == "tumor" else False,
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
            "tumor_loss": str(args.tumor_loss),
        }

    def tumor_loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if args.tumor_loss == "focal_dice":
            return dice_focal_loss(logits, target, alpha=float(args.tumor_alpha), gamma=float(args.tumor_gamma))
        return focal_tversky_loss(logits, target, alpha=float(args.tumor_alpha), gamma=float(args.tumor_gamma))

    ds_w = (0.4, 0.2, 0.1)  # fixed deep supervision weights (y1,y2,y3)

    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f"\n[{args.stage}] ===== epoch {epoch:03d}/{args.epochs} =====")
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

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    out = model(imgs)
                    logits, sdf_pred, aux_logits = unpack_out(out)

                    if args.stage == "tumor":
                        loss = tumor_loss_fn(logits, targets)
                        if aux_logits is not None:
                            for w, a in zip(ds_w, aux_logits):
                                loss = loss + float(w) * tumor_loss_fn(a, targets)
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

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                scaler.step(optimizer)
                scaler.update()

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
                print(f"[{args.stage}] validating...", flush=True)
                for imgs, targets, sdf_gt, _ids in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    sdf_gt = sdf_gt.to(device, non_blocking=True)

                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        out = model(imgs)
                        logits, sdf_pred, aux_logits = unpack_out(out)

                        if args.stage == "tumor":
                            loss = tumor_loss_fn(logits, targets)
                            if aux_logits is not None:
                                for w, a in zip(ds_w, aux_logits):
                                    loss = loss + float(w) * tumor_loss_fn(a, targets)
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
        print(f"\n[Interrupted] saving checkpoint -> {interrupt_path}")
        try:
            save_ckpt(interrupt_path, pack_ckpt(current_epoch))
            print(f"[Interrupted] checkpoint saved: {interrupt_path}")
        except KeyboardInterrupt:
            print("[WARN] interrupted during checkpoint saving; skipped.")

    finally:
        writer.close()


if __name__ == "__main__":
    main()
