#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train LiTS 3D segmentation models on nnUNetv2-preprocessed (.b2nd).

Supports:
- liver (coarse / refine)
- tumor (cropped by GT or predicted liver, optional liver prior channel)
- checkpoints + resume + optional pretrained init
- TensorBoard
- optional AMP
- optional MedNeXt-style backbone (for stronger tumor models)

Typical 3-stage:
1) Liver coarse (full-volume + hardneg):
   python train.py --preproc_dir $PREPROC_DIR --stage liver --save_dir train_logs/liver_coarse --use_priors 0 --liver_use_bbox 0
2) Liver refine (GT bbox ROI + priors):
   python train.py --preproc_dir $PREPROC_DIR --stage liver --save_dir train_logs/liver_refine --use_priors 1 --liver_use_bbox 1 --pretrained train_logs/liver_coarse/liver_best.pth
3) Tumor (ROI inside liver + stronger backbone):
   python train.py --preproc_dir $PREPROC_DIR --stage tumor --save_dir train_logs/tumor_mednext --backbone mednext --tumor_use_pred_liver 1 --tumor_pred_liver_dir train_logs/pred_liver_for_tumor
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataload.lits_datasets import LITSDatasetB2ND
from models.unet3D import UNet3D


# -------------------- losses --------------------

def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int = 2, smooth: float = 1e-5) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    probs = torch.softmax(logits, dim=1)
    tgt_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * tgt_oh, dims)
    union = torch.sum(probs + tgt_oh, dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return ce + (1.0 - dice.mean())


def dice_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0, smooth: float = 1e-5) -> torch.Tensor:
    """Focal CE + multi-class soft dice (for tumor imbalance)."""
    ce_voxel = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce_voxel)
    focal = (alpha * (1.0 - pt).pow(gamma) * ce_voxel).mean()

    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    tgt_oh = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    inter = torch.sum(probs * tgt_oh, dims)
    union = torch.sum(probs + tgt_oh, dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return focal + (1.0 - dice.mean())


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha_fp: float = 0.3,
    gamma: float = 1.0,
    smooth: float = 1e-5,
    bce_w: float = 0.5,
    tv_w: float = 0.5,
) -> torch.Tensor:
    """Binary focal-Tversky on foreground, with optional BCE mix.

    alpha_fp: weight on false positives (FP). beta_fn = 1 - alpha_fp.
    gamma   : focal exponent.
    """
    tgt = (target == 1).float()
    p = torch.softmax(logits, dim=1)[:, 1]  # foreground prob

    tp = (p * tgt).sum()
    fp = (p * (1.0 - tgt)).sum()
    fn = ((1.0 - p) * tgt).sum()
    beta_fn = 1.0 - alpha_fp
    tversky = (tp + smooth) / (tp + alpha_fp * fp + beta_fn * fn + smooth)
    ftv = (1.0 - tversky).clamp_min(0).pow(gamma)

    if bce_w > 0:
        logit_bin = logits[:, 1] - logits[:, 0]
        bce = F.binary_cross_entropy_with_logits(logit_bin, tgt, reduction="mean")
        return tv_w * ftv + bce_w * bce
    return ftv


def foreground_dice(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> float:
    pred = torch.argmax(logits, dim=1)
    pred_fg = (pred == 1).float()
    tgt_fg = (target == 1).float()
    inter = torch.sum(pred_fg * tgt_fg)
    union = torch.sum(pred_fg) + torch.sum(tgt_fg)
    return float((2.0 * inter + smooth) / (union + smooth))


def unpack_out(out):
    if isinstance(out, dict):
        return out["logits"], out.get("sdf", None)
    if isinstance(out, (tuple, list)):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


# -------------------- liver priors (soft morphology) --------------------

def soft_dilate(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.max_pool3d(p, kernel_size=k, stride=1, padding=k // 2)


def soft_erode(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return -F.max_pool3d(-p, kernel_size=k, stride=1, padding=k // 2)


def soft_closing(p: torch.Tensor, k: int = 5) -> torch.Tensor:
    return soft_erode(soft_dilate(p, k=k), k=k)


# -------------------- utils --------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, ckpt: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def load_pretrained(model: nn.Module, ckpt_path: str, device: torch.device):
    p = Path(ckpt_path)
    if not p.is_file():
        print(f"[WARN] pretrained ckpt not found: {p}")
        return
    ckpt = torch.load(str(p), map_location=device)
    state = ckpt.get("model_state", ckpt)

    model_state = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state.items():
        if k in model_state and hasattr(v, "shape") and hasattr(model_state[k], "shape"):
            if tuple(v.shape) == tuple(model_state[k].shape):
                filtered[k] = v
            else:
                dropped += 1
        elif k in model_state:
            filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"=> Loaded pretrained weights: {p}")
    if dropped:
        print(f"   [pretrained] dropped shape-mismatch keys: {dropped}")
    if missing:
        print(f"   [pretrained] missing keys: {len(missing)}")
    if unexpected:
        print(f"   [pretrained] unexpected keys: {len(unexpected)}")


def get_amp():
    try:
        from torch.amp import autocast, GradScaler
        return autocast, GradScaler, True
    except Exception:
        from torch.cuda.amp import autocast, GradScaler
        return autocast, GradScaler, False


# -------------------- args --------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--preproc_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="liver", choices=["liver", "tumor"])
    p.add_argument("--save_dir", type=str, default="train_logs/lits")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train_ratio", type=float, default=0.8)

    p.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])

    # backbone
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "mednext"])
    p.add_argument("--mednext_k", type=int, default=7)
    p.add_argument("--mednext_expansion", type=int, default=4)
    p.add_argument("--mednext_blocks", type=int, default=2)

    # training stability
    p.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--grad_clip", type=float, default=12.0)

    # liver-specific
    p.add_argument("--use_priors", type=int, default=1, help="liver priors: ROI-global coords + sdf head + soft closing")
    p.add_argument("--liver_use_bbox", type=int, default=0, help="1: sample inside GT liver bbox (refine); 0: full-volume (coarse)")
    p.add_argument("--liver_bbox_margin", type=int, default=16)
    p.add_argument("--dropout_p", type=float, default=0.0)

    # liver coarse sampling (hard negative) + loss
    p.add_argument("--liver_loss", type=str, default="dice_ce", choices=["dice_ce", "focal_tversky"])
    p.add_argument("--liver_alpha_fp", type=float, default=0.75)
    p.add_argument("--liver_gamma", type=float, default=1.33)
    p.add_argument("--liver_bce_w", type=float, default=0.5)
    p.add_argument("--liver_tv_w", type=float, default=0.5)

    p.add_argument("--liver_pos_ratio", type=float, default=0.5)
    p.add_argument("--liver_hardneg_ratio", type=float, default=0.4)
    p.add_argument("--liver_min_pos_vox", type=int, default=64)
    p.add_argument("--liver_pos_retries", type=int, default=16)
    p.add_argument("--liver_hardneg_retries", type=int, default=32)
    p.add_argument("--liver_body_minfrac", type=float, default=0.15)
    p.add_argument("--liver_air_value", type=float, default=-3.0825)
    p.add_argument("--liver_air_eps", type=float, default=0.02)

    # tumor: pred-liver bbox mix + liver prior channel
    p.add_argument("--tumor_use_pred_liver", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_pred_liver_dir", type=str, default=None)
    p.add_argument("--tumor_pred_bbox_ratio", type=float, default=1.0, help="ratio of using pred-liver bbox (rest uses GT liver bbox)")
    p.add_argument("--tumor_pred_bbox_from", type=str, default="mask", choices=["mask", "prob", "union"])
    p.add_argument("--tumor_pred_prob_thr", type=float, default=0.10)

    p.add_argument("--tumor_add_liver_prior", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_prior_type", type=str, default="mask", choices=["mask", "prob"])

    p.add_argument("--tumor_bbox_margin", type=int, default=24)
    p.add_argument("--tumor_pos_ratio", type=float, default=0.6)
    p.add_argument("--tumor_hardneg_ratio", type=float, default=0.3)

    # tumor sampling refinements
    p.add_argument("--tumor_min_pos_vox", type=int, default=1)
    p.add_argument("--tumor_pos_retries", type=int, default=12)
    p.add_argument("--tumor_hardneg_lowfrac", type=float, default=0.0)
    p.add_argument("--tumor_hardneg_lowq", type=float, default=10.0)

    # tumor losses
    p.add_argument("--tumor_loss", type=str, default="dice_focal", choices=["dice_focal", "focal_tversky"])
    p.add_argument("--tumor_alpha", type=float, default=0.75, help="dice_focal: focal alpha; focal_tversky: FP weight")
    p.add_argument("--tumor_gamma", type=float, default=2.0, help="dice_focal: focal gamma; focal_tversky: focal exponent")
    p.add_argument("--tumor_bce_w", type=float, default=0.5)
    p.add_argument("--tumor_tv_w", type=float, default=0.5)

    # tumor boundary / sdf constraints
    p.add_argument("--tumor_use_sdf", type=int, default=0, choices=[0, 1])
    p.add_argument("--tumor_sdf_w", type=float, default=0.3)
    p.add_argument("--tumor_boundary_w", type=float, default=0.3)
    p.add_argument("--tumor_boundary_sigma", type=float, default=0.25)

    # resume / init
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--pretrained", type=str, default=None)

    return p.parse_args()


# -------------------- main --------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(save_dir / "tb_logs" / args.stage))
    print(f"[INFO] TensorBoard: {save_dir / 'tb_logs' / args.stage}")

    patch_size = tuple(int(x) for x in args.patch_size)

    use_priors = bool(args.use_priors) and (args.stage == "liver")
    return_sdf = (args.stage == "liver") or (args.stage == "tumor" and bool(args.tumor_use_sdf))

    # dataset
    train_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=True,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=return_sdf,
        sdf_clip=20.0,
        sdf_margin=0,
        liver_use_bbox=bool(args.liver_use_bbox) if args.stage == "liver" else False,
        liver_bbox_margin=int(args.liver_bbox_margin),
        liver_pos_ratio=float(args.liver_pos_ratio),
        liver_hardneg_ratio=float(args.liver_hardneg_ratio),
        liver_min_pos_vox=int(args.liver_min_pos_vox),
        liver_pos_retries=int(args.liver_pos_retries),
        liver_hardneg_retries=int(args.liver_hardneg_retries),
        liver_body_minfrac=float(args.liver_body_minfrac),
        liver_air_value=float(args.liver_air_value),
        liver_air_eps=float(args.liver_air_eps),
        # tumor
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
        tumor_min_pos_vox=int(args.tumor_min_pos_vox),
        tumor_pos_retries=int(args.tumor_pos_retries),
        tumor_hardneg_lowfrac=float(args.tumor_hardneg_lowfrac),
        tumor_hardneg_lowq=float(args.tumor_hardneg_lowq),
    )

    val_ds = LITSDatasetB2ND(
        preproc_dir=args.preproc_dir,
        stage=args.stage,
        patch_size=patch_size,
        train=False,
        train_ratio=args.train_ratio,
        seed=args.seed,
        return_sdf=return_sdf,
        sdf_clip=20.0,
        sdf_margin=0,
        liver_use_bbox=bool(args.liver_use_bbox) if args.stage == "liver" else False,
        liver_bbox_margin=int(args.liver_bbox_margin),
        liver_pos_ratio=float(args.liver_pos_ratio),
        liver_hardneg_ratio=float(args.liver_hardneg_ratio),
        liver_min_pos_vox=int(args.liver_min_pos_vox),
        liver_pos_retries=int(args.liver_pos_retries),
        liver_hardneg_retries=int(args.liver_hardneg_retries),
        liver_body_minfrac=float(args.liver_body_minfrac),
        liver_air_value=float(args.liver_air_value),
        liver_air_eps=float(args.liver_air_eps),
        # tumor
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
        tumor_min_pos_vox=int(args.tumor_min_pos_vox),
        tumor_pos_retries=int(args.tumor_pos_retries),
        tumor_hardneg_lowfrac=float(args.tumor_hardneg_lowfrac),
        tumor_hardneg_lowq=float(args.tumor_hardneg_lowq),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(int(args.num_workers) > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, int(args.num_workers) // 2),
        pin_memory=True,
        persistent_workers=(max(1, int(args.num_workers) // 2) > 0),
    )

    # infer input channels from a sample
    sample_imgs, _, _, _ = next(iter(train_loader))
    in_channels = int(sample_imgs.shape[1])
    num_classes = 2

    use_sdf_head = bool(use_priors) or (args.stage == "tumor" and bool(args.tumor_use_sdf))
    use_coords = (args.stage == "liver") and bool(use_priors)

    model = UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=32,
        dropout_p=float(args.dropout_p),
        use_coords=use_coords,
        use_sdf_head=use_sdf_head,
        backbone=args.backbone,
        mednext_k=int(args.mednext_k),
        mednext_expansion=int(args.mednext_expansion),
        mednext_blocks=int(args.mednext_blocks),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    # amp
    autocast, GradScaler, amp_has_device = get_amp()
    use_amp = bool(args.use_amp) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)

    last_path = save_dir / f"{args.stage}_last.pth"
    best_path = save_dir / f"{args.stage}_best.pth"

    start_epoch = 1
    best_val_dice = 0.0

    def pack_ckpt(epoch: int) -> dict:
        return {
            "epoch": int(epoch),
            "best_val_dice": float(best_val_dice),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            # meta
            "stage": args.stage,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "patch_size": patch_size,
            "backbone": args.backbone,
            "mednext_k": int(args.mednext_k),
            "mednext_expansion": int(args.mednext_expansion),
            "mednext_blocks": int(args.mednext_blocks),
            "use_coords": bool(use_coords),
            "use_sdf_head": bool(use_sdf_head),
            "use_priors": bool(use_priors),
            # liver meta
            "liver_loss": args.liver_loss,
            "liver_alpha_fp": float(args.liver_alpha_fp),
            "liver_gamma": float(args.liver_gamma),
            "liver_pos_ratio": float(args.liver_pos_ratio),
            "liver_hardneg_ratio": float(args.liver_hardneg_ratio),
            # tumor meta
            "tumor_use_pred_liver": bool(args.tumor_use_pred_liver),
            "tumor_pred_bbox_ratio": float(args.tumor_pred_bbox_ratio),
            "tumor_add_liver_prior": bool(args.tumor_add_liver_prior),
            "tumor_prior_type": args.tumor_prior_type,
            "tumor_bbox_margin": int(args.tumor_bbox_margin),
            "tumor_loss": args.tumor_loss,
            "tumor_alpha": float(args.tumor_alpha),
            "tumor_gamma": float(args.tumor_gamma),
            "tumor_use_sdf": bool(args.tumor_use_sdf),
        }

    # resume
    if args.resume:
        p = Path(args.resume)
        if p.is_file():
            print(f"=> Resuming from {p}")
            ckpt = torch.load(str(p), map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)
            if "optimizer_state" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print("[WARN] optimizer_state load failed:", e)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_dice = float(ckpt.get("best_val_dice", 0.0))
            print(f"   start_epoch={start_epoch}, best_val_dice={best_val_dice:.4f}")

    if args.pretrained and (args.resume is None):
        load_pretrained(model, args.pretrained, device)

    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, sdf_gt: torch.Tensor, sdf_pred: Optional[torch.Tensor]) -> torch.Tensor:
        if args.stage == "tumor":
            if args.tumor_loss == "focal_tversky":
                loss = focal_tversky_loss(
                    logits,
                    targets,
                    alpha_fp=float(args.tumor_alpha),
                    gamma=float(args.tumor_gamma),
                    bce_w=float(args.tumor_bce_w),
                    tv_w=float(args.tumor_tv_w),
                )
            else:
                loss = dice_focal_loss(
                    logits,
                    targets,
                    alpha=float(args.tumor_alpha),
                    gamma=float(args.tumor_gamma),
                )

            # optional boundary & sdf constraints (only for tumor stage)
            if bool(args.tumor_use_sdf):
                tgt = (targets == 1).float()
                sdf = sdf_gt[:, 0]
                sdf = torch.nan_to_num(sdf, nan=0.0, posinf=0.0, neginf=0.0)
                if tgt.sum() > 0:
                    w = torch.exp(-sdf.abs() / float(args.tumor_boundary_sigma)).clamp(0.05, 1.0)
                    logit_bin = (logits[:, 1] - logits[:, 0]).float()
                    loss_b = F.binary_cross_entropy_with_logits(logit_bin, tgt.float(), weight=w.float(), reduction="mean")
                    loss = loss + float(args.tumor_boundary_w) * loss_b
                    if sdf_pred is not None:
                        loss_sdf = F.l1_loss(torch.tanh(sdf_pred[:, 0].float()), sdf.float())
                        loss = loss + float(args.tumor_sdf_w) * loss_sdf
            return loss

        # liver stage
        if args.liver_loss == "focal_tversky":
            loss = focal_tversky_loss(
                logits,
                targets,
                alpha_fp=float(args.liver_alpha_fp),
                gamma=float(args.liver_gamma),
                bce_w=float(args.liver_bce_w),
                tv_w=float(args.liver_tv_w),
            )
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

        return loss

    current_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, int(args.epochs) + 1):
            current_epoch = epoch
            model.train()

            train_loss = 0.0
            train_dice = 0.0
            n_train = 0

            for imgs, targets, sdf_gt, _ids in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                sdf_gt = sdf_gt.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                if amp_has_device:
                    ctx = autocast(device_type="cuda", enabled=use_amp)
                else:
                    ctx = autocast(enabled=use_amp)

                with ctx:
                    out = model(imgs)
                    logits, sdf_pred = unpack_out(out)
                    loss = compute_loss(logits, targets, sdf_gt, sdf_pred)

                if use_amp:
                    scaler.scale(loss).backward()
                    if float(args.grad_clip) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                    optimizer.step()

                train_loss += float(loss.detach().item())
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
                for imgs, targets, sdf_gt, _ids in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    sdf_gt = sdf_gt.to(device, non_blocking=True)

                    if amp_has_device:
                        ctx = autocast(device_type="cuda", enabled=use_amp)
                    else:
                        ctx = autocast(enabled=use_amp)

                    with ctx:
                        out = model(imgs)
                        logits, sdf_pred = unpack_out(out)
                        loss = compute_loss(logits, targets, sdf_gt, sdf_pred)

                    val_loss += float(loss.detach().item())
                    val_dice += foreground_dice(logits.detach(), targets)
                    n_val += 1

            val_loss /= max(1, n_val)
            val_dice /= max(1, n_val)

            writer.add_scalar(f"{args.stage}/loss_train", train_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_train", train_dice, epoch)
            writer.add_scalar(f"{args.stage}/loss_val", val_loss, epoch)
            writer.add_scalar(f"{args.stage}/dice_val", val_dice, epoch)

            print(
                f"[{args.stage}] epoch {epoch:03d}/{args.epochs} | "
                f"train loss {train_loss:.4f} dice {train_dice:.4f} | "
                f"val loss {val_loss:.4f} dice {val_dice:.4f}"
            )

            save_ckpt(last_path, pack_ckpt(epoch))

            if val_dice > best_val_dice:
                best_val_dice = float(val_dice)
                save_ckpt(best_path, pack_ckpt(epoch))
                print(f"  => new best_val_dice={best_val_dice:.4f} saved: {best_path}")

    except KeyboardInterrupt:
        interrupt_path = save_dir / f"{args.stage}_interrupt.pth"
        try:
            save_ckpt(interrupt_path, pack_ckpt(current_epoch))
            print(f"\n[Interrupted] checkpoint saved: {interrupt_path}")
        except Exception as e:
            print(f"\n[Interrupted] failed to save checkpoint: {e}")

    writer.close()


if __name__ == "__main__":
    main()