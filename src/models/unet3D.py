# models/unet3D.py
# -*- coding: utf-8 -*-
"""
A lightweight 3D U-Net (nnUNet-like) used by this project.

Important design choice (fixes the "coords shortcut" issue):
- This model DOES NOT auto-concatenate coordinate channels.
- If you want coords, generate them in the dataset / inference code and increase in_channels accordingly.
  (This makes coords consistent with ROI/global coordinates rather than per-patch local coords.)

Forward returns a dict:
  {"logits": (B,C,Z,Y,X), "sdf": (B,1,Z,Y,X) optional}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """Squeeze-Excitation for 3D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv3d(channels, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3, 4), keepdim=True)   # GAP
        w = F.leaky_relu(self.fc1(w), negative_slope=0.01, inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0, use_se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.drop = nn.Dropout3d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.se = SEBlock3D(out_ch) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in1(self.conv1(x)))
        x = self.drop(x)
        x = self.act(self.in2(self.conv2(x)))
        x = self.se(x)
        return x


class DownBlock(nn.Module):
    """Downsample by strided conv, then conv block."""
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.0, use_se: bool = False):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.inorm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.block = ConvBlock(out_ch, out_ch, dropout_p=dropout_p, use_se=use_se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.inorm(self.down(x)))
        x = self.block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.0, use_se: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, dropout_p=dropout_p, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        if diffZ != 0 or diffY != 0 or diffX != 0:
            x = F.pad(
                x,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2,
                 diffZ // 2, diffZ - diffZ // 2],
            )
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNet3D(nn.Module):
    """
    nnUNet 3d_fullres-like 3D U-Net:
      features_per_stage = [32, 64, 128, 256, 320, 320] when base_filters=32
    """
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_sdf_head: bool = False,
        use_se: bool = False,
        # kept for backward-compat (ignored; coords should be provided by dataset/infer)
        use_coords: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.use_sdf_head = bool(use_sdf_head)
        self.use_se = bool(use_se)
        self.use_coords = bool(use_coords)

        feats = [base_filters, 2 * base_filters, 4 * base_filters, 8 * base_filters, 10 * base_filters, 10 * base_filters]
        # encoder
        self.enc0 = ConvBlock(in_channels, feats[0], dropout_p=dropout_p, use_se=use_se)
        self.enc1 = DownBlock(feats[0], feats[1], dropout_p=dropout_p, use_se=use_se)
        self.enc2 = DownBlock(feats[1], feats[2], dropout_p=dropout_p, use_se=use_se)
        self.enc3 = DownBlock(feats[2], feats[3], dropout_p=dropout_p, use_se=use_se)
        self.enc4 = DownBlock(feats[3], feats[4], dropout_p=dropout_p, use_se=use_se)
        self.enc5 = DownBlock(feats[4], feats[5], dropout_p=dropout_p, use_se=use_se)

        # decoder
        self.up4 = UpBlock(feats[5], feats[4], feats[4], dropout_p=dropout_p, use_se=use_se)
        self.up3 = UpBlock(feats[4], feats[3], feats[3], dropout_p=dropout_p, use_se=use_se)
        self.up2 = UpBlock(feats[3], feats[2], feats[2], dropout_p=dropout_p, use_se=use_se)
        self.up1 = UpBlock(feats[2], feats[1], feats[1], dropout_p=dropout_p, use_se=use_se)
        self.up0 = UpBlock(feats[1], feats[0], feats[0], dropout_p=dropout_p, use_se=use_se)

        self.out_conv = nn.Conv3d(feats[0], num_classes, kernel_size=1, bias=True)

        if self.use_sdf_head:
            self.sdf_head = nn.Conv3d(feats[0], 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        # NOTE: coords should be concatenated to x before calling forward (if needed).
        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        y4 = self.up4(x5, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)

        logits = self.out_conv(y0)
        out = {"logits": logits}
        if self.use_sdf_head:
            out["sdf"] = self.sdf_head(y0)
        return out
