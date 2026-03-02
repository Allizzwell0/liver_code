# models/unet3D.py
# Minimal 3D U-Net (nnUNet-ish) + optional MedNeXt-style backbone.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------- helpers --------------------

class CoordConcat3D(nn.Module):
    """Concatenate normalized (z,y,x) coordinate channels to input.

    Input : (B,C,Z,Y,X)
    Output: (B,C+3,Z,Y,X)
    """

    def __init__(self, normalize: str = "minus1_1"):
        super().__init__()
        assert normalize in ["minus1_1", "0_1"]
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Z, Y, X = x.shape
        device, dtype = x.device, x.dtype

        if self.normalize == "minus1_1":
            zz = torch.linspace(-1, 1, Z, device=device, dtype=dtype)
            yy = torch.linspace(-1, 1, Y, device=device, dtype=dtype)
            xx = torch.linspace(-1, 1, X, device=device, dtype=dtype)
        else:
            zz = torch.linspace(0, 1, Z, device=device, dtype=dtype)
            yy = torch.linspace(0, 1, Y, device=device, dtype=dtype)
            xx = torch.linspace(0, 1, X, device=device, dtype=dtype)

        z, y, xg = torch.meshgrid(zz, yy, xx, indexing="ij")
        coords = torch.stack([z, y, xg], dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        return torch.cat([x, coords], dim=1)


# -------------------- nnUNet-ish U-Net blocks --------------------

class ConvBlock3d(nn.Module):
    """Conv3d + IN + LeakyReLU (x2), first conv may downsample by stride."""

    def __init__(self, in_ch: int, out_ch: int, first_stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class UpBlock3d(nn.Module):
    """Transposed-conv upsample + concat skip + conv block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3d(out_ch + skip_ch, out_ch, first_stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # if shapes differ due to odd sizes, pad/crop to match skip
        if x.shape[-3:] != skip.shape[-3:]:
            dz = skip.shape[-3] - x.shape[-3]
            dy = skip.shape[-2] - x.shape[-2]
            dx = skip.shape[-1] - x.shape[-1]
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class _UNet3D_Backbone(nn.Module):
    """Original nnUNet-ish U-Net backbone."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_coords: bool = False,
        use_sdf_head: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_sdf_head = use_sdf_head
        self.coord = CoordConcat3D() if use_coords else nn.Identity()

        f0 = base_filters
        f1 = base_filters * 2
        f2 = base_filters * 4
        f3 = base_filters * 8
        f4 = base_filters * 10
        f5 = base_filters * 10
        self.features_per_stage = (f0, f1, f2, f3, f4, f5)

        first_in = in_channels + 3 if use_coords else in_channels
        self.enc0 = ConvBlock3d(first_in, f0, first_stride=1)
        self.enc1 = ConvBlock3d(f0, f1, first_stride=2)
        self.enc2 = ConvBlock3d(f1, f2, first_stride=2)
        self.enc3 = ConvBlock3d(f2, f3, first_stride=2)
        self.enc4 = ConvBlock3d(f3, f4, first_stride=2)
        self.enc5 = ConvBlock3d(f4, f5, first_stride=2)

        self.bottom_dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.up4 = UpBlock3d(in_ch=f5, skip_ch=f4, out_ch=f4)
        self.up3 = UpBlock3d(in_ch=f4, skip_ch=f3, out_ch=f3)
        self.up2 = UpBlock3d(in_ch=f3, skip_ch=f2, out_ch=f2)
        self.up1 = UpBlock3d(in_ch=f2, skip_ch=f1, out_ch=f1)
        self.up0 = UpBlock3d(in_ch=f1, skip_ch=f0, out_ch=f0)

        self.out_conv = nn.Conv3d(f0, num_classes, kernel_size=1)
        self.sdf_head = nn.Conv3d(f0, 1, kernel_size=1) if use_sdf_head else None

    def forward(self, x: torch.Tensor):
        x = self.coord(x)

        x0 = self.enc0(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x5 = self.bottom_dropout(x5)

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


# -------------------- MedNeXt-style U-Net backbone --------------------

class _GN1(nn.Module):
    """GroupNorm with 1 group (LayerNorm-like for conv tensors)."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch, eps=1e-5, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MedNeXtBlock3D(nn.Module):
    """A light MedNeXt-ish block: DWConv(k) -> GN -> 1x1 expand -> GELU -> 1x1 project, residual."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 7, expansion: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        pad = k // 2

        # depthwise conv (keep channels)
        self.dw = nn.Conv3d(in_ch, in_ch, kernel_size=k, padding=pad, groups=in_ch, bias=False)
        self.norm1 = _GN1(in_ch)

        mid = max(out_ch, in_ch) * max(1, int(expansion))
        self.pw1 = nn.Conv3d(in_ch, mid, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(mid, out_ch, kernel_size=1, bias=True)

        self.norm2 = _GN1(out_ch)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.dw(x)
        y = self.norm1(y)
        y = self.pw2(self.act(self.pw1(y)))
        y = self.norm2(y)

        if self.skip is not None:
            identity = self.skip(identity)
        return y + identity


class MedNeXtStage(nn.Module):
    def __init__(self, ch: int, n_blocks: int, k: int, expansion: int):
        super().__init__()
        self.blocks = nn.Sequential(*[MedNeXtBlock3D(ch, ch, k=k, expansion=expansion) for _ in range(int(n_blocks))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class _MedNeXtUNet3D_Backbone(nn.Module):
    """Encoder-decoder with MedNeXt blocks.

    Channels follow nnUNet-like pattern: [32,64,128,256,320,320] (scaled by base_filters).
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_coords: bool = False,
        use_sdf_head: bool = False,
        k: int = 7,
        expansion: int = 4,
        blocks: int = 2,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_sdf_head = use_sdf_head
        self.coord = CoordConcat3D() if use_coords else nn.Identity()

        f0 = base_filters
        f1 = base_filters * 2
        f2 = base_filters * 4
        f3 = base_filters * 8
        f4 = base_filters * 10
        f5 = base_filters * 10
        self.features_per_stage = (f0, f1, f2, f3, f4, f5)

        first_in = in_channels + 3 if use_coords else in_channels

        self.stem = nn.Conv3d(first_in, f0, kernel_size=3, padding=1, bias=True)

        # encoder
        self.e0 = MedNeXtStage(f0, blocks, k=k, expansion=expansion)
        self.d1 = nn.Conv3d(f0, f1, kernel_size=2, stride=2, bias=True)
        self.e1 = MedNeXtStage(f1, blocks, k=k, expansion=expansion)
        self.d2 = nn.Conv3d(f1, f2, kernel_size=2, stride=2, bias=True)
        self.e2 = MedNeXtStage(f2, blocks, k=k, expansion=expansion)
        self.d3 = nn.Conv3d(f2, f3, kernel_size=2, stride=2, bias=True)
        self.e3 = MedNeXtStage(f3, blocks, k=k, expansion=expansion)
        self.d4 = nn.Conv3d(f3, f4, kernel_size=2, stride=2, bias=True)
        self.e4 = MedNeXtStage(f4, blocks, k=k, expansion=expansion)
        self.d5 = nn.Conv3d(f4, f5, kernel_size=2, stride=2, bias=True)
        self.e5 = MedNeXtStage(f5, blocks, k=k, expansion=expansion)

        self.bottom_dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # decoder
        self.u4 = nn.ConvTranspose3d(f5, f4, kernel_size=2, stride=2)
        self.f4 = nn.Conv3d(f4 + f4, f4, kernel_size=1)
        self.c4 = MedNeXtStage(f4, blocks, k=k, expansion=expansion)

        self.u3 = nn.ConvTranspose3d(f4, f3, kernel_size=2, stride=2)
        self.f3 = nn.Conv3d(f3 + f3, f3, kernel_size=1)
        self.c3 = MedNeXtStage(f3, blocks, k=k, expansion=expansion)

        self.u2 = nn.ConvTranspose3d(f3, f2, kernel_size=2, stride=2)
        self.f2 = nn.Conv3d(f2 + f2, f2, kernel_size=1)
        self.c2 = MedNeXtStage(f2, blocks, k=k, expansion=expansion)

        self.u1 = nn.ConvTranspose3d(f2, f1, kernel_size=2, stride=2)
        self.f1 = nn.Conv3d(f1 + f1, f1, kernel_size=1)
        self.c1 = MedNeXtStage(f1, blocks, k=k, expansion=expansion)

        self.u0 = nn.ConvTranspose3d(f1, f0, kernel_size=2, stride=2)
        self.f0 = nn.Conv3d(f0 + f0, f0, kernel_size=1)
        self.c0 = MedNeXtStage(f0, blocks, k=k, expansion=expansion)

        self.out_conv = nn.Conv3d(f0, num_classes, kernel_size=1)
        self.sdf_head = nn.Conv3d(f0, 1, kernel_size=1) if use_sdf_head else None

    def forward(self, x: torch.Tensor):
        x = self.coord(x)
        x0 = self.e0(self.stem(x))
        x1 = self.e1(self.d1(x0))
        x2 = self.e2(self.d2(x1))
        x3 = self.e3(self.d3(x2))
        x4 = self.e4(self.d4(x3))
        x5 = self.e5(self.d5(x4))
        x5 = self.bottom_dropout(x5)

        y4 = self.u4(x5)
        y4 = _pad_to_match(y4, x4)
        y4 = self.c4(self.f4(torch.cat([y4, x4], dim=1)))

        y3 = self.u3(y4)
        y3 = _pad_to_match(y3, x3)
        y3 = self.c3(self.f3(torch.cat([y3, x3], dim=1)))

        y2 = self.u2(y3)
        y2 = _pad_to_match(y2, x2)
        y2 = self.c2(self.f2(torch.cat([y2, x2], dim=1)))

        y1 = self.u1(y2)
        y1 = _pad_to_match(y1, x1)
        y1 = self.c1(self.f1(torch.cat([y1, x1], dim=1)))

        y0 = self.u0(y1)
        y0 = _pad_to_match(y0, x0)
        y0 = self.c0(self.f0(torch.cat([y0, x0], dim=1)))

        logits = self.out_conv(y0)
        out = {"logits": logits}
        if self.use_sdf_head:
            out["sdf"] = self.sdf_head(y0)
        return out


def _pad_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad/crop x spatially to match ref."""
    if x.shape[-3:] == ref.shape[-3:]:
        return x
    dz = ref.shape[-3] - x.shape[-3]
    dy = ref.shape[-2] - x.shape[-2]
    dx = ref.shape[-1] - x.shape[-1]
    # pad if needed
    if dz != 0 or dy != 0 or dx != 0:
        x = F.pad(x, [max(0, dx // 2), max(0, dx - dx // 2),
                      max(0, dy // 2), max(0, dy - dy // 2),
                      max(0, dz // 2), max(0, dz - dz // 2)])
    # crop if overshoot
    z, y, xx = x.shape[-3:]
    rz, ry, rx = ref.shape[-3:]
    x = x[..., :rz, :ry, :rx]
    return x


# -------------------- public model --------------------

class UNet3D(nn.Module):
    """Unified entry point.

    backbone:
      - 'unet'    : nnUNet-ish 3D U-Net
      - 'mednext' : MedNeXt-style blocks in U-Net scaffold

    Returns dict with key 'logits' (and optional 'sdf').
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_coords: bool = False,
        use_sdf_head: bool = False,
        backbone: str = "unet",
        mednext_k: int = 7,
        mednext_expansion: int = 4,
        mednext_blocks: int = 2,
    ):
        super().__init__()
        backbone = str(backbone).lower().strip()
        if backbone not in ["unet", "mednext"]:
            backbone = "unet"
        self.backbone = backbone

        if backbone == "unet":
            self.net = _UNet3D_Backbone(
                in_channels=in_channels,
                num_classes=num_classes,
                base_filters=base_filters,
                dropout_p=dropout_p,
                use_coords=use_coords,
                use_sdf_head=use_sdf_head,
            )
        else:
            self.net = _MedNeXtUNet3D_Backbone(
                in_channels=in_channels,
                num_classes=num_classes,
                base_filters=base_filters,
                dropout_p=dropout_p,
                use_coords=use_coords,
                use_sdf_head=use_sdf_head,
                k=int(mednext_k),
                expansion=int(mednext_expansion),
                blocks=int(mednext_blocks),
            )

    def forward(self, x: torch.Tensor):
        return self.net(x)
