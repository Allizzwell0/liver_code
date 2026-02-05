# models/unet3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConcat3D(nn.Module):
    """
    Concat normalized coords (z,y,x) as 3 extra channels.
    Output: (B, C+3, Z, Y, X)
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

        z, y, xg = torch.meshgrid(zz, yy, xx, indexing="ij")  # (Z,Y,X)
        coords = torch.stack([z, y, xg], dim=0).unsqueeze(0).repeat(B, 1, 1, 1, 1)  # (B,3,Z,Y,X)
        return torch.cat([x, coords], dim=1)


# -------------------- lightweight attention blocks -------------------- #

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation (3D). Very cheap channel attention."""
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(ch // r, 8)
        self.fc1 = nn.Conv3d(ch, hidden, kernel_size=1, bias=True)
        self.fc2 = nn.Conv3d(hidden, ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool3d(x, 1)
        s = F.leaky_relu(self.fc1(s), 0.01, inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class CBAM3D(nn.Module):
    """CBAM for 3D: channel attention + spatial attention (lightweight)."""
    def __init__(self, ch: int, r: int = 8, spatial_k: int = 7):
        super().__init__()
        hidden = max(ch // r, 8)
        self.mlp = nn.Sequential(
            nn.Conv3d(ch, hidden, 1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden, ch, 1, bias=True),
        )
        self.spatial = nn.Conv3d(2, 1, kernel_size=spatial_k, padding=spatial_k // 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel attention
        avg = F.adaptive_avg_pool3d(x, 1)
        mx = F.adaptive_max_pool3d(x, 1)
        ca = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        x = x * ca

        # spatial attention
        avg_c = torch.mean(x, dim=1, keepdim=True)
        mx_c = torch.max(x, dim=1, keepdim=True).values
        sa = torch.sigmoid(self.spatial(torch.cat([avg_c, mx_c], dim=1)))
        return x * sa


# -------------------- building blocks -------------------- #

class ConvBlock3d(nn.Module):
    """
    nnUNet-style 3D block:
      Conv3d(stride=first_stride) + IN3d + LeakyReLU
      Conv3d(stride=1)            + IN3d + LeakyReLU
    Optional: SE / CBAM (no extra params when disabled).
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        first_stride: int = 1,
        use_se: bool = False,
        use_cbam: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

        # keep compatibility: default is Identity (no params)
        if use_cbam:
            self.attn = CBAM3D(out_ch)
        elif use_se:
            self.attn = SEBlock3D(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.attn(x)
        return x


class AttentionGate3D(nn.Module):
    """Attention gate for 3D skip connections (Attention U-Net style)."""
    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(g_ch, inter_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_ch, eps=1e-5, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(x_ch, inter_ch, kernel_size=1, bias=True),
            nn.InstanceNorm3d(inter_ch, eps=1e-5, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.relu = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        a = self.relu(self.W_x(x) + self.W_g(g))
        alpha = self.psi(a)  # (B,1,Z,Y,X)
        return x * alpha


class UpBlock3d(nn.Module):
    """Upsample + (optional) attention-gated skip + concat + ConvBlock3d."""
    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        use_attn_gate: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
    ):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate3D(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 8)) if use_attn_gate else None
        self.conv = ConvBlock3d(in_ch=out_ch + skip_ch, out_ch=out_ch, first_stride=1, use_se=use_se, use_cbam=use_cbam)

    @staticmethod
    def _pad_or_crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, _, z, y, xx = x.shape
        rz, ry, rx = ref.size(2), ref.size(3), ref.size(4)

        dz, dy, dx = rz - z, ry - y, rx - xx

        # pad
        pad_x0, pad_x1 = max(0, dx // 2), max(0, dx - dx // 2)
        pad_y0, pad_y1 = max(0, dy // 2), max(0, dy - dy // 2)
        pad_z0, pad_z1 = max(0, dz // 2), max(0, dz - dz // 2)
        if pad_x0 or pad_x1 or pad_y0 or pad_y1 or pad_z0 or pad_z1:
            x = F.pad(x, [pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1])

        # center crop
        _, _, z, y, xx = x.shape
        if z > rz:
            s = (z - rz) // 2
            x = x[:, :, s:s + rz, :, :]
        if y > ry:
            s = (y - ry) // 2
            x = x[:, :, :, s:s + ry, :]
        if xx > rx:
            s = (xx - rx) // 2
            x = x[:, :, :, :, s:s + rx]
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._pad_or_crop_to_match(x, skip)
        if self.attn is not None:
            skip = self.attn(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    nnUNet-like 3D U-Net with:
      - optional coord channels (ROI-global coords)
      - optional attention gates on skip connections
      - optional SE/CBAM inside each stage (cheap, helps reduce FP by focusing on context)
      - optional deep supervision (aux logits) for tumor (tiny lesion)
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_coords: bool = True,
        use_sdf_head: bool = True,
        use_attn_gate: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_sdf_head = use_sdf_head
        self.use_attn_gate = use_attn_gate
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.deep_supervision = deep_supervision

        self.coord = CoordConcat3D() if use_coords else nn.Identity()

        f0 = base_filters
        f1 = base_filters * 2
        f2 = base_filters * 4
        f3 = base_filters * 8
        f4 = base_filters * 10
        f5 = base_filters * 10
        self.features_per_stage = (f0, f1, f2, f3, f4, f5)

        first_in = in_channels + 3 if use_coords else in_channels
        self.enc0 = ConvBlock3d(first_in, f0, first_stride=1, use_se=use_se, use_cbam=use_cbam)
        self.enc1 = ConvBlock3d(f0, f1, first_stride=2, use_se=use_se, use_cbam=use_cbam)
        self.enc2 = ConvBlock3d(f1, f2, first_stride=2, use_se=use_se, use_cbam=use_cbam)
        self.enc3 = ConvBlock3d(f2, f3, first_stride=2, use_se=use_se, use_cbam=use_cbam)
        self.enc4 = ConvBlock3d(f3, f4, first_stride=2, use_se=use_se, use_cbam=use_cbam)
        self.enc5 = ConvBlock3d(f4, f5, first_stride=2, use_se=use_se, use_cbam=use_cbam)

        self.bottom_dropout = nn.Dropout3d(p=dropout_p) if dropout_p > 0 else nn.Identity()

        self.up4 = UpBlock3d(f5, f4, f4, use_attn_gate=use_attn_gate, use_se=use_se, use_cbam=use_cbam)
        self.up3 = UpBlock3d(f4, f3, f3, use_attn_gate=use_attn_gate, use_se=use_se, use_cbam=use_cbam)
        self.up2 = UpBlock3d(f3, f2, f2, use_attn_gate=use_attn_gate, use_se=use_se, use_cbam=use_cbam)
        self.up1 = UpBlock3d(f2, f1, f1, use_attn_gate=use_attn_gate, use_se=use_se, use_cbam=use_cbam)
        self.up0 = UpBlock3d(f1, f0, f0, use_attn_gate=use_attn_gate, use_se=use_se, use_cbam=use_cbam)

        self.out_conv = nn.Conv3d(f0, num_classes, kernel_size=1)
        self.sdf_head = nn.Conv3d(f0, 1, kernel_size=1) if use_sdf_head else None

        # deep supervision heads (decoder stages: y1/y2/y3) -> upsample to full size
        if deep_supervision:
            self.aux1 = nn.Conv3d(f1, num_classes, kernel_size=1)
            self.aux2 = nn.Conv3d(f2, num_classes, kernel_size=1)
            self.aux3 = nn.Conv3d(f3, num_classes, kernel_size=1)
        else:
            self.aux1 = self.aux2 = self.aux3 = None

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

        if self.deep_supervision and (self.aux1 is not None):
            # upsample aux logits to match logits spatial size
            sz = logits.shape[2:]
            aux_logits = []
            aux_logits.append(F.interpolate(self.aux1(y1), size=sz, mode="trilinear", align_corners=False))
            aux_logits.append(F.interpolate(self.aux2(y2), size=sz, mode="trilinear", align_corners=False))
            aux_logits.append(F.interpolate(self.aux3(y3), size=sz, mode="trilinear", align_corners=False))
            out["aux_logits"] = aux_logits

        if self.use_sdf_head:
            out["sdf"] = self.sdf_head(y0)

        return out
