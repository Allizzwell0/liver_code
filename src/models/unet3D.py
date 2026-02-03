# models/unet3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConcat3D(nn.Module):
    """
    在 forward 时给输入拼上归一化坐标通道 (z,y,x) -> 3 channels
    输出: (B, C+3, Z, Y, X)
    """
    def __init__(self, normalize: str = "minus1_1"):
        super().__init__()
        assert normalize in ["minus1_1", "0_1"]
        self.normalize = normalize

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,Z,Y,X)
        B, C, Z, Y, X = x.shape
        device = x.device
        dtype = x.dtype

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


class ConvBlock3d(nn.Module):
    """
    nnUNet-style 3D conv block:
    Conv3d (stride = first_stride) + IN3d + LeakyReLU
    Conv3d (stride = 1)            + IN3d + LeakyReLU
    """
    def __init__(self, in_ch: int, out_ch: int, first_stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_ch, out_ch, kernel_size=3, stride=first_stride, padding=1, bias=True
        )
        self.norm1 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.norm2 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class AttentionGate3D(nn.Module):
    """Attention gate for 3D skip connections (Attention U-Net style).

    Inputs:
      x: skip feature map (B, F_l, Z,Y,X)
      g: gating feature map (B, F_g, Z,Y,X)  (usually decoder feature after upsample)

    Output:
      attended skip feature map with same shape as x.
    """
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
        # x: skip, g: gate (same spatial size)
        a = self.relu(self.W_x(x) + self.W_g(g))
        alpha = self.psi(a)  # (B,1,Z,Y,X)
        return x * alpha

class UpBlock3d(nn.Module):
    """
    上采样 + 拼接 skip + ConvBlock3d

    参数说明：
      in_ch   : 输入（来自更深层 decoder）的通道数
      skip_ch : skip connection 的通道数
      out_ch  : 当前 stage 的输出通道数
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attn_gate: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.attn = AttentionGate3D(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 8)) if use_attn_gate else None
        self.conv = ConvBlock3d(in_ch=out_ch + skip_ch, out_ch=out_ch, first_stride=1)

    @staticmethod
    def _pad_or_crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        让 x 的 (Z,Y,X) 和 ref 完全一致：
          - x 小：pad
          - x 大：center crop
        """
        _, _, z, y, xx = x.shape
        rz, ry, rx = ref.size(2), ref.size(3), ref.size(4)

        dz, dy, dx = rz - z, ry - y, rx - xx

        # pad (only when positive)
        pad_x0 = max(0, dx // 2)
        pad_x1 = max(0, dx - dx // 2)
        pad_y0 = max(0, dy // 2)
        pad_y1 = max(0, dy - dy // 2)
        pad_z0 = max(0, dz // 2)
        pad_z1 = max(0, dz - dz // 2)
        if pad_x0 or pad_x1 or pad_y0 or pad_y1 or pad_z0 or pad_z1:
            x = F.pad(x, [pad_x0, pad_x1, pad_y0, pad_y1, pad_z0, pad_z1])

        # crop (only when negative)
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
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    nnUNet 3d_fullres 风格的 3D U-Net（Dataset003_Liver）：
    - features_per_stage = [32, 64, 128, 256, 320, 320]
    - strides = [1, 2, 2, 2, 2, 2]
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
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_sdf_head = use_sdf_head
        self.use_attn_gate = use_attn_gate
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

        self.up4 = UpBlock3d(in_ch=f5, skip_ch=f4, out_ch=f4, use_attn_gate=use_attn_gate)
        self.up3 = UpBlock3d(in_ch=f4, skip_ch=f3, out_ch=f3, use_attn_gate=use_attn_gate)
        self.up2 = UpBlock3d(in_ch=f3, skip_ch=f2, out_ch=f2, use_attn_gate=use_attn_gate)
        self.up1 = UpBlock3d(in_ch=f2, skip_ch=f1, out_ch=f1, use_attn_gate=use_attn_gate)
        self.up0 = UpBlock3d(in_ch=f1, skip_ch=f0, out_ch=f0, use_attn_gate=use_attn_gate)

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
