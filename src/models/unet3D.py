# models/unet3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    两个 3x3x3 Conv + IN + LeakyReLU
    可选 residual（in_ch == out_ch 时启用）
    """
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)

        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        out = self.block(x)
        if self.residual:
            out = out + x
        return out


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, residual=False)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    典型 U-Net up block:
    - ConvTranspose3d 上采样
    - concat skip
    - DoubleConv
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # in_ch: 来自上一层 decoder 的通道数
        self.up = nn.ConvTranspose3d(
            in_ch, in_ch // 2,
            kernel_size=2,
            stride=2
        )
        # concat 后通道 = in_ch // 2 + skip_ch = out_ch * 2
        self.conv = DoubleConv(in_ch, out_ch, residual=False)

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐（保证和 skip 对上）
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2,
             diffZ // 2, diffZ - diffZ // 2],
        )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    LiTS 用的 3D U-Net：
      - 深度 4（下采样 4 次）
      - 通道：base, 2*base, 4*base, 8*base, 16*base
      - InstanceNorm3d + LeakyReLU
      - bottleneck 加 Dropout3d 轻微正则
    """
    def __init__(self, in_channels=1, num_classes=2, base_filters=32, dropout_p=0.2):
        super().__init__()

        # encoder
        self.inc = DoubleConv(in_channels, base_filters, residual=False)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        # bottleneck
        self.bottom_conv = DoubleConv(base_filters * 16, base_filters * 16, residual=False)
        self.bottom_dropout = nn.Dropout3d(p=dropout_p)

        # decoder
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)

        self.out_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)       # 原尺寸
        x2 = self.down1(x1)    # 1/2
        x3 = self.down2(x2)    # 1/4
        x4 = self.down3(x3)    # 1/8
        x5 = self.down4(x4)    # 1/16

        # bottleneck
        xb = self.bottom_conv(x5)
        xb = self.bottom_dropout(xb)

        # decoder
        x = self.up1(xb, x4)   # 1/8
        x = self.up2(x, x3)    # 1/4
        x = self.up3(x, x2)    # 1/2
        x = self.up4(x, x1)    # 1/1

        logits = self.out_conv(x)
        return logits
