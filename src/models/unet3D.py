# models/unet3D.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # 尺寸对齐
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
    def __init__(self, in_channels=1, num_classes=2, base_filters=32):
        super().__init__()
        # 下采样 4 层
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)

        # bottleneck
        self.bottom = DoubleConv(base_filters * 16, base_filters * 16)

        # 上采样 4 层
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)

        self.out_conv = nn.Conv3d(base_filters, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)       # 原尺寸
        x2 = self.down1(x1)    # 1/2
        x3 = self.down2(x2)    # 1/4
        x4 = self.down3(x3)    # 1/8
        x5 = self.down4(x4)    # 1/16

        xb = self.bottom(x5)

        x = self.up1(xb, x4)   # 回到 1/8
        x = self.up2(x, x3)    # 1/4
        x = self.up3(x, x2)    # 1/2
        x = self.up4(x, x1)    # 1/1 (原尺寸)

        logits = self.out_conv(x)
        return logits
