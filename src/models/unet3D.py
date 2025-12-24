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
            in_ch, out_ch,
            kernel_size=3,
            stride=first_stride,
            padding=1,
            bias=True,  # 和 nnUNetPlans 中的 conv_bias=True 对齐
        )
        self.norm1 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(
            out_ch, out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.norm2 = nn.InstanceNorm3d(out_ch, eps=1e-5, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        return x


class UpBlock3d(nn.Module):
    """
    上采样 + 拼接 skip + ConvBlock3d

    参数说明：
      in_ch   : 输入（来自更深层 decoder）的通道数
      skip_ch : skip connection 的通道数
      out_ch  : 当前 stage 的输出通道数
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # 上采样到更高分辨率，输出通道直接设为 out_ch
        self.up = nn.ConvTranspose3d(
            in_ch,
            out_ch,
            kernel_size=2,
            stride=2,
        )
        # 上采样后与 skip 拼接 -> 通道 = out_ch + skip_ch
        self.conv = ConvBlock3d(
            in_ch=out_ch + skip_ch,
            out_ch=out_ch,
            first_stride=1,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # 对齐空间尺寸（防止由于奇偶数导致的 1 voxel 差异）
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

        x = torch.cat([skip, x], dim=1)  # (B, skip_ch + out_ch, ...)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    nnUNet 3d_fullres 风格的 3D U-Net（Dataset003_Liver）：

    - n_stages = 6
    - features_per_stage = [32, 64, 128, 256, 320, 320]
    - strides = [1, 2, 2, 2, 2, 2]
    - 每个 stage 2 个 Conv3d + InstanceNorm3d + LeakyReLU(0.01)
    - 下采样通过 stride=2 的卷积实现（无 MaxPool）

    为兼容你之前的调用方式，仍保留 base_filters 参数：
      - 当 base_filters=32 时，正好得到上面的配置。
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_filters: int = 32,
        dropout_p: float = 0.0,
        use_coords: bool = True, 
        use_sdf_head: bool = True
    ):
        super().__init__()
        self.use_coords = use_coords
        self.use_sdf_head = use_sdf_head
        self.coord = CoordConcat3D() if use_coords else nn.Identity()


        # 对应 nnUNet 3d_fullres 的 features_per_stage:
        # [32, 64, 128, 256, 320, 320]
        f0 = base_filters               # 32
        f1 = base_filters * 2           # 64
        f2 = base_filters * 4           # 128
        f3 = base_filters * 8           # 256
        f4 = base_filters * 10          # 320
        f5 = base_filters * 10          # 320
        self.features_per_stage = (f0, f1, f2, f3, f4, f5)

        # encoder（6 个 stage）
        # strides: [1, 2, 2, 2, 2, 2]
        first_in = in_channels + 3 if use_coords else in_channels
        self.enc0 = ConvBlock3d(first_in, f0, first_stride=1)
        self.enc1 = ConvBlock3d(f0, f1, first_stride=2)           # stage 1
        self.enc2 = ConvBlock3d(f1, f2, first_stride=2)           # stage 2
        self.enc3 = ConvBlock3d(f2, f3, first_stride=2)           # stage 3
        self.enc4 = ConvBlock3d(f3, f4, first_stride=2)           # stage 4
        self.enc5 = ConvBlock3d(f4, f5, first_stride=2)           # stage 5 (bottom)

        # 可选 bottom dropout（nnUNet 默认无 dropout，这里给个选项）
        if dropout_p > 0:
            self.bottom_dropout = nn.Dropout3d(p=dropout_p)
        else:
            self.bottom_dropout = nn.Identity()

        # decoder（5 个 up，对应 n_conv_per_stage_decoder = [2,2,2,2,2]）
        # 每一层: UpBlock3d(in_ch_deep, skip_ch_shallower, out_ch = skip_ch)
        self.up4 = UpBlock3d(in_ch=f5, skip_ch=f4, out_ch=f4)  # 1/32 -> 1/16
        self.up3 = UpBlock3d(in_ch=f4, skip_ch=f3, out_ch=f3)  # 1/16 -> 1/8
        self.up2 = UpBlock3d(in_ch=f3, skip_ch=f2, out_ch=f2)  # 1/8  -> 1/4
        self.up1 = UpBlock3d(in_ch=f2, skip_ch=f1, out_ch=f1)  # 1/4  -> 1/2
        self.up0 = UpBlock3d(in_ch=f1, skip_ch=f0, out_ch=f0)  # 1/2  -> 1/1

        # 最后 1x1x1 输出类别预测
        self.out_conv = nn.Conv3d(f0, num_classes, kernel_size=1)

        # 额外：SDF 辅助头（见改动B）
        self.sdf_head = nn.Conv3d(f0, 1, kernel_size=1) if use_sdf_head else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.coord(x)   # 拼坐标通道
        # encoder path
        x0 = self.enc0(x)           # stage 0, full res
        x1 = self.enc1(x0)          # stage 1, 1/2
        x2 = self.enc2(x1)          # stage 2, 1/4
        x3 = self.enc3(x2)          # stage 3, 1/8
        x4 = self.enc4(x3)          # stage 4, 1/16
        x5 = self.enc5(x4)          # stage 5, 1/32 (bottom)

        x5 = self.bottom_dropout(x5)

        # decoder path (从最深层往回)
        y4 = self.up4(x5, x4)       # -> 对齐 stage 4 (1/16)
        y3 = self.up3(y4, x3)       # -> 对齐 stage 3 (1/8)
        y2 = self.up2(y3, x2)       # -> 对齐 stage 2 (1/4)
        y1 = self.up1(y2, x1)       # -> 对齐 stage 1 (1/2)
        y0 = self.up0(y1, x0)       # -> 对齐 stage 0 (full res)

        logits = self.out_conv(y0)  # (B, num_classes, Z, Y, X)
        # if self.use_sdf_head:
        #     sdf = self.sdf_head(y0)  # (B,1,Z,Y,X)
        #     return logits, sdf
        # return logits
    
        out = {"logits": logits}
        if self.use_sdf_head:
            out["sdf"] = self.sdf_head(y0)
        return out

