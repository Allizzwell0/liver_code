# 网络结构与功能说明文档

本文档说明本项目使用的 3D U-Net（nnUNet 风格）网络结构与功能模块。网络定义位于 `src/models/unet3D.py`，可作为肝脏/肿瘤分割主干模型使用。整体结构采用 3D 编码器-解码器架构（encoder-decoder），并可选配坐标通道输入与 SDF（Signed Distance Field）预测头，以提升空间感知能力与边界学习效果。

## 1. 总体结构概览

- **结构类型**：3D U-Net（nnUNet 3d_fullres 风格）。
- **输入**：三维体数据 `(B, C, Z, Y, X)`，默认 `C=1`。
- **输出**：
  - `logits`：`num_classes` 通道的体素级分类预测。
  - `sdf`（可选）：1 通道的 SDF 回归输出，用于边界/形状约束。
- **特性**：
  - 编码器与解码器均为 **Conv3d + InstanceNorm3d + LeakyReLU** 的双卷积块；
  - 下采样通过卷积 `stride=2` 完成，上采样通过 `ConvTranspose3d` 完成；
  - 解码阶段使用 skip connection 进行高分辨率特征融合；
  - 可选 **坐标拼接**（CoordConcat3D），为每个体素引入归一化 `(z, y, x)` 坐标通道。

## 2. 关键模块说明

### 2.1 CoordConcat3D：坐标通道拼接

**作用**：将归一化的 3D 坐标 `(z, y, x)` 拼接到输入特征通道中，提升网络对空间位置的感知能力。默认输出通道数为 `C+3`。

- 支持归一化方式：
  - `minus1_1`：坐标归一化到 `[-1, 1]`；
  - `0_1`：坐标归一化到 `[0, 1]`。

该模块在 `UNet3D.forward` 入口处执行，可通过 `use_coords` 参数控制启用。若启用，则输入的通道数自动增加 3 个坐标通道。

### 2.2 ConvBlock3d：基础卷积块

**结构**（nnUNet 风格）：

1. `Conv3d`（可选 stride，用于下采样）
2. `InstanceNorm3d`
3. `LeakyReLU`
4. `Conv3d`（stride=1）
5. `InstanceNorm3d`
6. `LeakyReLU`

该结构用于编码器与解码器中所有卷积特征提取阶段。

### 2.3 UpBlock3d：上采样+跳跃连接块

**功能**：

1. 先使用 `ConvTranspose3d` 对深层特征进行上采样；
2. 将上采样结果与编码器对应层的 skip connection 特征拼接；
3. 拼接后经过 `ConvBlock3d` 进行特征融合。

**尺寸对齐**：内部包含 `_pad_or_crop_to_match`，确保上采样后的特征在 `(Z, Y, X)` 维度上与 skip 特征一致，必要时进行 padding 或中心裁剪，以避免尺寸错配。

## 3. 编码器（Encoder）结构

编码器由 6 个阶段组成，每个阶段使用 `ConvBlock3d`。第 1 层 stride=1，其余层 stride=2 以实现下采样。特征通道逐步增加：

```
[32, 64, 128, 256, 320, 320]
```

具体为：

- `enc0`: 输入 -> 32 通道（不下采样）
- `enc1`: 32 -> 64 通道（stride=2）
- `enc2`: 64 -> 128 通道（stride=2）
- `enc3`: 128 -> 256 通道（stride=2）
- `enc4`: 256 -> 320 通道（stride=2）
- `enc5`: 320 -> 320 通道（stride=2）

在 `enc5` 后可使用 `Dropout3d`（由 `dropout_p` 控制）进行正则化。

## 4. 解码器（Decoder）结构

解码器由 5 个 `UpBlock3d` 组成，与编码器形成对称结构：

```
enc5 -> up4 -> up3 -> up2 -> up1 -> up0
```

每个 `UpBlock3d` 输出通道数分别为 `[320, 256, 128, 64, 32]`，并通过 skip connection 与 `enc4` 到 `enc0` 对应层特征融合。

解码器输出的最高分辨率特征 `y0` 进入输出头（Output Heads）。

## 5. 输出头（Output Heads）

### 5.1 分类输出头（logits）

- 使用 `1×1×1` 卷积将特征映射到 `num_classes` 通道，输出 `logits`。
- 常用于多类别分割任务（如背景+肝脏，或背景+肝脏+肿瘤）。

### 5.2 SDF 输出头（可选）

- 若 `use_sdf_head=True`，则使用 `1×1×1` 卷积输出 1 通道 SDF 回归预测。
- 用于增强边界监督或提供形状先验。

最终输出格式为：

```python
{
    "logits": logits,
    "sdf": sdf  # 可选
}
```

## 6. 前向流程总结

1. 输入体数据 `(B, C, Z, Y, X)`；
2. 可选拼接坐标通道；
3. 编码器逐层下采样并提取特征；
4. 解码器逐层上采样并与编码器特征融合；
5. 输出分类 logits 与可选的 SDF 结果。

## 7. 可配置参数说明（UNet3D）

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `in_channels` | 输入通道数 | 1 |
| `num_classes` | 输出类别数 | 2 |
| `base_filters` | 基础通道数（f0） | 32 |
| `dropout_p` | 底部 Dropout 概率 | 0.0 |
| `use_coords` | 是否拼接坐标通道 | True |
| `use_sdf_head` | 是否输出 SDF | True |

## 8. 典型用途

- **肝脏分割**：只需 `logits` 输出，设置 `num_classes=2`（背景+肝脏）。
- **肿瘤分割**：可使用多类别输出，如背景+肝脏+肿瘤。
- **边界增强**：启用 `use_sdf_head` 以训练 SDF 分支，强化形状与边界信息。

---

若需进一步了解训练/推理流程，可参考 `src/train.py` 与 `src/infer.py` 相关脚本，但本说明聚焦于网络结构与功能模块。
