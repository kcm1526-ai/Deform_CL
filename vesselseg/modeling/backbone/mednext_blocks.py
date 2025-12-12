"""
MedNeXt Building Blocks
Based on: https://github.com/MIC-DKFZ/MedNeXt (MICCAI 2023)
MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


class LayerNorm(nn.Module):
    """
    LayerNorm that supports both channels_first and channels_last formats.
    Channels_first is common for 3D medical imaging (N, C, D, H, W).
    """
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_first"
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # For 5D: (N, C, D, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """
    Global Response Normalization layer.
    Normalizes feature responses across spatial dimensions.
    """
    def __init__(self, in_channels: int, dim: int = 3):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, in_channels, *([1] * dim)))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, *([1] * dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute L2 norm across spatial dimensions
        if x.ndim == 5:  # 3D: (N, C, D, H, W)
            gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        else:  # 2D: (N, C, H, W)
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # Normalize
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class MedNeXtBlock(nn.Module):
    """
    Standard MedNeXt Block with depthwise separable convolutions.

    Structure:
    1. Depthwise convolution
    2. Normalization
    3. 1x1 expansion convolution
    4. GELU activation
    5. Optional GRN
    6. 1x1 compression convolution
    7. Residual connection
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = "group",  # "group" or "layer"
        n_groups: int = 4,
        grn: bool = False,
        dim: int = 3  # 2D or 3D
    ):
        super().__init__()

        self.do_res = do_res
        self.dim = dim

        # Select conv and norm based on dimensionality
        if dim == 3:
            conv_layer = nn.Conv3d
        else:
            conv_layer = nn.Conv2d

        # Depthwise convolution
        padding = kernel_size // 2
        self.conv1 = conv_layer(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_channels,
            bias=True
        )

        # Normalization
        if norm_type == "group":
            self.norm = nn.GroupNorm(
                num_groups=n_groups,
                num_channels=in_channels
            )
        else:
            self.norm = LayerNorm(in_channels)

        # 1x1 expansion
        self.conv2 = conv_layer(
            in_channels,
            in_channels * exp_r,
            kernel_size=1,
            bias=True
        )

        # Activation
        self.act = nn.GELU()

        # Optional GRN
        self.grn = GRN(in_channels * exp_r, dim=dim) if grn else None

        # 1x1 compression
        self.conv3 = conv_layer(
            in_channels * exp_r,
            out_channels,
            kernel_size=1,
            bias=True
        )

        # Residual connection (if channels differ)
        if in_channels != out_channels and do_res:
            self.res_conv = conv_layer(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)

        if self.do_res:
            if self.res_conv is not None:
                identity = self.res_conv(identity)
            x = x + identity

        return x


class MedNeXtDownBlock(nn.Module):
    """
    MedNeXt Downsampling Block with stride-2 convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: int = 4,
        grn: bool = False,
        dim: int = 3
    ):
        super().__init__()

        self.do_res = do_res

        if dim == 3:
            conv_layer = nn.Conv3d
        else:
            conv_layer = nn.Conv2d

        padding = kernel_size // 2

        # Downsampling depthwise conv
        self.conv1 = conv_layer(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            groups=in_channels,
            bias=True
        )

        # Normalization
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        else:
            self.norm = LayerNorm(in_channels)

        # 1x1 expansion
        self.conv2 = conv_layer(
            in_channels,
            in_channels * exp_r,
            kernel_size=1,
            bias=True
        )

        self.act = nn.GELU()

        self.grn = GRN(in_channels * exp_r, dim=dim) if grn else None

        # 1x1 compression
        self.conv3 = conv_layer(
            in_channels * exp_r,
            out_channels,
            kernel_size=1,
            bias=True
        )

        # Residual with downsampling
        if do_res:
            self.res_conv = conv_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)

        if self.do_res:
            identity = self.res_conv(identity)
            x = x + identity

        return x


class MedNeXtUpBlock(nn.Module):
    """
    MedNeXt Upsampling Block with transposed convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = "group",
        n_groups: int = 4,
        grn: bool = False,
        dim: int = 3
    ):
        super().__init__()

        self.do_res = do_res

        if dim == 3:
            conv_layer = nn.Conv3d
            conv_transpose_layer = nn.ConvTranspose3d
        else:
            conv_layer = nn.Conv2d
            conv_transpose_layer = nn.ConvTranspose2d

        padding = kernel_size // 2

        # Upsampling transposed conv
        self.conv1 = conv_transpose_layer(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            groups=in_channels,
            bias=True
        )

        # Normalization
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        else:
            self.norm = LayerNorm(in_channels)

        # 1x1 expansion
        self.conv2 = conv_layer(
            in_channels,
            in_channels * exp_r,
            kernel_size=1,
            bias=True
        )

        self.act = nn.GELU()

        self.grn = GRN(in_channels * exp_r, dim=dim) if grn else None

        # 1x1 compression
        self.conv3 = conv_layer(
            in_channels * exp_r,
            out_channels,
            kernel_size=1,
            bias=True
        )

        # Residual with upsampling
        if do_res:
            self.res_conv = conv_transpose_layer(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                output_padding=1,
                bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)

        if self.do_res:
            identity = self.res_conv(identity)
            x = x + identity

        return x


class OutBlock(nn.Module):
    """
    Output block for final segmentation output.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim: int = 3
    ):
        super().__init__()

        if dim == 3:
            conv_layer = nn.Conv3d
        else:
            conv_layer = nn.Conv2d

        self.conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
