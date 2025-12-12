"""
MedNeXt Building Blocks
Exact implementation matching: https://github.com/MIC-DKFZ/MedNeXt
For compatibility with pretrained nnUNet MedNeXt weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class MedNeXtBlock(nn.Module):
    """
    MedNeXt Block with depthwise separable convolutions.
    Matches the original MedNeXt implementation for weight compatibility.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        n_groups: int = None,
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        if dim == '2d':
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # Automatic group calculation
        if n_groups is None:
            n_groups = in_channels

        # 1. Depthwise convolution
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels
        )

        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        else:
            self.norm = nn.BatchNorm3d(in_channels) if dim == '3d' else nn.BatchNorm2d(in_channels)

        # 3. Pointwise expansion
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 4. Activation
        self.act = nn.GELU()

        # 5. Optional GRN (Global Response Normalization)
        self.grn = GRN(exp_r * in_channels, dim=dim) if grn else None

        # 6. Pointwise compression
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = x
        x1 = self.conv1(x1)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn is not None:
            x1 = self.grn(x1)
        x1 = self.conv3(x1)
        if self.do_res:
            x1 = x + x1
        return x1


class MedNeXtDownBlock(nn.Module):
    """
    MedNeXt Downsampling Block with strided convolution.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = False,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        if dim == '2d':
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # Resample path for residual
        self.resample_do_res = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        ) if do_res else None

        # 1. Depthwise strided convolution (downsampling)
        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        else:
            self.norm = nn.BatchNorm3d(in_channels) if dim == '3d' else nn.BatchNorm2d(in_channels)

        # 3. Pointwise expansion
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 4. Activation
        self.act = nn.GELU()

        # 5. Optional GRN
        self.grn = GRN(exp_r * in_channels, dim=dim) if grn else None

        # 6. Pointwise compression to out_channels
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn is not None:
            x1 = self.grn(x1)
        x1 = self.conv3(x1)
        if self.do_res:
            res = self.resample_do_res(x)
            x1 = x1 + res
        return x1


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
        do_res: bool = False,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__()

        self.do_res = do_res

        assert dim in ['2d', '3d']
        if dim == '2d':
            conv = nn.Conv2d
            conv_tr = nn.ConvTranspose2d
        else:
            conv = nn.Conv3d
            conv_tr = nn.ConvTranspose3d

        # Resample path for residual
        self.resample_do_res = conv_tr(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2
        ) if do_res else None

        # 1. Transposed depthwise convolution (upsampling)
        self.conv1 = conv_tr(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

        # 2. Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=in_channels,
                num_channels=in_channels
            )
        elif norm_type == 'layer':
            self.norm = LayerNorm(
                normalized_shape=in_channels,
                data_format='channels_first'
            )
        else:
            self.norm = nn.BatchNorm3d(in_channels) if dim == '3d' else nn.BatchNorm2d(in_channels)

        # 3. Pointwise expansion
        self.conv2 = conv(
            in_channels=in_channels,
            out_channels=exp_r * in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 4. Activation
        self.act = nn.GELU()

        # 5. Optional GRN
        self.grn = GRN(exp_r * in_channels, dim=dim) if grn else None

        # 6. Pointwise compression
        self.conv3 = conv(
            in_channels=exp_r * in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x, dummy_tensor=None):
        x1 = self.conv1(x)
        x1 = self.act(self.conv2(self.norm(x1)))
        if self.grn is not None:
            x1 = self.grn(x1)
        x1 = self.conv3(x1)
        if self.do_res:
            res = self.resample_do_res(x)
            x1 = x1 + res
        return x1


class OutBlock(nn.Module):
    """
    Output block for final segmentation prediction.
    """
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        dim: str = '3d'
    ):
        super().__init__()

        assert dim in ['2d', '3d']
        if dim == '2d':
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        self.conv_out = conv(in_channels, n_classes, kernel_size=1)

    def forward(self, x, dummy_tensor=None):
        return self.conv_out(x)


class LayerNorm(nn.Module):
    """
    LayerNorm supporting channels_first format (N, C, D, H, W).
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
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 5:  # 3D
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:  # 2D
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """
    Global Response Normalization layer.
    """
    def __init__(self, in_channels: int, dim: str = '3d'):
        super().__init__()
        if dim == '3d':
            self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:  # 3D
            gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        else:  # 2D
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x
