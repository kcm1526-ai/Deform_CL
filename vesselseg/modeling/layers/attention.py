"""
Attention mechanisms for 3D medical image segmentation.

This module contains attention blocks designed to improve feature
representation for vessel segmentation:
- SE (Squeeze-and-Excitation) blocks for channel attention
- CBAM (Convolutional Block Attention Module) for channel + spatial attention
- Coordinate Attention for position-aware attention
- Multi-head self-attention for global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation block for channel attention.

    Reference: "Squeeze-and-Excitation Networks" (Hu et al., CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Channel reduction ratio for bottleneck
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC layers
        y = self.fc(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ChannelAttention3D(nn.Module):
    """
    Channel attention module using both avg and max pooling.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention3D(nn.Module):
    """
    3D Spatial attention module.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate along channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)
        # Spatial attention map
        return self.sigmoid(self.conv(concat))


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module.

    Combines channel and spatial attention for comprehensive feature refinement.
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class CoordinateAttention3D(nn.Module):
    """
    3D Coordinate Attention for position-aware attention.

    Encodes spatial information along each axis independently.
    Reference: "Coordinate Attention for Efficient Mobile Network Design"
    """

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mid_channels = max(8, channels // reduction)

        self.conv1 = nn.Conv3d(channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_d = nn.Conv3d(mid_channels, channels, kernel_size=1, bias=False)
        self.conv_h = nn.Conv3d(mid_channels, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv3d(mid_channels, channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.size()

        # Pool along each axis
        x_d = self.pool_d(x).view(b, c, d, 1, 1)
        x_h = self.pool_h(x).view(b, c, 1, h, 1)
        x_w = self.pool_w(x).view(b, c, 1, 1, w)

        # Shared transform
        y_d = self.act(self.bn1(self.conv1(x_d)))
        y_h = self.act(self.bn1(self.conv1(x_h)))
        y_w = self.act(self.bn1(self.conv1(x_w)))

        # Separate attention for each axis
        a_d = self.sigmoid(self.conv_d(y_d))
        a_h = self.sigmoid(self.conv_h(y_h))
        a_w = self.sigmoid(self.conv_w(y_w))

        # Combine attention maps
        attention = a_d * a_h * a_w

        return x * attention


class AxialAttention3D(nn.Module):
    """
    Axial attention for efficient self-attention in 3D volumes.

    Decomposes 3D attention into three 1D attentions along each axis.
    Much more efficient than full 3D self-attention.
    """

    def __init__(self, channels: int, num_heads: int = 8, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Separate projections for each axis
        self.qkv_d = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.qkv_h = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.qkv_w = nn.Linear(channels, channels * 3, bias=qkv_bias)

        self.proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def axis_attention(self, x: torch.Tensor, qkv_layer: nn.Module,
                       axis: int) -> torch.Tensor:
        """Apply attention along a specific axis."""
        shape = x.shape
        # Reshape to bring target axis to sequence position
        if axis == 0:  # D axis
            x = x.permute(0, 3, 4, 2, 1).contiguous()  # B, H, W, D, C
            seq_len = shape[2]
        elif axis == 1:  # H axis
            x = x.permute(0, 2, 4, 3, 1).contiguous()  # B, D, W, H, C
            seq_len = shape[3]
        else:  # W axis
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # B, D, H, W, C
            seq_len = shape[4]

        b = x.shape[0]
        spatial = x.shape[1] * x.shape[2]
        x = x.view(b * spatial, seq_len, -1)

        # QKV projection
        qkv = qkv_layer(x).reshape(b * spatial, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B*spatial, heads, seq, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b * spatial, seq_len, -1)

        # Reshape back
        if axis == 0:
            x = x.view(b, shape[3], shape[4], shape[2], -1).permute(0, 4, 3, 1, 2)
        elif axis == 1:
            x = x.view(b, shape[2], shape[4], shape[3], -1).permute(0, 4, 1, 3, 2)
        else:
            x = x.view(b, shape[2], shape[3], shape[4], -1).permute(0, 4, 1, 2, 3)

        return x.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Apply attention along each axis
        x_d = self.axis_attention(x, self.qkv_d, 0)
        x_h = self.axis_attention(x, self.qkv_h, 1)
        x_w = self.axis_attention(x, self.qkv_w, 2)

        # Combine
        x = x_d + x_h + x_w

        # Project and residual
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).view(b * d * h * w, c)
        x = self.proj(x)
        x = self.norm(x)
        x = x.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)

        return identity + x


class MultiScaleAttention3D(nn.Module):
    """
    Multi-scale attention for capturing features at different scales.

    Useful for detecting both thick and thin vessels simultaneously.
    """

    def __init__(self, channels: int, scales: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.scales = scales

        self.attentions = nn.ModuleList([
            CBAM3D(channels) for _ in scales
        ])

        self.fusion = nn.Conv3d(channels * len(scales), channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, d, h, w = x.shape
        features = []

        for scale, attention in zip(self.scales, self.attentions):
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool3d(x, kernel_size=scale, stride=scale)
                # Apply attention
                x_scaled = attention(x_scaled)
                # Upsample back
                x_scaled = F.interpolate(x_scaled, size=(d, h, w), mode='trilinear',
                                        align_corners=False)
            else:
                x_scaled = attention(x)

            features.append(x_scaled)

        # Concatenate and fuse
        x_cat = torch.cat(features, dim=1)
        return self.fusion(x_cat) + x  # Residual connection


class AttentionGate3D(nn.Module):
    """
    Attention Gate for skip connections in U-Net.

    Suppresses irrelevant features and highlights salient features.
    Reference: "Attention U-Net" (Oktay et al., 2018)
    """

    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: Optional[int] = None):
        """
        Args:
            gate_channels: Channels from the gating signal (decoder path)
            skip_channels: Channels from the skip connection (encoder path)
            inter_channels: Intermediate channels (default: skip_channels // 2)
        """
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2

        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal from decoder (lower resolution)
            x: Skip connection from encoder (higher resolution)

        Returns:
            Attention-weighted skip features
        """
        # Upsample gating signal if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
