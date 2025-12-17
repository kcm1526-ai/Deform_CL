"""
Attention Blocks for Enhanced Medical Image Segmentation

Implements:
1. Squeeze-and-Excitation (SE) - Channel attention
2. Spatial Attention - Focus on important spatial regions
3. CBAM - Convolutional Block Attention Module (Channel + Spatial)
4. Attention Gate - For skip connections
5. Self-Attention - Long-range dependencies (for bottleneck)
6. Multi-Scale Attention - Vessel-specific multi-scale feature aggregation

References:
- SE-Net: https://arxiv.org/abs/1709.01507
- CBAM: https://arxiv.org/abs/1807.06521
- Attention U-Net: https://arxiv.org/abs/1804.03999
- Non-local Neural Networks: https://arxiv.org/abs/1711.07971
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SqueezeExcitation3D(nn.Module):
    """
    3D Squeeze-and-Excitation block for channel attention.

    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        activation: str = 'relu'
    ):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)

        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x.shape
        # Squeeze: Global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1, 1)
        # Scale
        return x * y.expand_as(x)


class SpatialAttention3D(nn.Module):
    """
    3D Spatial Attention module.

    Generates a spatial attention map to focus on important regions.
    Uses both max-pooling and average-pooling along the channel axis.
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
        # Concatenate and convolve
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        # Generate attention map
        return x * self.sigmoid(y)


class CBAM3D(nn.Module):
    """
    3D Convolutional Block Attention Module.

    Combines channel attention (SE) and spatial attention sequentially.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7
    ):
        super().__init__()
        self.channel_attention = SqueezeExcitation3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AttentionGate3D(nn.Module):
    """
    3D Attention Gate for skip connections.

    Filters features from encoder before concatenating with decoder,
    suppressing irrelevant regions and highlighting salient features.

    Reference: Attention U-Net (Oktay et al., 2018)
    """

    def __init__(
        self,
        gate_channels: int,  # From decoder (gating signal)
        skip_channels: int,  # From encoder (skip connection)
        inter_channels: Optional[int] = None
    ):
        super().__init__()

        if inter_channels is None:
            inter_channels = skip_channels // 2
        inter_channels = max(inter_channels, 8)

        # Transformations
        self.W_gate = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_channels), inter_channels)
        )

        self.W_skip = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, inter_channels), inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        gate: torch.Tensor,  # From decoder
        skip: torch.Tensor   # From encoder
    ) -> torch.Tensor:
        # Match spatial dimensions
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(
                gate, size=skip.shape[2:],
                mode='trilinear', align_corners=False
            )

        # Compute attention
        g = self.W_gate(gate)
        s = self.W_skip(skip)

        # Additive attention
        psi = self.relu(g + s)
        psi = self.psi(psi)

        # Apply attention to skip connection
        return skip * psi


class MultiHeadSelfAttention3D(nn.Module):
    """
    3D Multi-Head Self-Attention for capturing long-range dependencies.

    Computes self-attention over spatial dimensions, useful for
    capturing global context in tubular structures like vessels.

    Uses efficient attention with reduced spatial resolution.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 2  # Spatial reduction ratio for efficiency
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(channels, channels, bias=qkv_bias)
        self.kv = nn.Linear(channels, channels * 2, bias=qkv_bias)

        # Spatial reduction for K and V (efficiency)
        if sr_ratio > 1:
            self.sr = nn.Conv3d(
                channels, channels,
                kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_norm = nn.LayerNorm(channels)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W

        # Reshape for attention: (B, C, D, H, W) -> (B, N, C)
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Query
        q = self.q(x_flat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Key and Value with spatial reduction for efficiency
        if self.sr_ratio > 1:
            x_sr = self.sr(x)  # Reduce spatial dimensions
            x_sr = x_sr.flatten(2).transpose(1, 2)  # (B, N_reduced, C)
            x_sr = self.sr_norm(x_sr)
            kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_flat).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)

        # Reshape back: (B, N, C) -> (B, C, D, H, W)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)

        return x + out  # Residual connection


class EfficientSelfAttention3D(nn.Module):
    """
    Efficient 3D Self-Attention using linear complexity attention.

    Uses the efficient attention mechanism from EfficientViT/Linear Attention
    which reduces complexity from O(N^2) to O(N).

    Better suited for high-resolution 3D medical images.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)

        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W

        # Reshape for attention
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # QKV projection
        qkv = self.qkv(x_flat).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Normalize Q and K for linear attention
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Linear attention: O(N) instead of O(N^2)
        # Compute K^T @ V first (head_dim x head_dim), then Q @ (K^T @ V)
        attn = k.transpose(-2, -1) @ v  # (B, heads, head_dim, head_dim)
        attn = attn * self.temperature
        out = q @ attn  # (B, heads, N, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, C, D, H, W)

        return x + out  # Residual


class MultiScaleAttention3D(nn.Module):
    """
    Multi-Scale Attention for vessel segmentation.

    Aggregates features from multiple scales to capture vessels
    of different sizes. Particularly important for lung vessels
    which range from large pulmonary arteries to tiny capillaries.
    """

    def __init__(
        self,
        channels: int,
        scales: Tuple[int, ...] = (1, 2, 4),
        reduction: int = 4
    ):
        super().__init__()
        self.scales = scales

        # Per-scale convolutions
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels, channels // reduction, kernel_size=3,
                         padding=s, dilation=s, bias=False),
                nn.GroupNorm(min(8, channels // reduction), channels // reduction),
                nn.ReLU(inplace=True)
            )
            for s in scales
        ])

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(len(scales) * (channels // reduction), channels,
                     kernel_size=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale features
        multi_scale = [conv(x) for conv in self.scale_convs]

        # Concatenate and generate attention
        fused = torch.cat(multi_scale, dim=1)
        attn = self.fusion(fused)

        return x * attn


class VesselAttentionBlock(nn.Module):
    """
    Complete attention block designed for vessel segmentation.

    Combines:
    1. Channel attention (SE)
    2. Spatial attention
    3. Multi-scale attention (vessel-specific)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_multi_scale: bool = True
    ):
        super().__init__()

        self.se = SqueezeExcitation3D(channels, reduction)
        self.spatial = SpatialAttention3D(kernel_size=7)

        if use_multi_scale:
            self.multi_scale = MultiScaleAttention3D(channels)
        else:
            self.multi_scale = None

        # Final fusion
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        out = self.se(x)

        # Spatial attention
        out = self.spatial(out)

        # Multi-scale attention
        if self.multi_scale is not None:
            out = self.multi_scale(out)

        # Learnable residual
        return x + self.gamma * out


class EdgeEnhancementModule(nn.Module):
    """
    Edge Enhancement Module for vessel boundary detection.

    Vessels have thin boundaries that are often lost in deep networks.
    This module enhances edge features to improve boundary segmentation.

    Reference: ER-Net (Edge-Reinforced Network)
    """

    def __init__(self, channels: int):
        super().__init__()

        # Sobel-like edge detection (learnable)
        self.edge_conv = nn.Conv3d(channels, channels, kernel_size=3,
                                   padding=1, bias=False, groups=channels)

        # Edge refinement
        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
        )

        # Learnable combination
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)

        # Initialize edge conv with Sobel-like kernels
        self._init_edge_kernels()

    def _init_edge_kernels(self):
        """Initialize with Sobel-like kernels for edge detection."""
        # Simple Laplacian-like initialization
        with torch.no_grad():
            kernel = torch.zeros(3, 3, 3)
            kernel[1, 1, 1] = -6
            kernel[0, 1, 1] = 1
            kernel[2, 1, 1] = 1
            kernel[1, 0, 1] = 1
            kernel[1, 2, 1] = 1
            kernel[1, 1, 0] = 1
            kernel[1, 1, 2] = 1

            # Expand to all channels
            weight = self.edge_conv.weight.data
            for i in range(weight.shape[0]):
                weight[i, 0] = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detect edges
        edges = self.edge_conv(x)
        edges = torch.abs(edges)  # Edge magnitude

        # Refine edges
        edges = self.refine(edges)

        # Enhance original features with edges
        return x + self.alpha * edges
