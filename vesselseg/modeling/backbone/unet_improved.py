"""
Improved 3D U-Net backbone for thin vessel segmentation.

Enhancements over the original U-Net:
1. Attention gates in skip connections (Attention U-Net)
2. SE blocks after each conv block
3. Deep supervision at multiple scales
4. Residual connections in encoder/decoder blocks
5. Multi-scale feature aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from detectron2.modeling import BACKBONE_REGISTRY

from ..layers.conv_blocks import Conv3d, ConvTranspose3d, get_norm_3d
from ..layers.attention import (
    SEBlock3D, CBAM3D, AttentionGate3D, MultiScaleAttention3D
)


class ResidualConvBlock3D(nn.Module):
    """
    Residual convolutional block with SE attention.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'SyncBN',
                 use_se: bool = True, se_reduction: int = 16):
        super().__init__()

        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = get_norm_3d(norm, out_channels)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = get_norm_3d(norm, out_channels)
        self.relu = nn.ReLU(inplace=True)

        # SE attention
        self.use_se = use_se
        if use_se:
            self.se = SEBlock3D(out_channels, reduction=se_reduction)

        # Skip connection
        self.skip = nn.Identity() if in_channels == out_channels else \
                   nn.Sequential(
                       Conv3d(in_channels, out_channels, kernel_size=1),
                       get_norm_3d(norm, out_channels)
                   )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.use_se:
            out = self.se(out)

        out = out + identity
        out = self.relu(out)

        return out


class EncoderBlock(nn.Module):
    """
    Encoder block with residual connections and downsampling.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = 'SyncBN',
                 use_se: bool = True, downsample: bool = True):
        super().__init__()

        self.conv_block = ResidualConvBlock3D(in_channels, out_channels, norm, use_se)

        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (skip_features, downsampled_features) if downsample else (features, features)
        """
        skip = self.conv_block(x)

        if self.downsample:
            down = self.pool(skip)
            return skip, down
        else:
            return skip, skip


class DecoderBlock(nn.Module):
    """
    Decoder block with attention gate and residual connections.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm: str = 'SyncBN', use_attention_gate: bool = True,
                 use_se: bool = True):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.use_attention_gate = use_attention_gate
        if use_attention_gate:
            self.attention_gate = AttentionGate3D(in_channels, skip_channels)

        # After concatenation
        combined_channels = in_channels + skip_channels
        self.conv_block = ResidualConvBlock3D(combined_channels, out_channels, norm, use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample
        x = self.upsample(x)

        # Ensure same size
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        # Apply attention gate to skip connection
        if self.use_attention_gate:
            skip = self.attention_gate(x, skip)

        # Concatenate and process
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)

        return x


class DeepSupervisionHead(nn.Module):
    """
    Deep supervision head for auxiliary loss at intermediate scales.
    """

    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, ...]) -> torch.Tensor:
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='trilinear', align_corners=False)
        return x


class UNetImproved(nn.Module):
    """
    Improved U-Net with attention gates, SE blocks, and deep supervision.

    Key improvements:
    1. Residual connections in all blocks
    2. SE attention after each conv block
    3. Attention gates in skip connections
    4. Deep supervision at multiple decoder levels
    5. Multi-scale feature aggregation at the bottleneck
    """

    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 num_layers: int = 4,
                 num_classes: int = 1,
                 norm: str = 'SyncBN',
                 use_attention_gates: bool = True,
                 use_se: bool = True,
                 deep_supervision: bool = True,
                 deep_supervision_weights: Tuple[float, ...] = (0.5, 0.25, 0.125)):
        """
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels (doubled at each level)
            num_layers: Number of encoder/decoder levels
            num_classes: Number of output classes
            norm: Normalization type
            use_attention_gates: Whether to use attention gates in skip connections
            use_se: Whether to use SE blocks
            deep_supervision: Whether to use deep supervision
            deep_supervision_weights: Weights for deep supervision losses
        """
        super().__init__()

        self.num_layers = num_layers
        self.deep_supervision = deep_supervision
        self.deep_supervision_weights = deep_supervision_weights

        # Calculate channel sizes for each level
        channels = [base_channels * (2 ** i) for i in range(num_layers + 1)]

        # Stem (initial convolution)
        self.stem = nn.Sequential(
            Conv3d(in_channels, base_channels // 2, kernel_size=3, padding=1),
            get_norm_3d(norm, base_channels // 2),
            nn.ReLU(inplace=True),
            Conv3d(base_channels // 2, base_channels, kernel_size=3, padding=1),
            get_norm_3d(norm, base_channels),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.encoders = nn.ModuleList()
        for i in range(num_layers):
            self.encoders.append(
                EncoderBlock(channels[i], channels[i + 1], norm, use_se,
                           downsample=(i < num_layers - 1))
            )

        # Bottleneck with multi-scale attention
        self.bottleneck = nn.Sequential(
            ResidualConvBlock3D(channels[num_layers], channels[num_layers], norm, use_se),
            MultiScaleAttention3D(channels[num_layers], scales=(1, 2)),
            ResidualConvBlock3D(channels[num_layers], channels[num_layers], norm, use_se),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            in_ch = channels[i + 1]
            skip_ch = channels[i + 1] if i == num_layers - 1 else channels[i + 1]
            out_ch = channels[i]

            # Special handling for first decoder (connects to bottleneck)
            if i == num_layers - 1:
                skip_ch = channels[i]

            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, out_ch, norm, use_attention_gates, use_se)
            )

        # Final segmentation head
        self.seg_head = nn.Sequential(
            Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
            get_norm_3d(norm, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, num_classes, kernel_size=1),
        )

        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList()
            for i in range(min(3, num_layers - 1)):
                self.ds_heads.append(
                    DeepSupervisionHead(channels[i + 1], num_classes)
                )

        # Output channels for feature extraction (for centerline deformation)
        self._out_feature_channels = base_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional deep supervision outputs.

        Args:
            x: Input tensor (B, C, D, H, W)

        Returns:
            Dictionary containing:
                - 'seg': Final segmentation output
                - 'features': Features for downstream tasks (centerline deformation)
                - 'ds_outputs': Deep supervision outputs (if enabled)
        """
        original_size = x.shape[2:]

        # Stem
        x = self.stem(x)

        # Encoder path with skip connections
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with deep supervision
        ds_outputs = []
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 1 - i
            x = decoder(x, skips[skip_idx])

            # Collect deep supervision outputs
            if self.deep_supervision and i < len(self.ds_heads):
                ds_out = self.ds_heads[i](x, original_size)
                ds_outputs.append(ds_out)

        # Final output
        features = x
        seg_output = self.seg_head(x)

        # Resize to original size if needed
        if seg_output.shape[2:] != original_size:
            seg_output = F.interpolate(seg_output, size=original_size,
                                       mode='trilinear', align_corners=False)

        outputs = {
            'seg': seg_output,
            'features': features,
        }

        if self.deep_supervision and ds_outputs:
            outputs['ds_outputs'] = ds_outputs

        return outputs

    @property
    def out_feature_channels(self) -> int:
        return self._out_feature_channels


@BACKBONE_REGISTRY.register()
def build_unet_improved_backbone(cfg, input_shape=None):
    """
    Build improved U-Net backbone from config.
    """
    in_channels = 1  # Grayscale medical images
    base_channels = cfg.MODEL.UNETENCODER.BASE_CHANNELS
    num_layers = cfg.MODEL.UNETENCODER.NUM_LAYERS
    norm = cfg.MODEL.UNETENCODER.NORM

    # New config options with defaults
    use_attention_gates = getattr(cfg.MODEL.UNETENCODER, 'USE_ATTENTION_GATES', True)
    use_se = getattr(cfg.MODEL.UNETENCODER, 'USE_SE', True)
    deep_supervision = getattr(cfg.MODEL.UNETENCODER, 'DEEP_SUPERVISION', True)

    model = UNetImproved(
        in_channels=in_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        num_classes=1,
        norm=norm,
        use_attention_gates=use_attention_gates,
        use_se=use_se,
        deep_supervision=deep_supervision,
    )

    return model


class UNetImprovedWithThinVesselLoss(nn.Module):
    """
    Wrapper that adds thin vessel loss computation to UNet.
    """

    def __init__(self, backbone: UNetImproved, thin_vessel_loss_cfg: dict):
        super().__init__()
        self.backbone = backbone

        # Import here to avoid circular imports
        from ..layers.thin_vessel_loss import ThinVesselLoss

        self.thin_vessel_loss = ThinVesselLoss(
            dice_weight=thin_vessel_loss_cfg.get('dice_weight', 0.3),
            cldice_weight=thin_vessel_loss_cfg.get('cldice_weight', 0.3),
            focal_weight=thin_vessel_loss_cfg.get('focal_weight', 0.2),
            boundary_weight=thin_vessel_loss_cfg.get('boundary_weight', 0.1),
            multiscale_weight=thin_vessel_loss_cfg.get('multiscale_weight', 0.1),
        )

        self.ds_weight = thin_vessel_loss_cfg.get('deep_supervision_weight', 0.3)

    def forward(self, x: torch.Tensor,
                target: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with loss computation.

        Args:
            x: Input tensor
            target: Ground truth segmentation (for training)

        Returns:
            Dictionary with 'seg', 'features', and optionally 'loss_dict'
        """
        outputs = self.backbone(x)

        if target is not None:
            # Apply sigmoid to get probabilities
            seg_prob = torch.sigmoid(outputs['seg'])

            # Compute main loss
            total_loss, loss_dict = self.thin_vessel_loss(seg_prob, target)

            # Add deep supervision loss
            if 'ds_outputs' in outputs and outputs['ds_outputs']:
                ds_loss = 0.0
                for ds_out in outputs['ds_outputs']:
                    ds_prob = torch.sigmoid(ds_out)
                    ds_l, _ = self.thin_vessel_loss(ds_prob, target)
                    ds_loss += ds_l
                ds_loss /= len(outputs['ds_outputs'])
                loss_dict['loss_ds'] = ds_loss
                total_loss = (1 - self.ds_weight) * total_loss + self.ds_weight * ds_loss

            loss_dict['loss_total'] = total_loss
            outputs['loss_dict'] = loss_dict

        return outputs
