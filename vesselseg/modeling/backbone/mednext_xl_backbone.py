"""
MedNeXt-XL Backbone with Full Attention for Vessel Segmentation

A large-scale MedNeXt architecture enhanced with comprehensive attention mechanisms
designed specifically for lung vessel segmentation.

Key Features:
1. Large capacity: 96 base channels, [3,4,12,12,12,12,12,4,3] blocks
2. Squeeze-and-Excitation attention in every block
3. Attention-gated skip connections
4. Self-attention at bottleneck
5. Multi-scale vessel attention
6. Edge enhancement module
7. Deep supervision
8. Gradient checkpointing for memory efficiency

Target: ~400M parameters, Dice > 0.88

Reference implementations:
- STU-Net: https://github.com/uni-medical/STU-Net
- MedNeXt: https://github.com/MIC-DKFZ/MedNeXt
- Attention U-Net: https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from typing import Optional, List, Dict, Union, Tuple
import math

from detectron2.modeling import Backbone, BACKBONE_REGISTRY

from .attention_blocks import (
    SqueezeExcitation3D,
    AttentionGate3D,
    EfficientSelfAttention3D,
    VesselAttentionBlock,
    EdgeEnhancementModule,
    MultiScaleAttention3D,
    CBAM3D
)


class MedNeXtBlockXL(nn.Module):
    """
    Enhanced MedNeXt Block with SE attention.

    Compared to original MedNeXt block:
    - Added Squeeze-and-Excitation attention
    - Larger kernel size support (up to 7)
    - GRN (Global Response Normalization) option
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        grn: bool = True,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        self.do_res = do_res

        # Depthwise convolution with large kernel
        self.conv1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=in_channels
        )

        # Normalization
        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=min(in_channels, 32),
                num_channels=in_channels
            )
        else:
            self.norm = nn.InstanceNorm3d(in_channels)

        # Expansion convolution
        self.conv2 = nn.Conv3d(
            in_channels, exp_r * in_channels,
            kernel_size=1
        )

        self.act = nn.GELU()

        # GRN (Global Response Normalization)
        self.grn = None
        if grn:
            self.grn = GRN(exp_r * in_channels)

        # Compression convolution
        self.conv3 = nn.Conv3d(
            exp_r * in_channels, out_channels,
            kernel_size=1
        )

        # SE attention
        self.se = None
        if use_se:
            self.se = SqueezeExcitation3D(out_channels, reduction=se_reduction)

        # Residual connection adjustment
        if in_channels != out_channels:
            self.res_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = None

    def forward(self, x: torch.Tensor, dummy=None) -> torch.Tensor:
        residual = x

        # Main path
        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)

        # SE attention
        if self.se is not None:
            x = self.se(x)

        # Residual connection
        if self.do_res:
            if self.res_conv is not None:
                residual = self.res_conv(residual)
            x = x + residual

        return x


class GRN(nn.Module):
    """
    Global Response Normalization layer.

    Normalizes features based on global statistics, improving
    feature diversity and model capacity.

    Reference: ConvNeXt V2
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global feature aggregation
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        # Normalization
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class MedNeXtDownBlockXL(nn.Module):
    """Downsampling block with SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        grn: bool = True
    ):
        super().__init__()

        self.do_res = do_res

        # Strided convolution for downsampling
        self.conv1 = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            groups=in_channels
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=min(in_channels, 32),
                num_channels=in_channels
            )
        else:
            self.norm = nn.InstanceNorm3d(in_channels)

        self.conv2 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1)
        self.act = nn.GELU()

        self.grn = None
        if grn:
            self.grn = GRN(exp_r * in_channels)

        self.conv3 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1)

        # SE attention
        self.se = SqueezeExcitation3D(out_channels, reduction=16)

        # Residual with pooling
        self.res_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor, dummy=None) -> torch.Tensor:
        residual = self.res_conv(x)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)
        x = self.se(x)

        if self.do_res:
            x = x + residual

        return x


class MedNeXtUpBlockXL(nn.Module):
    """Upsampling block with SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        do_res: bool = True,
        norm_type: str = 'group',
        grn: bool = True
    ):
        super().__init__()

        self.do_res = do_res

        # Transposed convolution for upsampling
        self.conv1 = nn.ConvTranspose3d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
            groups=in_channels
        )

        if norm_type == 'group':
            self.norm = nn.GroupNorm(
                num_groups=min(in_channels, 32),
                num_channels=in_channels
            )
        else:
            self.norm = nn.InstanceNorm3d(in_channels)

        self.conv2 = nn.Conv3d(in_channels, exp_r * in_channels, kernel_size=1)
        self.act = nn.GELU()

        self.grn = None
        if grn:
            self.grn = GRN(exp_r * in_channels)

        self.conv3 = nn.Conv3d(exp_r * in_channels, out_channels, kernel_size=1)

        # SE attention
        self.se = SqueezeExcitation3D(out_channels, reduction=16)

        # Residual with upsampling
        self.res_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x: torch.Tensor, dummy=None) -> torch.Tensor:
        residual = self.res_conv(x)

        x = self.conv1(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.act(x)

        if self.grn is not None:
            x = self.grn(x)

        x = self.conv3(x)
        x = self.se(x)

        if self.do_res:
            # Match sizes
            if x.shape != residual.shape:
                residual = F.interpolate(
                    residual, size=x.shape[2:],
                    mode='trilinear', align_corners=False
                )
            x = x + residual

        return x


class OutBlock(nn.Module):
    """Output block for segmentation head."""

    def __init__(self, in_channels: int, n_classes: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, dummy=None) -> torch.Tensor:
        return self.conv(x)


class MedNeXtXL(nn.Module):
    """
    MedNeXt-XL: Extra Large MedNeXt with Full Attention.

    Architecture:
    - 96 base channels (3x original)
    - [3, 4, 12, 12, 12, 12, 12, 4, 3] block counts (much deeper)
    - Expansion ratio 4
    - Kernel size 7
    - SE attention in every block
    - Attention gates at skip connections
    - Self-attention at bottleneck
    - Edge enhancement
    - Deep supervision

    Target: ~400M parameters
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_channels: int = 96,  # Large base channels
        n_classes: int = 1,
        exp_r: int = 4,
        kernel_size: int = 7,
        deep_supervision: bool = True,
        block_counts: List[int] = None,
        use_attention_gates: bool = True,
        use_self_attention: bool = True,
        use_edge_enhancement: bool = True,
        use_vessel_attention: bool = True,
        checkpoint_style: str = 'outside_block'
    ):
        super().__init__()

        self.do_ds = deep_supervision

        if block_counts is None:
            # XL configuration: much deeper
            block_counts = [3, 4, 12, 12, 12, 12, 12, 4, 3]

        self.block_counts = block_counts
        self.use_attention_gates = use_attention_gates
        self.use_checkpointing = checkpoint_style == 'outside_block'

        # Dummy tensor for gradient checkpointing
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        # Stem
        self.stem = nn.Conv3d(in_channels, n_channels, kernel_size=1)

        # ============ ENCODER ============
        # Stage 0
        self.enc_block_0 = self._make_stage(
            n_channels, n_channels, block_counts[0], exp_r, kernel_size
        )
        self.down_0 = MedNeXtDownBlockXL(
            n_channels, 2 * n_channels, exp_r, kernel_size
        )

        # Stage 1
        self.enc_block_1 = self._make_stage(
            2 * n_channels, 2 * n_channels, block_counts[1], exp_r, kernel_size
        )
        self.down_1 = MedNeXtDownBlockXL(
            2 * n_channels, 4 * n_channels, exp_r, kernel_size
        )

        # Stage 2
        self.enc_block_2 = self._make_stage(
            4 * n_channels, 4 * n_channels, block_counts[2], exp_r, kernel_size
        )
        self.down_2 = MedNeXtDownBlockXL(
            4 * n_channels, 8 * n_channels, exp_r, kernel_size
        )

        # Stage 3
        self.enc_block_3 = self._make_stage(
            8 * n_channels, 8 * n_channels, block_counts[3], exp_r, kernel_size
        )
        self.down_3 = MedNeXtDownBlockXL(
            8 * n_channels, 16 * n_channels, exp_r, kernel_size
        )

        # ============ BOTTLENECK ============
        self.bottleneck = self._make_stage(
            16 * n_channels, 16 * n_channels, block_counts[4], exp_r, kernel_size
        )

        # Self-attention at bottleneck
        if use_self_attention:
            self.bottleneck_attention = EfficientSelfAttention3D(
                16 * n_channels, num_heads=16
            )
        else:
            self.bottleneck_attention = None

        # ============ DECODER ============
        # Attention gates for skip connections
        if use_attention_gates:
            self.attn_gate_3 = AttentionGate3D(16 * n_channels, 8 * n_channels)
            self.attn_gate_2 = AttentionGate3D(8 * n_channels, 4 * n_channels)
            self.attn_gate_1 = AttentionGate3D(4 * n_channels, 2 * n_channels)
            self.attn_gate_0 = AttentionGate3D(2 * n_channels, n_channels)

        # Up Stage 3
        self.up_3 = MedNeXtUpBlockXL(
            16 * n_channels, 8 * n_channels, exp_r, kernel_size
        )
        self.dec_block_3 = self._make_stage(
            8 * n_channels, 8 * n_channels, block_counts[5], exp_r, kernel_size
        )

        # Up Stage 2
        self.up_2 = MedNeXtUpBlockXL(
            8 * n_channels, 4 * n_channels, exp_r, kernel_size
        )
        self.dec_block_2 = self._make_stage(
            4 * n_channels, 4 * n_channels, block_counts[6], exp_r, kernel_size
        )

        # Up Stage 1
        self.up_1 = MedNeXtUpBlockXL(
            4 * n_channels, 2 * n_channels, exp_r, kernel_size
        )
        self.dec_block_1 = self._make_stage(
            2 * n_channels, 2 * n_channels, block_counts[7], exp_r, kernel_size
        )

        # Up Stage 0
        self.up_0 = MedNeXtUpBlockXL(
            2 * n_channels, n_channels, exp_r, kernel_size
        )
        self.dec_block_0 = self._make_stage(
            n_channels, n_channels, block_counts[8], exp_r, kernel_size
        )

        # ============ VESSEL-SPECIFIC MODULES ============
        if use_vessel_attention:
            self.vessel_attention = VesselAttentionBlock(n_channels)
        else:
            self.vessel_attention = None

        if use_edge_enhancement:
            self.edge_enhance = EdgeEnhancementModule(n_channels)
        else:
            self.edge_enhance = None

        # ============ OUTPUT HEADS ============
        self.out_0 = OutBlock(n_channels, n_classes)

        # Deep supervision outputs
        if deep_supervision:
            self.out_1 = OutBlock(2 * n_channels, n_classes)
            self.out_2 = OutBlock(4 * n_channels, n_classes)
            self.out_3 = OutBlock(8 * n_channels, n_classes)
            self.out_4 = OutBlock(16 * n_channels, n_classes)

        # Initialize weights
        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        exp_r: int,
        kernel_size: int
    ) -> nn.Sequential:
        """Create a stage with multiple blocks."""
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MedNeXtBlockXL(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    exp_r=exp_r,
                    kernel_size=kernel_size,
                    do_res=True,
                    grn=True,
                    use_se=True
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _checkpoint_forward(self, module, x):
        """Forward with gradient checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(module, x, self.dummy_tensor, use_reentrant=False)
        return module(x)

    def _checkpoint_sequential(self, sequential, x):
        """Forward through sequential with checkpointing."""
        if self.use_checkpointing and self.training:
            for layer in sequential:
                x = checkpoint.checkpoint(layer, x, self.dummy_tensor, use_reentrant=False)
            return x
        return sequential(x)

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Stem
        x = self.stem(x)

        # Encoder
        x_res_0 = self._checkpoint_sequential(self.enc_block_0, x)
        x = self._checkpoint_forward(self.down_0, x_res_0)

        x_res_1 = self._checkpoint_sequential(self.enc_block_1, x)
        x = self._checkpoint_forward(self.down_1, x_res_1)

        x_res_2 = self._checkpoint_sequential(self.enc_block_2, x)
        x = self._checkpoint_forward(self.down_2, x_res_2)

        x_res_3 = self._checkpoint_sequential(self.enc_block_3, x)
        x = self._checkpoint_forward(self.down_3, x_res_3)

        # Bottleneck
        x = self._checkpoint_sequential(self.bottleneck, x)

        # Self-attention at bottleneck
        if self.bottleneck_attention is not None:
            x = self.bottleneck_attention(x)

        # Deep supervision output 4
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        # Decoder with attention gates
        # Stage 3
        x_up_3 = self._checkpoint_forward(self.up_3, x)
        x_up_3 = self._match_size(x_up_3, x_res_3)
        if self.use_attention_gates:
            x_res_3 = self.attn_gate_3(x_up_3, x_res_3)
        x = x_res_3 + x_up_3
        x = self._checkpoint_sequential(self.dec_block_3, x)

        if self.do_ds:
            x_ds_3 = self.out_3(x)

        # Stage 2
        x_up_2 = self._checkpoint_forward(self.up_2, x)
        x_up_2 = self._match_size(x_up_2, x_res_2)
        if self.use_attention_gates:
            x_res_2 = self.attn_gate_2(x_up_2, x_res_2)
        x = x_res_2 + x_up_2
        x = self._checkpoint_sequential(self.dec_block_2, x)

        if self.do_ds:
            x_ds_2 = self.out_2(x)

        # Stage 1
        x_up_1 = self._checkpoint_forward(self.up_1, x)
        x_up_1 = self._match_size(x_up_1, x_res_1)
        if self.use_attention_gates:
            x_res_1 = self.attn_gate_1(x_up_1, x_res_1)
        x = x_res_1 + x_up_1
        x = self._checkpoint_sequential(self.dec_block_1, x)

        if self.do_ds:
            x_ds_1 = self.out_1(x)

        # Stage 0
        x_up_0 = self._checkpoint_forward(self.up_0, x)
        x_up_0 = self._match_size(x_up_0, x_res_0)
        if self.use_attention_gates:
            x_res_0 = self.attn_gate_0(x_up_0, x_res_0)
        x = x_res_0 + x_up_0
        x = self._checkpoint_sequential(self.dec_block_0, x)

        # Vessel-specific enhancements
        if self.vessel_attention is not None:
            x = self.vessel_attention(x)

        if self.edge_enhance is not None:
            x = self.edge_enhance(x)

        # Final output
        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x


class MedNeXtXLBackbone(Backbone):
    """
    MedNeXt-XL backbone compatible with Detectron2.
    """

    def __init__(
        self,
        cfg,
        input_shape,
        pretrained_path: Optional[str] = None
    ):
        super().__init__()

        in_channels = input_shape.channels
        base_channels = cfg.MODEL.UNETENCODER.BASE_CHANNELS
        deep_supervision = cfg.MODEL.UNETENCODER.DEEP_SUPERVISION

        # XL configuration
        kernel_size = getattr(cfg.MODEL, 'MEDNEXT_KERNEL_SIZE', 7)
        use_attention_gates = getattr(cfg.MODEL, 'USE_ATTENTION_GATES', True)
        use_self_attention = getattr(cfg.MODEL, 'USE_SELF_ATTENTION', True)
        use_edge_enhancement = getattr(cfg.MODEL, 'USE_EDGE_ENHANCEMENT', True)
        use_vessel_attention = getattr(cfg.MODEL, 'USE_VESSEL_ATTENTION', True)

        # Build MedNeXt-XL
        self.mednext = MedNeXtXL(
            in_channels=in_channels,
            n_channels=base_channels,
            n_classes=1,
            exp_r=4,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            block_counts=[3, 4, 12, 12, 12, 12, 12, 4, 3],
            use_attention_gates=use_attention_gates,
            use_self_attention=use_self_attention,
            use_edge_enhancement=use_edge_enhancement,
            use_vessel_attention=use_vessel_attention,
            checkpoint_style='outside_block'
        )

        # Feature info
        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        num_layers = cfg.MODEL.UNETENCODER.NUM_LAYERS
        for i in range(num_layers + 1):
            stride = 2 ** i
            name = f'feat_{stride}x'
            self._out_features.append(name)
            self._out_feature_channels[name] = base_channels * (2 ** min(i, 4))
            self._out_feature_strides[name] = stride

        self.return_seg_logits = 'seg' in cfg.MODEL.TASK
        self.return_inter_feats = 'cline' in cfg.MODEL.TASK
        self.deep_supervision = deep_supervision
        self.n_channels = base_channels

        # Inter block for cline_deformer compatibility
        if self.return_inter_feats:
            feat_4x_channels = base_channels * 4
            inter_channels = 24
            self.inter_block = nn.Sequential(
                nn.Conv3d(feat_4x_channels, feat_4x_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(feat_4x_channels, inter_channels, kernel_size=1),
            )
        else:
            self.inter_block = None

        # Load pretrained if available
        if pretrained_path and pretrained_path.strip():
            self._load_pretrained(pretrained_path)

        # Print model info
        self._print_model_info()

    def _load_pretrained(self, path: str):
        """Load pretrained weights."""
        print(f"Loading pretrained weights from: {path}")
        try:
            state_dict = torch.load(path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # Try to load (may be partial match)
            missing, unexpected = self.mednext.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def _print_model_info(self):
        """Print model parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MedNeXt-XL Model Info:")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"  Base channels: {self.n_channels}")
        print(f"  Block counts: {self.mednext.block_counts}")

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='trilinear', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-scale features."""
        # Get features through custom forward to capture intermediates
        x_stem = self.mednext.stem(x)

        # Encoder with gradient checkpointing
        x_res_0 = self.mednext._checkpoint_sequential(self.mednext.enc_block_0, x_stem)
        x = self.mednext._checkpoint_forward(self.mednext.down_0, x_res_0)

        x_res_1 = self.mednext._checkpoint_sequential(self.mednext.enc_block_1, x)
        x = self.mednext._checkpoint_forward(self.mednext.down_1, x_res_1)

        x_res_2 = self.mednext._checkpoint_sequential(self.mednext.enc_block_2, x)
        x = self.mednext._checkpoint_forward(self.mednext.down_2, x_res_2)

        x_res_3 = self.mednext._checkpoint_sequential(self.mednext.enc_block_3, x)
        x = self.mednext._checkpoint_forward(self.mednext.down_3, x_res_3)

        # Bottleneck
        x_bottleneck = self.mednext._checkpoint_sequential(self.mednext.bottleneck, x)
        if self.mednext.bottleneck_attention is not None:
            x_bottleneck = self.mednext.bottleneck_attention(x_bottleneck)

        # Decoder
        x_up_3 = self.mednext._checkpoint_forward(self.mednext.up_3, x_bottleneck)
        x_up_3 = self._match_size(x_up_3, x_res_3)
        if self.mednext.use_attention_gates:
            x_res_3_attn = self.mednext.attn_gate_3(x_up_3, x_res_3)
        else:
            x_res_3_attn = x_res_3
        d3 = x_res_3_attn + x_up_3
        d3 = self.mednext._checkpoint_sequential(self.mednext.dec_block_3, d3)

        x_up_2 = self.mednext._checkpoint_forward(self.mednext.up_2, d3)
        x_up_2 = self._match_size(x_up_2, x_res_2)
        if self.mednext.use_attention_gates:
            x_res_2_attn = self.mednext.attn_gate_2(x_up_2, x_res_2)
        else:
            x_res_2_attn = x_res_2
        d2 = x_res_2_attn + x_up_2
        d2 = self.mednext._checkpoint_sequential(self.mednext.dec_block_2, d2)

        x_up_1 = self.mednext._checkpoint_forward(self.mednext.up_1, d2)
        x_up_1 = self._match_size(x_up_1, x_res_1)
        if self.mednext.use_attention_gates:
            x_res_1_attn = self.mednext.attn_gate_1(x_up_1, x_res_1)
        else:
            x_res_1_attn = x_res_1
        d1 = x_res_1_attn + x_up_1
        d1 = self.mednext._checkpoint_sequential(self.mednext.dec_block_1, d1)

        x_up_0 = self.mednext._checkpoint_forward(self.mednext.up_0, d1)
        x_up_0 = self._match_size(x_up_0, x_res_0)
        if self.mednext.use_attention_gates:
            x_res_0_attn = self.mednext.attn_gate_0(x_up_0, x_res_0)
        else:
            x_res_0_attn = x_res_0
        d0 = x_res_0_attn + x_up_0
        d0 = self.mednext._checkpoint_sequential(self.mednext.dec_block_0, d0)

        # Vessel-specific enhancements
        if self.mednext.vessel_attention is not None:
            d0 = self.mednext.vessel_attention(d0)
        if self.mednext.edge_enhance is not None:
            d0 = self.mednext.edge_enhance(d0)

        # Build output dictionary
        feat_4x_out = self.inter_block(d2) if self.inter_block is not None else d2

        outputs = {
            'feat_16x': x_bottleneck,
            'feat_8x': d3,
            'feat_4x': feat_4x_out,
            'feat_2x': d1,
            'feat_1x': d0,
        }

        if self.return_seg_logits:
            outputs['seg'] = self.mednext.out_0(d0)

        return outputs

    def output_shape(self):
        from ..layers.conv_blocks import ShapeSpec3d
        return {
            name: ShapeSpec3d(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_mednext_xl_backbone(cfg, input_shape=None):
    """
    Build MedNeXt-XL backbone.

    Config options:
        cfg.MODEL.UNETENCODER.BASE_CHANNELS: Base channels (default: 96 for XL)
        cfg.MODEL.MEDNEXT_KERNEL_SIZE: Kernel size (default: 7)
        cfg.MODEL.USE_ATTENTION_GATES: Use attention gates (default: True)
        cfg.MODEL.USE_SELF_ATTENTION: Use self-attention at bottleneck (default: True)
        cfg.MODEL.USE_EDGE_ENHANCEMENT: Use edge enhancement (default: True)
        cfg.MODEL.USE_VESSEL_ATTENTION: Use vessel attention (default: True)
    """
    from ..layers.conv_blocks import ShapeSpec3d

    if input_shape is None:
        input_shape = ShapeSpec3d(channels=1)

    pretrained_path = getattr(cfg.MODEL.BACKBONE, 'PRETRAINED', '')

    return MedNeXtXLBackbone(
        cfg,
        input_shape,
        pretrained_path=pretrained_path if pretrained_path else None
    )
