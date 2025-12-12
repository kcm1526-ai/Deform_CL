"""
MedNeXt Backbone for 3D Medical Image Segmentation
Based on: https://github.com/MIC-DKFZ/MedNeXt (MICCAI 2023)

Supports:
- Multiple model sizes (S, B, M, L)
- Kernel size 3 or 5
- Deep supervision
- Pretrained weight loading (nnUNet format)
- Backbone freezing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple
import fvcore.nn.weight_init as weight_init

from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from ..layers.conv_blocks import ShapeSpec3d
from .mednext_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock
)


# Model configurations: (n_channels, exp_r, block_counts)
# exp_r can be int (same for all) or list (per-level)
MODEL_CONFIGS = {
    'S': {
        'n_channels': 32,
        'exp_r': 2,
        'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2],
        'checkpoint_style': None,
    },
    'B': {
        'n_channels': 32,
        'exp_r': [2, 3, 4, 4, 4, 4, 4, 3, 2],
        'block_counts': [2, 2, 2, 2, 2, 2, 2, 2, 2],
        'checkpoint_style': None,
    },
    'M': {
        'n_channels': 32,
        'exp_r': [2, 3, 4, 4, 4, 4, 4, 3, 2],
        'block_counts': [3, 4, 4, 4, 4, 4, 4, 4, 3],
        'checkpoint_style': 'outside_block',
    },
    'L': {
        'n_channels': 32,
        'exp_r': [3, 4, 8, 8, 8, 8, 8, 4, 3],
        'block_counts': [3, 4, 8, 8, 8, 8, 8, 4, 3],
        'checkpoint_style': 'outside_block',
    },
}


class MedNeXt(nn.Module):
    """
    MedNeXt: A fully ConvNeXt encoder-decoder network for 3D medical image segmentation.

    Architecture:
    - Encoder with 4 downsampling stages
    - Bottleneck
    - Decoder with 4 upsampling stages
    - Skip connections
    - Optional deep supervision
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_channels: int = 32,
        n_classes: int = 1,
        exp_r: Union[int, List[int]] = 2,
        kernel_size: int = 3,
        deep_supervision: bool = True,
        do_res: bool = True,
        do_res_up_down: bool = True,
        block_counts: List[int] = None,
        norm_type: str = "group",
        grn: bool = False,
        dim: int = 3
    ):
        super().__init__()

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        self.deep_supervision = deep_supervision
        self.n_classes = n_classes
        self.dim = dim

        # Handle expansion ratio (can be int or list)
        if isinstance(exp_r, int):
            exp_r = [exp_r] * 9  # 4 encoder + 1 bottleneck + 4 decoder

        # Channel progression for encoder/decoder
        # Stem -> 32, Down1 -> 64, Down2 -> 128, Down3 -> 256, Bottleneck -> 512
        self.enc_channels = [
            n_channels,
            n_channels * 2,
            n_channels * 4,
            n_channels * 8,
            n_channels * 16
        ]
        self.dec_channels = self.enc_channels[::-1]

        # Stem block
        if dim == 3:
            conv_layer = nn.Conv3d
        else:
            conv_layer = nn.Conv2d

        self.stem = conv_layer(
            in_channels,
            n_channels,
            kernel_size=1,
            bias=True
        )

        # Encoder stages (each has blocks + downsampling)
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels, n_channels,
                exp_r=exp_r[0], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[0])
        ])
        self.down_0 = MedNeXtDownBlock(
            n_channels, n_channels * 2,
            exp_r=exp_r[0], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 2, n_channels * 2,
                exp_r=exp_r[1], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[1])
        ])
        self.down_1 = MedNeXtDownBlock(
            n_channels * 2, n_channels * 4,
            exp_r=exp_r[1], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 4, n_channels * 4,
                exp_r=exp_r[2], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[2])
        ])
        self.down_2 = MedNeXtDownBlock(
            n_channels * 4, n_channels * 8,
            exp_r=exp_r[2], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 8, n_channels * 8,
                exp_r=exp_r[3], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[3])
        ])
        self.down_3 = MedNeXtDownBlock(
            n_channels * 8, n_channels * 16,
            exp_r=exp_r[3], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 16, n_channels * 16,
                exp_r=exp_r[4], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[4])
        ])

        # Decoder stages
        self.up_3 = MedNeXtUpBlock(
            n_channels * 16, n_channels * 8,
            exp_r=exp_r[5], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 8, n_channels * 8,
                exp_r=exp_r[5], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(
            n_channels * 8, n_channels * 4,
            exp_r=exp_r[6], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 4, n_channels * 4,
                exp_r=exp_r[6], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock(
            n_channels * 4, n_channels * 2,
            exp_r=exp_r[7], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels * 2, n_channels * 2,
                exp_r=exp_r[7], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock(
            n_channels * 2, n_channels,
            exp_r=exp_r[8], kernel_size=kernel_size,
            do_res=do_res_up_down, norm_type=norm_type, grn=grn, dim=dim
        )
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                n_channels, n_channels,
                exp_r=exp_r[8], kernel_size=kernel_size,
                do_res=do_res, norm_type=norm_type, grn=grn, dim=dim
            ) for _ in range(block_counts[8])
        ])

        # Output block
        self.out_block = OutBlock(n_channels, n_classes, dim=dim)

        # Deep supervision outputs (for auxiliary losses)
        if deep_supervision:
            self.ds_out_3 = OutBlock(n_channels * 8, n_classes, dim=dim)
            self.ds_out_2 = OutBlock(n_channels * 4, n_classes, dim=dim)
            self.ds_out_1 = OutBlock(n_channels * 2, n_classes, dim=dim)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # Encoder
        x = self.stem(x)

        # Stage 0
        x0 = self.enc_block_0(x)
        x = self.down_0(x0)

        # Stage 1
        x1 = self.enc_block_1(x)
        x = self.down_1(x1)

        # Stage 2
        x2 = self.enc_block_2(x)
        x = self.down_2(x2)

        # Stage 3
        x3 = self.enc_block_3(x)
        x = self.down_3(x3)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up_3(x)
        x = x + x3  # Skip connection
        x = self.dec_block_3(x)
        ds3 = x

        x = self.up_2(x)
        x = x + x2
        x = self.dec_block_2(x)
        ds2 = x

        x = self.up_1(x)
        x = x + x1
        x = self.dec_block_1(x)
        ds1 = x

        x = self.up_0(x)
        x = x + x0
        x = self.dec_block_0(x)

        # Output
        out = self.out_block(x)

        if self.deep_supervision and self.training:
            return {
                'seg': out,
                'ds3': self.ds_out_3(ds3),
                'ds2': self.ds_out_2(ds2),
                'ds1': self.ds_out_1(ds1),
            }

        return out

    def get_encoder_features(self) -> Dict[str, torch.Tensor]:
        """Return intermediate encoder features for use in other heads."""
        return self._encoder_features


class MedNeXtBackbone(Backbone):
    """
    MedNeXt backbone compatible with Detectron2's backbone interface.
    Supports pretrained weight loading and freezing.
    """

    def __init__(
        self,
        cfg,
        input_shape,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False
    ):
        super().__init__()

        in_channels = input_shape.channels
        norm = cfg.MODEL.UNETENCODER.NORM
        base_channels = cfg.MODEL.UNETENCODER.BASE_CHANNELS
        num_layers = cfg.MODEL.UNETENCODER.NUM_LAYERS

        # MedNeXt config from cfg
        model_size = getattr(cfg.MODEL, 'MEDNEXT_SIZE', 'S')
        kernel_size = getattr(cfg.MODEL, 'MEDNEXT_KERNEL_SIZE', 3)
        deep_supervision = cfg.MODEL.UNETENCODER.DEEP_SUPERVISION

        # Get model config
        model_cfg = MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['S'])

        # Build MedNeXt encoder-decoder
        self.mednext = MedNeXt(
            in_channels=in_channels,
            n_channels=base_channels,
            n_classes=1,  # Will be overridden by segmentor head
            exp_r=model_cfg['exp_r'],
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=model_cfg['block_counts'],
            norm_type="group",
            grn=False,
            dim=3
        )

        # Setup output features
        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        # Feature names matching UNet backbone format
        for i in range(num_layers + 1):
            stride = 2 ** i
            name = f'feat_{stride}x'
            self._out_features.append(name)
            self._out_feature_channels[name] = base_channels * (2 ** min(i, 4))
            self._out_feature_strides[name] = stride

        self.return_seg_logits = 'seg' in cfg.MODEL.TASK
        self.return_inter_feats = 'cline' in cfg.MODEL.TASK

        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_encoder()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-scale features."""
        # Get MedNeXt stem output
        x_stem = self.mednext.stem(x)

        # Encoder stages
        x0 = self.mednext.enc_block_0(x_stem)
        x_down0 = self.mednext.down_0(x0)

        x1 = self.mednext.enc_block_1(x_down0)
        x_down1 = self.mednext.down_1(x1)

        x2 = self.mednext.enc_block_2(x_down1)
        x_down2 = self.mednext.down_2(x2)

        x3 = self.mednext.enc_block_3(x_down2)
        x_down3 = self.mednext.down_3(x3)

        # Bottleneck
        x_bottleneck = self.mednext.bottleneck(x_down3)

        # Decoder with skip connections
        d3 = self.mednext.up_3(x_bottleneck)
        d3 = d3 + x3
        d3 = self.mednext.dec_block_3(d3)

        d2 = self.mednext.up_2(d3)
        d2 = d2 + x2
        d2 = self.mednext.dec_block_2(d2)

        d1 = self.mednext.up_1(d2)
        d1 = d1 + x1
        d1 = self.mednext.dec_block_1(d1)

        d0 = self.mednext.up_0(d1)
        d0 = d0 + x0
        d0 = self.mednext.dec_block_0(d0)

        # Build output dictionary matching UNet interface
        outputs = {
            'feat_16x': x_bottleneck,  # Bottleneck (deepest)
            'feat_8x': d3,
            'feat_4x': d2,
            'feat_2x': d1,
            'feat_1x': d0,  # Full resolution
        }

        if self.return_seg_logits:
            outputs['seg'] = self.mednext.out_block(d0)

        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec3d(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def load_pretrained(self, pretrained_path: str):
        """
        Load pretrained weights from nnUNet format.

        Args:
            pretrained_path: Path to model_best.model or model_best.model.pkl
        """
        path = Path(pretrained_path)

        if path.suffix == '.pkl':
            # Load pickle file (contains model config)
            with open(path, 'rb') as f:
                checkpoint_info = pickle.load(f)
            print(f"Loaded checkpoint info from {path}")
            # The actual weights are in the .model file
            model_path = path.with_suffix('')  # Remove .pkl
            if model_path.exists():
                pretrained_path = str(model_path)

        # Load the model weights
        print(f"Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'network_weights' in checkpoint:
                state_dict = checkpoint['network_weights']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Try to load weights with flexible key matching
        model_state = self.mednext.state_dict()
        matched_keys = []
        unmatched_keys = []

        for key in state_dict.keys():
            # Try direct match
            if key in model_state:
                if state_dict[key].shape == model_state[key].shape:
                    model_state[key] = state_dict[key]
                    matched_keys.append(key)
                else:
                    unmatched_keys.append(f"{key} (shape mismatch)")
            else:
                # Try removing prefix
                for prefix in ['module.', 'mednext.', 'encoder.', 'network.']:
                    new_key = key.replace(prefix, '')
                    if new_key in model_state:
                        if state_dict[key].shape == model_state[new_key].shape:
                            model_state[new_key] = state_dict[key]
                            matched_keys.append(key)
                            break
                else:
                    unmatched_keys.append(key)

        self.mednext.load_state_dict(model_state, strict=False)

        print(f"Loaded {len(matched_keys)} pretrained parameters")
        if unmatched_keys:
            print(f"Unmatched keys: {len(unmatched_keys)}")
            if len(unmatched_keys) <= 10:
                for k in unmatched_keys:
                    print(f"  - {k}")

    def freeze_encoder(self):
        """Freeze all encoder parameters (for transfer learning)."""
        # Freeze stem
        for param in self.mednext.stem.parameters():
            param.requires_grad = False

        # Freeze encoder blocks
        encoder_modules = [
            self.mednext.enc_block_0, self.mednext.down_0,
            self.mednext.enc_block_1, self.mednext.down_1,
            self.mednext.enc_block_2, self.mednext.down_2,
            self.mednext.enc_block_3, self.mednext.down_3,
            self.mednext.bottleneck,
        ]

        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = False

        print("Encoder frozen. Only decoder parameters will be trained.")

    def freeze(self, freeze_at: int = 0):
        """
        Freeze backbone up to a certain stage.

        Args:
            freeze_at: Stage to freeze up to (0 = no freezing)
        """
        if freeze_at >= 1:
            for param in self.mednext.stem.parameters():
                param.requires_grad = False

        stages = [
            (self.mednext.enc_block_0, self.mednext.down_0),
            (self.mednext.enc_block_1, self.mednext.down_1),
            (self.mednext.enc_block_2, self.mednext.down_2),
            (self.mednext.enc_block_3, self.mednext.down_3),
        ]

        for idx, (block, down) in enumerate(stages, start=2):
            if freeze_at >= idx:
                for param in block.parameters():
                    param.requires_grad = False
                for param in down.parameters():
                    param.requires_grad = False

        if freeze_at >= 6:
            for param in self.mednext.bottleneck.parameters():
                param.requires_grad = False

        return self

    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        print("All parameters unfrozen.")


@BACKBONE_REGISTRY.register()
def build_mednext_backbone(cfg, input_shape=None):
    """
    Build MedNeXt backbone.

    Config options:
        cfg.MODEL.MEDNEXT_SIZE: Model size ('S', 'B', 'M', 'L')
        cfg.MODEL.MEDNEXT_KERNEL_SIZE: Kernel size (3 or 5)
        cfg.MODEL.BACKBONE.FREEZE_AT: Freeze stages (0 = no freeze)
        cfg.MODEL.BACKBONE.PRETRAINED: Path to pretrained weights
        cfg.MODEL.BACKBONE.FREEZE_BACKBONE: Whether to freeze entire encoder
    """
    from ..layers.conv_blocks import ShapeSpec3d

    if input_shape is None:
        input_shape = ShapeSpec3d(channels=1)

    pretrained_path = getattr(cfg.MODEL.BACKBONE, 'PRETRAINED', None)
    freeze_backbone = getattr(cfg.MODEL.BACKBONE, 'FREEZE_BACKBONE', False)
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    backbone = MedNeXtBackbone(
        cfg,
        input_shape,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone
    )

    if freeze_at > 0 and not freeze_backbone:
        backbone.freeze(freeze_at)

    return backbone
