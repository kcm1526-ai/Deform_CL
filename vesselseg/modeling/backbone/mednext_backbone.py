"""
MedNeXt Backbone for 3D Medical Image Segmentation
Exact implementation matching: https://github.com/MIC-DKFZ/MedNeXt

Supports:
- Pretrained weight loading from nnUNet MedNeXt models (model_best.model)
- Backbone freezing for transfer learning
- Deep supervision
- Gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, Dict, Union
import pickle

from detectron2.modeling import Backbone, BACKBONE_REGISTRY

from .mednext_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock
)


class MedNeXt(nn.Module):
    """
    MedNeXt: A fully ConvNeXt encoder-decoder for 3D medical image segmentation.

    This implementation exactly matches the original MedNeXt for weight compatibility.
    Reference: https://github.com/MIC-DKFZ/MedNeXt
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_channels: int = 32,
        n_classes: int = 1,
        exp_r: Union[int, List[int]] = 4,
        kernel_size: int = 7,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: str = None,
        block_counts: List[int] = None,
        norm_type: str = 'group',
        dim: str = '3d',
        grn: bool = False
    ):
        super().__init__()

        self.do_ds = deep_supervision
        assert checkpoint_style in [None, 'outside_block']
        self.inside_block_checkpointing = False
        self.outside_block_checkpointing = False
        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True

        assert dim in ['2d', '3d']

        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        if kernel_size is not None:
            enc_kernel_size = kernel_size
            dec_kernel_size = kernel_size

        if dim == '2d':
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d

        # Stem
        self.stem = conv(in_channels, n_channels, kernel_size=1)

        # Handle expansion ratio (can be int or list)
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]

        # Encoder Block 0
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[0],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[0])
        ])

        self.down_0 = MedNeXtDownBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim
        )

        # Encoder Block 1
        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[1],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[1])
        ])

        self.down_1 = MedNeXtDownBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        # Encoder Block 2
        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[2],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[2])
        ])

        self.down_2 = MedNeXtDownBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        # Encoder Block 3
        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[3],
                kernel_size=enc_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[3])
        ])

        self.down_3 = MedNeXtDownBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=enc_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 16,
                out_channels=n_channels * 16,
                exp_r=exp_r[4],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[4])
        ])

        # Decoder
        self.up_3 = MedNeXtUpBlock(
            in_channels=16 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 8,
                out_channels=n_channels * 8,
                exp_r=exp_r[5],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(
            in_channels=8 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 4,
                out_channels=n_channels * 4,
                exp_r=exp_r[6],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock(
            in_channels=4 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels * 2,
                out_channels=n_channels * 2,
                exp_r=exp_r[7],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock(
            in_channels=2 * n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=dec_kernel_size,
            do_res=do_res_up_down,
            norm_type=norm_type,
            dim=dim,
            grn=grn
        )

        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(
                in_channels=n_channels,
                out_channels=n_channels,
                exp_r=exp_r[8],
                kernel_size=dec_kernel_size,
                do_res=do_res,
                norm_type=norm_type,
                dim=dim,
                grn=grn
            )
            for _ in range(block_counts[8])
        ])

        # Output blocks
        self.out_0 = OutBlock(in_channels=n_channels, n_classes=n_classes, dim=dim)

        # Dummy tensor for gradient checkpointing bug workaround
        self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

        # Deep supervision outputs
        if deep_supervision:
            self.out_1 = OutBlock(in_channels=n_channels * 2, n_classes=n_classes, dim=dim)
            self.out_2 = OutBlock(in_channels=n_channels * 4, n_classes=n_classes, dim=dim)
            self.out_3 = OutBlock(in_channels=n_channels * 8, n_classes=n_classes, dim=dim)
            self.out_4 = OutBlock(in_channels=n_channels * 16, n_classes=n_classes, dim=dim)

        self.block_counts = block_counts

    def iterative_checkpoint(self, sequential_block, x):
        """Forward through sequential block with gradient checkpointing."""
        for layer in sequential_block:
            x = checkpoint.checkpoint(layer, x, self.dummy_tensor, use_reentrant=False)
        return x

    def _match_size(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Match upsampled tensor size to skip connection size."""
        up_size = x_up.shape[2:]
        skip_size = x_skip.shape[2:]

        if up_size == skip_size:
            return x_up

        # Use interpolation to match sizes
        return torch.nn.functional.interpolate(
            x_up, size=skip_size, mode='trilinear', align_corners=False
        )

    def forward(self, x):
        x = self.stem(x)

        if self.outside_block_checkpointing:
            x_res_0 = self.iterative_checkpoint(self.enc_block_0, x)
            x = checkpoint.checkpoint(self.down_0, x_res_0, self.dummy_tensor, use_reentrant=False)
            x_res_1 = self.iterative_checkpoint(self.enc_block_1, x)
            x = checkpoint.checkpoint(self.down_1, x_res_1, self.dummy_tensor, use_reentrant=False)
            x_res_2 = self.iterative_checkpoint(self.enc_block_2, x)
            x = checkpoint.checkpoint(self.down_2, x_res_2, self.dummy_tensor, use_reentrant=False)
            x_res_3 = self.iterative_checkpoint(self.enc_block_3, x)
            x = checkpoint.checkpoint(self.down_3, x_res_3, self.dummy_tensor, use_reentrant=False)

            x = self.iterative_checkpoint(self.bottleneck, x)
            if self.do_ds:
                x_ds_4 = checkpoint.checkpoint(self.out_4, x, self.dummy_tensor, use_reentrant=False)

            x_up_3 = checkpoint.checkpoint(self.up_3, x, self.dummy_tensor, use_reentrant=False)
            x_up_3 = self._match_size(x_up_3, x_res_3)
            dec_x = x_res_3 + x_up_3
            x = self.iterative_checkpoint(self.dec_block_3, dec_x)
            if self.do_ds:
                x_ds_3 = checkpoint.checkpoint(self.out_3, x, self.dummy_tensor, use_reentrant=False)
            del x_res_3, x_up_3

            x_up_2 = checkpoint.checkpoint(self.up_2, x, self.dummy_tensor, use_reentrant=False)
            x_up_2 = self._match_size(x_up_2, x_res_2)
            dec_x = x_res_2 + x_up_2
            x = self.iterative_checkpoint(self.dec_block_2, dec_x)
            if self.do_ds:
                x_ds_2 = checkpoint.checkpoint(self.out_2, x, self.dummy_tensor, use_reentrant=False)
            del x_res_2, x_up_2

            x_up_1 = checkpoint.checkpoint(self.up_1, x, self.dummy_tensor, use_reentrant=False)
            x_up_1 = self._match_size(x_up_1, x_res_1)
            dec_x = x_res_1 + x_up_1
            x = self.iterative_checkpoint(self.dec_block_1, dec_x)
            if self.do_ds:
                x_ds_1 = checkpoint.checkpoint(self.out_1, x, self.dummy_tensor, use_reentrant=False)
            del x_res_1, x_up_1

            x_up_0 = checkpoint.checkpoint(self.up_0, x, self.dummy_tensor, use_reentrant=False)
            x_up_0 = self._match_size(x_up_0, x_res_0)
            dec_x = x_res_0 + x_up_0
            x = self.iterative_checkpoint(self.dec_block_0, dec_x)
            del x_res_0, x_up_0, dec_x

            x = checkpoint.checkpoint(self.out_0, x, self.dummy_tensor, use_reentrant=False)
        else:
            # Standard forward pass
            x_res_0 = self.enc_block_0(x)
            x = self.down_0(x_res_0)
            x_res_1 = self.enc_block_1(x)
            x = self.down_1(x_res_1)
            x_res_2 = self.enc_block_2(x)
            x = self.down_2(x_res_2)
            x_res_3 = self.enc_block_3(x)
            x = self.down_3(x_res_3)

            x = self.bottleneck(x)
            if self.do_ds:
                x_ds_4 = self.out_4(x)

            x_up_3 = self.up_3(x)
            x_up_3 = self._match_size(x_up_3, x_res_3)
            dec_x = x_res_3 + x_up_3
            x = self.dec_block_3(dec_x)
            if self.do_ds:
                x_ds_3 = self.out_3(x)
            del x_res_3, x_up_3

            x_up_2 = self.up_2(x)
            x_up_2 = self._match_size(x_up_2, x_res_2)
            dec_x = x_res_2 + x_up_2
            x = self.dec_block_2(dec_x)
            if self.do_ds:
                x_ds_2 = self.out_2(x)
            del x_res_2, x_up_2

            x_up_1 = self.up_1(x)
            x_up_1 = self._match_size(x_up_1, x_res_1)
            dec_x = x_res_1 + x_up_1
            x = self.dec_block_1(dec_x)
            if self.do_ds:
                x_ds_1 = self.out_1(x)
            del x_res_1, x_up_1

            x_up_0 = self.up_0(x)
            x_up_0 = self._match_size(x_up_0, x_res_0)
            dec_x = x_res_0 + x_up_0
            x = self.dec_block_0(dec_x)
            del x_res_0, x_up_0, dec_x

            x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        else:
            return x


def load_pretrained_mednext(
    model: MedNeXt,
    pretrained_path: str,
    strict: bool = False
) -> MedNeXt:
    """
    Load pretrained weights from nnUNet MedNeXt format.

    Args:
        model: MedNeXt model instance
        pretrained_path: Path to model_best.model file
        strict: Whether to strictly enforce key matching

    Returns:
        Model with loaded weights
    """
    print(f"Loading pretrained weights from: {pretrained_path}")

    # Load checkpoint
    saved_model = torch.load(pretrained_path, map_location='cpu')

    # Get state dict
    if isinstance(saved_model, dict):
        if 'state_dict' in saved_model:
            state_dict = saved_model['state_dict']
        elif 'network_weights' in saved_model:
            state_dict = saved_model['network_weights']
        elif 'model' in saved_model:
            state_dict = saved_model['model']
        else:
            state_dict = saved_model
    else:
        state_dict = saved_model

    # Process keys (remove 'module.' prefix if present)
    curr_state_dict_keys = list(model.state_dict().keys())
    new_state_dict = OrderedDict()

    matched = 0
    unmatched = []

    for k, value in state_dict.items():
        key = k
        # Remove 'module.' prefix if present (from DataParallel)
        if key not in curr_state_dict_keys and key.startswith('module.'):
            key = key[7:]

        if key in curr_state_dict_keys:
            # Check shape compatibility
            if value.shape == model.state_dict()[key].shape:
                new_state_dict[key] = value
                matched += 1
            else:
                unmatched.append(f"{key} (shape mismatch: {value.shape} vs {model.state_dict()[key].shape})")
        else:
            unmatched.append(key)

    # Load weights
    model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {matched}/{len(state_dict)} pretrained parameters")
    if unmatched:
        print(f"Unmatched keys ({len(unmatched)}):")
        for k in unmatched[:10]:  # Show first 10
            print(f"  - {k}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    return model


class MedNeXtBackbone(Backbone):
    """
    MedNeXt backbone compatible with Detectron2's backbone interface.
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
        base_channels = cfg.MODEL.UNETENCODER.BASE_CHANNELS
        num_layers = cfg.MODEL.UNETENCODER.NUM_LAYERS
        deep_supervision = cfg.MODEL.UNETENCODER.DEEP_SUPERVISION

        # MedNeXt config
        kernel_size = getattr(cfg.MODEL, 'MEDNEXT_KERNEL_SIZE', 3)
        n_classes = 1  # Binary segmentation output

        # Build MedNeXt - exact parameters for MedNeXt-S kernel3
        self.mednext = MedNeXt(
            in_channels=in_channels,
            n_channels=base_channels,
            n_classes=n_classes,
            exp_r=2,
            kernel_size=kernel_size,
            deep_supervision=deep_supervision,
            do_res=True,
            do_res_up_down=True,
            block_counts=[2, 2, 2, 2, 2, 2, 2, 2, 2],
            norm_type='group',
            dim='3d',
            grn=False
        )

        # Setup output features for detectron2
        self._out_features = []
        self._out_feature_channels = {}
        self._out_feature_strides = {}

        # Feature outputs at different scales
        for i in range(num_layers + 1):
            stride = 2 ** i
            name = f'feat_{stride}x'
            self._out_features.append(name)
            self._out_feature_channels[name] = base_channels * (2 ** min(i, 4))
            self._out_feature_strides[name] = stride

        self.return_seg_logits = 'seg' in cfg.MODEL.TASK
        self.deep_supervision = deep_supervision
        self.n_channels = base_channels

        # Load pretrained weights if provided
        if pretrained_path and pretrained_path.strip():
            load_pretrained_mednext(self.mednext, pretrained_path)

        # Freeze backbone if requested
        if freeze_backbone:
            self.freeze_encoder()

    def _match_size(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Match upsampled tensor size to skip connection size by cropping/padding."""
        # Get spatial dimensions (last 3 dims for 3D)
        up_size = x_up.shape[2:]
        skip_size = x_skip.shape[2:]

        if up_size == skip_size:
            return x_up

        # Crop if upsampled is larger
        slices = [slice(None), slice(None)]  # batch, channel
        for i in range(3):
            if up_size[i] > skip_size[i]:
                diff = up_size[i] - skip_size[i]
                start = diff // 2
                slices.append(slice(start, start + skip_size[i]))
            elif up_size[i] < skip_size[i]:
                # Need to pad - use interpolation instead
                return torch.nn.functional.interpolate(
                    x_up, size=skip_size, mode='trilinear', align_corners=False
                )
            else:
                slices.append(slice(None))

        return x_up[tuple(slices)]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-scale features."""
        # Get intermediate features manually
        x_stem = self.mednext.stem(x)

        # Encoder
        x_res_0 = self.mednext.enc_block_0(x_stem)
        x_down_0 = self.mednext.down_0(x_res_0)

        x_res_1 = self.mednext.enc_block_1(x_down_0)
        x_down_1 = self.mednext.down_1(x_res_1)

        x_res_2 = self.mednext.enc_block_2(x_down_1)
        x_down_2 = self.mednext.down_2(x_res_2)

        x_res_3 = self.mednext.enc_block_3(x_down_2)
        x_down_3 = self.mednext.down_3(x_res_3)

        # Bottleneck
        x_bottleneck = self.mednext.bottleneck(x_down_3)

        # Decoder with skip connections (match sizes before addition)
        x_up_3 = self.mednext.up_3(x_bottleneck)
        x_up_3 = self._match_size(x_up_3, x_res_3)
        d3 = x_res_3 + x_up_3
        d3 = self.mednext.dec_block_3(d3)

        x_up_2 = self.mednext.up_2(d3)
        x_up_2 = self._match_size(x_up_2, x_res_2)
        d2 = x_res_2 + x_up_2
        d2 = self.mednext.dec_block_2(d2)

        x_up_1 = self.mednext.up_1(d2)
        x_up_1 = self._match_size(x_up_1, x_res_1)
        d1 = x_res_1 + x_up_1
        d1 = self.mednext.dec_block_1(d1)

        x_up_0 = self.mednext.up_0(d1)
        x_up_0 = self._match_size(x_up_0, x_res_0)
        d0 = x_res_0 + x_up_0
        d0 = self.mednext.dec_block_0(d0)

        # Build output dictionary
        outputs = {
            'feat_16x': x_bottleneck,
            'feat_8x': d3,
            'feat_4x': d2,
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

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        # Freeze stem
        for param in self.mednext.stem.parameters():
            param.requires_grad = False

        # Freeze encoder blocks and downsampling
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

        print("Encoder frozen. Only decoder will be trained.")

    def freeze(self, freeze_at: int = 0):
        """Freeze up to a certain stage."""
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


@BACKBONE_REGISTRY.register()
def build_mednext_backbone(cfg, input_shape=None):
    """
    Build MedNeXt backbone.

    Config:
        cfg.MODEL.MEDNEXT_KERNEL_SIZE: Kernel size (3 or 5)
        cfg.MODEL.BACKBONE.PRETRAINED: Path to pretrained weights
        cfg.MODEL.BACKBONE.FREEZE_BACKBONE: Whether to freeze encoder
        cfg.MODEL.BACKBONE.FREEZE_AT: Freeze stages (0 = no freeze)
    """
    from ..layers.conv_blocks import ShapeSpec3d

    if input_shape is None:
        input_shape = ShapeSpec3d(channels=1)

    pretrained_path = getattr(cfg.MODEL.BACKBONE, 'PRETRAINED', '')
    freeze_backbone = getattr(cfg.MODEL.BACKBONE, 'FREEZE_BACKBONE', False)
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT

    backbone = MedNeXtBackbone(
        cfg,
        input_shape,
        pretrained_path=pretrained_path if pretrained_path else None,
        freeze_backbone=freeze_backbone
    )

    if freeze_at > 0 and not freeze_backbone:
        backbone.freeze(freeze_at)

    return backbone
