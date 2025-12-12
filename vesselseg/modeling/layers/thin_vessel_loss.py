"""
Thin Vessel Loss Functions for improved vessel segmentation.

This module contains specialized loss functions designed for thin tubular
structure segmentation, addressing the challenges of:
1. Thin vessel disappearance
2. Topology/connectivity preservation
3. Boundary accuracy

Key loss functions:
- clDice (centerline Dice): Preserves vessel connectivity
- Focal Dice: Focuses on hard-to-segment thin vessels
- Boundary Loss: Improves boundary accuracy
- Multi-scale Loss: Captures vessels at different scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SoftSkeletonize3D(nn.Module):
    """
    Differentiable soft skeletonization for 3D volumes.

    Uses iterative morphological erosion to approximate the skeleton/centerline.
    This is crucial for clDice loss computation.
    """

    def __init__(self, num_iterations: int = 10):
        super().__init__()
        self.num_iterations = num_iterations

        # 3D erosion kernel (3x3x3 minimum filter approximation)
        kernel = torch.ones(1, 1, 3, 3, 3) / 27.0
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute soft skeleton of input segmentation.

        Args:
            x: Input tensor of shape (B, 1, D, H, W) with values in [0, 1]

        Returns:
            Soft skeleton tensor of same shape
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)

        skeleton = x.clone()

        for _ in range(self.num_iterations):
            # Soft erosion using average pooling
            eroded = F.avg_pool3d(skeleton, kernel_size=3, stride=1, padding=1)
            # Threshold to keep only persistent structures
            eroded = torch.clamp(eroded * 1.5 - 0.25, 0, 1)

            # Update skeleton: keep pixels that survive erosion
            skeleton = skeleton * eroded

        return skeleton


class ClDiceLoss(nn.Module):
    """
    Centerline Dice Loss (clDice) for tubular structure segmentation.

    Reference: "clDice - a Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation" (Shit et al., CVPR 2021)

    clDice = 2 * (|S_pred ∩ V_gt| + |S_gt ∩ V_pred|) / (|S_pred| + |S_gt|)

    Where S is the soft skeleton and V is the volume.
    """

    def __init__(self, num_iterations: int = 10, alpha: float = 0.5, smooth: float = 1e-5):
        """
        Args:
            num_iterations: Number of erosion iterations for skeletonization
            alpha: Weight between Dice and clDice (0=pure Dice, 1=pure clDice)
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.skeletonize = SoftSkeletonize3D(num_iterations)
        self.alpha = alpha
        self.smooth = smooth

    def soft_dice(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute soft Dice coefficient."""
        intersection = (pred * target).sum()
        return (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute clDice loss.

        Args:
            pred: Predicted segmentation (B, 1, D, H, W) or (B, D, H, W), values in [0, 1]
            target: Ground truth segmentation, same shape as pred

        Returns:
            clDice loss value (1 - clDice)
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # Compute soft skeletons
        skel_pred = self.skeletonize(pred)
        skel_target = self.skeletonize(target)

        # Topology precision: How much of predicted skeleton is in GT volume
        tprec = (skel_pred * target).sum() / (skel_pred.sum() + self.smooth)

        # Topology sensitivity: How much of GT skeleton is in predicted volume
        tsens = (skel_target * pred).sum() / (skel_target.sum() + self.smooth)

        # clDice coefficient
        cl_dice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)

        # Standard Dice coefficient
        dice = self.soft_dice(pred, target)

        # Combined loss
        combined_dice = (1 - self.alpha) * dice + self.alpha * cl_dice

        return 1.0 - combined_dice


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss that focuses on hard-to-segment regions (thin vessels).

    Combines Focal Loss weighting with Dice to emphasize difficult samples.
    """

    def __init__(self, gamma: float = 2.0, smooth: float = 1e-5):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            smooth: Smoothing factor
        """
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Dice loss.

        Args:
            pred: Predicted segmentation (B, 1, D, H, W), values in [0, 1]
            target: Ground truth segmentation

        Returns:
            Focal Dice loss value
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # Compute per-voxel weights based on prediction confidence
        # Hard examples have predictions close to 0.5
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Weighted intersection and union
        intersection = (focal_weight * pred * target).sum()
        pred_sum = (focal_weight * pred).sum()
        target_sum = (focal_weight * target).sum()

        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that emphasizes vessel boundaries.

    Uses distance transform to weight boundary regions more heavily.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

        # Sobel-like 3D gradient kernels
        self.register_buffer('sobel_x', self._create_sobel_kernel(0))
        self.register_buffer('sobel_y', self._create_sobel_kernel(1))
        self.register_buffer('sobel_z', self._create_sobel_kernel(2))

    def _create_sobel_kernel(self, axis: int) -> torch.Tensor:
        """Create 3D Sobel kernel for gradient computation."""
        kernel = torch.zeros(1, 1, 3, 3, 3)
        if axis == 0:  # z gradient
            kernel[0, 0, 0, 1, 1] = -1
            kernel[0, 0, 2, 1, 1] = 1
        elif axis == 1:  # y gradient
            kernel[0, 0, 1, 0, 1] = -1
            kernel[0, 0, 1, 2, 1] = 1
        else:  # x gradient
            kernel[0, 0, 1, 1, 0] = -1
            kernel[0, 0, 1, 1, 2] = 1
        return kernel

    def compute_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Compute soft boundary map using gradients."""
        if x.dim() == 4:
            x = x.unsqueeze(1)

        # Compute gradients
        gx = F.conv3d(x, self.sobel_x, padding=1)
        gy = F.conv3d(x, self.sobel_y, padding=1)
        gz = F.conv3d(x, self.sobel_z, padding=1)

        # Gradient magnitude
        boundary = torch.sqrt(gx**2 + gy**2 + gz**2 + self.smooth)

        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary loss.

        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation

        Returns:
            Boundary loss value
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # Compute boundaries
        pred_boundary = self.compute_boundary(pred)
        target_boundary = self.compute_boundary(target)

        # Boundary Dice
        intersection = (pred_boundary * target_boundary).sum()
        boundary_dice = (2.0 * intersection + self.smooth) / \
                       (pred_boundary.sum() + target_boundary.sum() + self.smooth)

        return 1.0 - boundary_dice


class MultiScaleLoss(nn.Module):
    """
    Multi-scale loss for capturing vessels at different sizes.

    Applies loss at multiple resolutions to handle both thick and thin vessels.
    """

    def __init__(self, scales: Tuple[float, ...] = (1.0, 0.5, 0.25),
                 weights: Optional[Tuple[float, ...]] = None,
                 smooth: float = 1e-5):
        """
        Args:
            scales: Downsampling factors for each scale
            weights: Weight for each scale (default: equal weights)
            smooth: Smoothing factor
        """
        super().__init__()
        self.scales = scales
        self.weights = weights if weights else tuple(1.0 / len(scales) for _ in scales)
        self.smooth = smooth

    def dice_at_scale(self, pred: torch.Tensor, target: torch.Tensor,
                      scale: float) -> torch.Tensor:
        """Compute Dice at a specific scale."""
        if scale < 1.0:
            # Downsample
            pred = F.interpolate(pred, scale_factor=scale, mode='trilinear',
                               align_corners=False)
            target = F.interpolate(target.float(), scale_factor=scale, mode='trilinear',
                                  align_corners=False)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale Dice loss.

        Args:
            pred: Predicted segmentation (B, 1, D, H, W)
            target: Ground truth segmentation

        Returns:
            Multi-scale Dice loss value
        """
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        total_dice = 0.0
        for scale, weight in zip(self.scales, self.weights):
            dice = self.dice_at_scale(pred, target, scale)
            total_dice += weight * dice

        return 1.0 - total_dice


class ThinVesselLoss(nn.Module):
    """
    Combined loss function optimized for thin vessel segmentation.

    Combines multiple losses:
    - Standard Dice for overall accuracy
    - clDice for topology preservation
    - Focal Dice for hard examples
    - Boundary loss for edge accuracy
    - Multi-scale loss for vessels of different sizes
    """

    def __init__(self,
                 dice_weight: float = 0.3,
                 cldice_weight: float = 0.3,
                 focal_weight: float = 0.2,
                 boundary_weight: float = 0.1,
                 multiscale_weight: float = 0.1,
                 cldice_alpha: float = 0.5,
                 focal_gamma: float = 2.0,
                 smooth: float = 1e-5):
        """
        Args:
            dice_weight: Weight for standard Dice loss
            cldice_weight: Weight for clDice loss
            focal_weight: Weight for Focal Dice loss
            boundary_weight: Weight for boundary loss
            multiscale_weight: Weight for multi-scale loss
            cldice_alpha: Alpha parameter for clDice
            focal_gamma: Gamma parameter for Focal loss
            smooth: Smoothing factor
        """
        super().__init__()

        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.multiscale_weight = multiscale_weight
        self.smooth = smooth

        # Initialize component losses
        self.cldice_loss = ClDiceLoss(alpha=cldice_alpha, smooth=smooth)
        self.focal_loss = FocalDiceLoss(gamma=focal_gamma, smooth=smooth)
        self.boundary_loss = BoundaryLoss(smooth=smooth)
        self.multiscale_loss = MultiScaleLoss(smooth=smooth)

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Standard Dice loss."""
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined thin vessel loss.

        Args:
            pred: Predicted segmentation (B, 1, D, H, W) or (B, D, H, W)
            target: Ground truth segmentation

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses
        """
        # Ensure proper dimensions
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # Ensure pred is in [0, 1] range (apply sigmoid if needed)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Compute individual losses
        loss_dice = self.dice_loss(pred, target)
        loss_cldice = self.cldice_loss(pred, target)
        loss_focal = self.focal_loss(pred, target)
        loss_boundary = self.boundary_loss(pred, target)
        loss_multiscale = self.multiscale_loss(pred, target)

        # Combine losses
        total_loss = (self.dice_weight * loss_dice +
                     self.cldice_weight * loss_cldice +
                     self.focal_weight * loss_focal +
                     self.boundary_weight * loss_boundary +
                     self.multiscale_weight * loss_multiscale)

        loss_dict = {
            'loss_dice': loss_dice,
            'loss_cldice': loss_cldice,
            'loss_focal': loss_focal,
            'loss_boundary': loss_boundary,
            'loss_multiscale': loss_multiscale,
        }

        return total_loss, loss_dict


class DistanceWeightedDiceLoss(nn.Module):
    """
    Distance-weighted Dice loss that emphasizes thin structures.

    Uses distance transform to weight voxels - thinner structures get higher weights.
    """

    def __init__(self, sigma: float = 5.0, smooth: float = 1e-5):
        """
        Args:
            sigma: Controls the spread of distance weighting
            smooth: Smoothing factor
        """
        super().__init__()
        self.sigma = sigma
        self.smooth = smooth

    def compute_distance_weights(self, target: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based weights (thin structures get higher weights).

        Uses erosion iterations to estimate distance from boundary.
        """
        weights = torch.ones_like(target)

        eroded = target.clone()
        for i in range(1, 10):
            # Approximate erosion with min pooling
            eroded = -F.max_pool3d(-eroded, kernel_size=3, stride=1, padding=1)
            # Thin structures erode faster, so they have smaller eroded values

        # Weight inversely proportional to how much survives erosion
        # Thin parts (small eroded values) get higher weights
        distance_estimate = target - eroded
        weights = 1.0 + distance_estimate * self.sigma

        return weights

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute distance-weighted Dice loss."""
        if pred.dim() == 4:
            pred = pred.unsqueeze(1)
        if target.dim() == 4:
            target = target.unsqueeze(1)

        # Compute weights (higher for thin structures)
        weights = self.compute_distance_weights(target)

        # Weighted Dice
        intersection = (weights * pred * target).sum()
        pred_sum = (weights * pred).sum()
        target_sum = (weights * target).sum()

        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)

        return 1.0 - dice
