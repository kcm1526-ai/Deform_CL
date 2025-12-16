"""
Test script for Vessel Segmentation using trained DeformCL model.

This script:
1. Loads the trained model from checkpoint
2. Reads the CSV file to filter test samples
3. Runs inference on each test sample
4. Saves segmentation results in ORIGINAL coordinates (not cropped)

Usage:
    python test_vessel_seg.py \
        --config-file configs/vessel_mednext.yaml \
        --checkpoint ./outputs/vessel_mednext/model_final.pth \
        --csv-file impulse2_rl.csv \
        --output-dir inference_mednext \
        --subset test

    # For specific GPU:
    CUDA_VISIBLE_DEVICES=0 python test_vessel_seg.py ...
"""

import os
import argparse
import csv
import copy
import logging
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data.transforms import AugmentationList, AugInput

from vesselseg.config import add_seg3d_config
from vesselseg.data.datasets import load_npz_cta_dataset
from vesselseg.data.dataset_mapper import build_vessel_transform_gen, CenterCrop
from train_utils import load_split_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_cfg(config_file, checkpoint_path):
    """Setup config for inference."""
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.freeze()
    return cfg


def get_test_samples_from_csv(dataset_dicts, csv_path, subset='test'):
    """Filter dataset dicts to only include samples from the specified subset."""
    split_map = load_split_csv(csv_path)

    filtered_dicts = []
    for d in dataset_dicts:
        file_id = d.get("file_id", "")
        case_id = os.path.basename(file_id).replace('.npz', '')

        if case_id in split_map:
            if split_map[case_id] == subset:
                filtered_dicts.append(d)
        else:
            for pid, s in split_map.items():
                if pid in case_id or case_id in pid:
                    if s == subset:
                        filtered_dicts.append(d)
                    break

    logger.info(f"Found {len(filtered_dicts)} samples for '{subset}' subset")
    return filtered_dicts


def save_nifti(array, output_path, spacing=None, origin=None, direction=None):
    """
    Save array as NIfTI file using SimpleITK for proper spatial info.
    SimpleITK arrays are in (Z, Y, X) order.
    """
    import SimpleITK as sitk

    # Create SimpleITK image
    sitk_img = sitk.GetImageFromArray(array.astype(np.float32))

    # Set spatial information if available
    if spacing is not None:
        sitk_img.SetSpacing(tuple(float(s) for s in spacing))
    if origin is not None:
        sitk_img.SetOrigin(tuple(float(o) for o in origin))
    if direction is not None:
        sitk_img.SetDirection(tuple(float(d) for d in direction))

    sitk.WriteImage(sitk_img, output_path)


def save_nifti_nibabel(array, output_path, affine=None):
    """Save array as NIfTI file using nibabel (fallback)."""
    if affine is None:
        affine = np.eye(4)
    nii_img = nib.Nifti1Image(array.astype(np.float32), affine)
    nib.save(nii_img, output_path)


def compute_dice(pred, gt, smooth=1e-5):
    """Compute Dice score."""
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0).astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class InferenceDataMapper:
    """
    Custom data mapper for inference that tracks crop coordinates.
    This allows us to place predictions back into original volume coordinates.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.class_id = cfg.MODEL.PRED_CLASS
        self.pad = np.array([20, 20, 20])

        # Build augmentations (CenterCrop for inference)
        augmentations = build_vessel_transform_gen(cfg, is_train=False)
        self.augmentations = AugmentationList(augmentations)

    def __call__(self, dataset_dict):
        """
        Process a single sample and return both the processed data
        and the crop coordinates for reconstruction.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        npz_file = np.load(dataset_dict["file_name"], allow_pickle=True)

        # Load ORIGINAL data
        image_original = npz_file["img"].astype(np.float32)
        seg_original = npz_file["seg"].copy()
        cline_original = npz_file["cline"].copy()

        original_shape = image_original.shape

        # Normalize image
        image = image_original / 1024.0
        seg = seg_original.copy()
        cline = cline_original.copy()

        src_shape = np.array(image.shape)

        # Find ROI based on segmentation mask (same as VesselClineDeformDatasetMapper)
        if self.class_id > 0:
            seg_mask = (seg == self.class_id)
        else:
            seg_mask = (seg > 0)

        if seg_mask.any():
            indices = np.array(np.where(seg_mask))
            roi_start = np.maximum(indices.min(1) - self.pad, 0)
            roi_end = np.minimum(indices.max(1) + 1 + self.pad, src_shape)
        else:
            roi_start = np.array([0, 0, 0])
            roi_end = src_shape

        # Store ROI crop coordinates
        dataset_dict["roi_start"] = roi_start.copy()
        dataset_dict["roi_end"] = roi_end.copy()
        dataset_dict["original_shape"] = original_shape

        # Crop to ROI
        image_roi = image[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
        seg_roi = seg[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]
        cline_roi = cline[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]]

        # Filter to single class
        if self.class_id > 0:
            seg_roi[seg_roi != self.class_id] = 0
            cline_roi[cline_roi != self.class_id] = 0

        # Store ROI shape before augmentation
        roi_shape = image_roi.shape
        dataset_dict["roi_shape"] = roi_shape

        # Apply augmentations (CenterCrop for inference)
        aug_input = AugInput(image=image_roi, sem_seg=seg_roi)
        transforms = self.augmentations(aug_input)
        image_aug = aug_input.image

        # Get crop transform info if CenterCrop was applied
        crop_start_in_roi = np.array([0, 0, 0])
        for tfm in transforms.transforms:
            if hasattr(tfm, 'x0'):  # CropTransform has x0, y0, etc.
                # Note: CropTransform uses (h0, w0, d0) = (x0, y0, z0)
                crop_start_in_roi = np.array([tfm.x0, tfm.y0, getattr(tfm, 'z0', 0)])
                break

        dataset_dict["crop_start_in_roi"] = crop_start_in_roi

        # Apply same transforms to seg and cline
        cline_aug = transforms.apply_image(cline_roi)
        seg_aug = transforms.apply_image(seg_roi)

        # Store final shape after augmentation
        dataset_dict["aug_shape"] = image_aug.shape

        # Create tensors for model input
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_aug[None, ...]))
        dataset_dict["cline"] = torch.as_tensor(np.ascontiguousarray(cline_aug[None, ...]))
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg_aug[None, ...]))

        # Store UNMODIFIED original ground truth for comparison
        # Don't filter by class_id here - save raw data from NPZ
        dataset_dict["seg_original_raw"] = npz_file["seg"].copy()  # Completely raw
        dataset_dict["seg_original"] = seg_original  # After class filtering
        dataset_dict["seg_roi"] = seg_roi  # Cropped but before CenterCrop
        dataset_dict["npz_path"] = dataset_dict["file_name"]  # Store path for debug

        # Store spatial information from NPZ for proper NIfTI saving
        if "spacing" in npz_file:
            dataset_dict["spacing"] = npz_file["spacing"]
        if "origin" in npz_file:
            dataset_dict["origin"] = npz_file["origin"]
        if "direction" in npz_file:
            dataset_dict["direction"] = npz_file["direction"]

        return dataset_dict


def reconstruct_full_volume(pred_seg, dataset_dict):
    """
    Reconstruct prediction in original volume coordinates.

    The prediction is in augmented (CenterCropped) coordinates.
    We need to:
    1. Place it back into ROI coordinates
    2. Place ROI back into original coordinates
    """
    original_shape = dataset_dict["original_shape"]
    roi_start = dataset_dict["roi_start"]
    roi_end = dataset_dict["roi_end"]
    roi_shape = dataset_dict["roi_shape"]
    crop_start_in_roi = dataset_dict["crop_start_in_roi"]
    aug_shape = dataset_dict["aug_shape"]

    # Create full volume (all zeros)
    full_pred = np.zeros(original_shape, dtype=np.float32)

    # First, place prediction into ROI coordinates
    roi_pred = np.zeros(roi_shape, dtype=np.float32)

    # The prediction shape should match aug_shape (or be slightly different due to model padding)
    pred_shape = pred_seg.shape

    # Determine where in ROI the prediction goes
    # crop_start_in_roi tells us where CenterCrop started
    h0, w0, d0 = crop_start_in_roi
    h1 = min(h0 + pred_shape[0], roi_shape[0])
    w1 = min(w0 + pred_shape[1], roi_shape[1])
    d1 = min(d0 + pred_shape[2], roi_shape[2])

    # Clip prediction to fit
    pred_h = min(pred_shape[0], h1 - h0)
    pred_w = min(pred_shape[1], w1 - w0)
    pred_d = min(pred_shape[2], d1 - d0)

    roi_pred[h0:h0+pred_h, w0:w0+pred_w, d0:d0+pred_d] = pred_seg[:pred_h, :pred_w, :pred_d]

    # Now place ROI into full volume
    rh = roi_end[0] - roi_start[0]
    rw = roi_end[1] - roi_start[1]
    rd = roi_end[2] - roi_start[2]

    full_pred[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1], roi_start[2]:roi_end[2]] = \
        roi_pred[:rh, :rw, :rd]

    return full_pred


def run_inference(cfg, model, dataset_dicts, output_dir, device):
    """Run inference on all samples and save results in original coordinates."""
    model.eval()
    mapper = InferenceDataMapper(cfg)

    # Create output directories
    seg_dir = os.path.join(output_dir, "segmentations")
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    results = []

    logger.info(f"Running inference on {len(dataset_dicts)} samples...")

    with torch.no_grad():
        for idx, dataset_dict in enumerate(tqdm(dataset_dicts, desc="Inference")):
            file_id = dataset_dict.get("file_id", f"sample_{idx}")
            case_id = os.path.basename(file_id).replace('.npz', '')

            logger.info(f"Processing: {case_id}")

            try:
                # Process data through our custom mapper
                data = mapper(dataset_dict)

                # Log coordinate info
                logger.info(f"  Original shape: {data['original_shape']}")
                logger.info(f"  ROI: {data['roi_start']} to {data['roi_end']}")
                logger.info(f"  ROI shape: {data['roi_shape']}")
                logger.info(f"  Crop start in ROI: {data['crop_start_in_roi']}")
                logger.info(f"  Aug shape: {data['aug_shape']}")
                logger.info(f"  Input tensor shape: {data['image'].shape}")

                # Prepare input for model
                inputs = [{
                    "image": data["image"].to(device),
                    "seg": data["seg"].to(device),
                    "cline": data["cline"].to(device),
                    "file_id": file_id,
                }]

                # Run inference
                outputs = model(inputs)
                output = outputs[0]

                pred_seg = output.get("seg", None)

                if pred_seg is not None:
                    pred_seg_np = pred_seg.cpu().numpy()
                    logger.info(f"  Model output shape: {pred_seg_np.shape}")

                    # Binarize prediction
                    pred_binary = (pred_seg_np > 0.5).astype(np.float32)

                    # Reconstruct in full volume coordinates
                    full_pred = reconstruct_full_volume(pred_binary, data)
                    logger.info(f"  Full pred shape: {full_pred.shape}")
                    logger.info(f"  Full pred foreground voxels: {full_pred.sum()}")

                    # Get original ground truth - use RAW (unfiltered) data
                    gt_raw = data["seg_original_raw"]
                    gt_filtered = data["seg_original"]

                    # Debug: show what's in the data
                    logger.info(f"  NPZ path: {data['npz_path']}")
                    logger.info(f"  GT raw unique values: {np.unique(gt_raw)}")
                    logger.info(f"  GT raw shape: {gt_raw.shape}")
                    logger.info(f"  GT raw foreground (>0): {(gt_raw > 0).sum()}")
                    logger.info(f"  GT filtered (class_id={cfg.MODEL.PRED_CLASS}) foreground: {(gt_filtered > 0).sum()}")

                    # Use RAW ground truth for saving (to match /raid)
                    gt_binary = (gt_raw > 0).astype(np.float32)
                    logger.info(f"  Original GT foreground voxels: {gt_binary.sum()}")

                    # Compute Dice on FULL volume
                    dice_full = compute_dice(full_pred, gt_binary)
                    logger.info(f"  Dice (full volume): {dice_full:.4f}")

                    # Also compute Dice on cropped region for comparison
                    gt_cropped = data["seg"].cpu().numpy()
                    if gt_cropped.ndim == 4:
                        gt_cropped = gt_cropped[0]
                    gt_cropped_binary = (gt_cropped > 0).astype(np.float32)

                    # Match shapes for cropped comparison
                    min_shape = tuple(min(p, g) for p, g in zip(pred_binary.shape, gt_cropped_binary.shape))
                    pred_crop = pred_binary[:min_shape[0], :min_shape[1], :min_shape[2]]
                    gt_crop = gt_cropped_binary[:min_shape[0], :min_shape[1], :min_shape[2]]
                    dice_cropped = compute_dice(pred_crop, gt_crop)
                    logger.info(f"  Dice (cropped): {dice_cropped:.4f}")

                    # Get spatial info from NPZ
                    spacing = data.get("spacing", None)
                    origin = data.get("origin", None)
                    direction = data.get("direction", None)

                    logger.info(f"  Spatial info - spacing: {spacing}, origin: {origin}")

                    # Save prediction in ORIGINAL coordinates with proper spatial info
                    seg_path = os.path.join(seg_dir, f"{case_id}_pred.nii.gz")
                    save_nifti(full_pred, seg_path, spacing, origin, direction)

                    # Save original ground truth for comparison (should match /raid exactly)
                    gt_path = os.path.join(gt_dir, f"{case_id}_gt.nii.gz")
                    save_nifti(gt_binary, gt_path, spacing, origin, direction)

                    results.append({
                        "case_id": case_id,
                        "file_id": file_id,
                        "dice_full": dice_full,
                        "dice_cropped": dice_cropped,
                        "seg_path": seg_path,
                        "gt_path": gt_path,
                    })
                else:
                    logger.warning(f"  No segmentation output!")
                    results.append({
                        "case_id": case_id,
                        "file_id": file_id,
                        "dice_full": 0.0,
                        "dice_cropped": 0.0,
                        "error": "No segmentation output",
                    })

            except Exception as e:
                logger.error(f"Error processing {case_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "case_id": case_id,
                    "file_id": file_id,
                    "dice_full": 0.0,
                    "dice_cropped": 0.0,
                    "error": str(e),
                })

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def save_results(results, output_dir):
    """Save results to CSV file and print summary."""
    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, 'w', newline='') as f:
        fieldnames = ["case_id", "file_id", "dice_full", "dice_cropped", "seg_path", "gt_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    # Compute summary statistics
    dice_full_scores = [r["dice_full"] for r in results if "error" not in r]
    dice_cropped_scores = [r["dice_cropped"] for r in results if "error" not in r]

    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Successful: {len(dice_full_scores)}")
    logger.info(f"Failed: {len(results) - len(dice_full_scores)}")
    logger.info("-"*60)

    if dice_full_scores:
        logger.info(f"Dice Score (FULL VOLUME - original coordinates):")
        logger.info(f"  Mean: {np.mean(dice_full_scores):.4f}")
        logger.info(f"  Std:  {np.std(dice_full_scores):.4f}")
        logger.info(f"  Min:  {np.min(dice_full_scores):.4f}")
        logger.info(f"  Max:  {np.max(dice_full_scores):.4f}")
        logger.info("-"*60)
        logger.info(f"Dice Score (cropped region):")
        logger.info(f"  Mean: {np.mean(dice_cropped_scores):.4f}")
        logger.info(f"  Std:  {np.std(dice_cropped_scores):.4f}")

    logger.info("-"*60)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Predictions saved to: {os.path.join(output_dir, 'segmentations')}")
    logger.info(f"Ground truth saved to: {os.path.join(output_dir, 'ground_truth')}")
    logger.info("="*60)

    return {
        "mean_dice_full": np.mean(dice_full_scores) if dice_full_scores else 0,
        "mean_dice_cropped": np.mean(dice_cropped_scores) if dice_cropped_scores else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Vessel Segmentation Model")
    parser.add_argument("--config-file", default="configs/vessel_mednext.yaml")
    parser.add_argument("--checkpoint", default="./outputs/vessel_mednext/model_final.pth")
    parser.add_argument("--csv-file", default="impulse2_rl.csv")
    parser.add_argument("--output-dir", default="inference_mednext")
    parser.add_argument("--subset", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-dir", default="./VesselSeg_Data")
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Vessel Segmentation Testing (Original Coordinates)")
    logger.info("="*60)
    logger.info(f"Config: {args.config_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*60)

    # Validate files
    for path, name in [(args.config_file, "Config"), (args.checkpoint, "Checkpoint"),
                       (args.csv_file, "CSV"), (args.data_dir, "Data dir")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")

    # Setup
    cfg = setup_cfg(args.config_file, args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build and load model
    logger.info("Building model...")
    model = build_model(cfg)
    model.to(device)

    logger.info("Loading checkpoint...")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset_dicts = load_npz_cta_dataset(args.data_dir)
    logger.info(f"Found {len(dataset_dicts)} total samples")

    # Filter to subset
    test_dicts = get_test_samples_from_csv(dataset_dicts, args.csv_file, args.subset)

    if len(test_dicts) == 0:
        logger.error(f"No samples found for subset '{args.subset}'")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    results = run_inference(cfg, model, test_dicts, args.output_dir, device)

    # Save results
    save_results(results, args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
