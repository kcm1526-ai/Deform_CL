"""
Test script for Vessel Segmentation using trained DeformCL model.

This script:
1. Loads the trained model from checkpoint
2. Reads the CSV file to filter test samples
3. Runs inference on each test sample
4. Saves segmentation results and computes metrics

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
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from vesselseg.config import add_seg3d_config
from vesselseg.data.datasets import load_npz_cta_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_split_csv(csv_path):
    """
    Load train/val/test split from CSV file.
    CSV format: pid,subset
    where subset is one of: train, val, test

    Returns: dict mapping pid -> subset
    """
    split_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['pid'].strip()
            subset = row['subset'].strip().lower()
            split_map[pid] = subset
    return split_map


def get_test_samples(dataset_dicts, csv_path, subset='test'):
    """
    Filter dataset dicts to only include samples from the specified subset.

    Args:
        dataset_dicts: list of dataset dicts from DatasetCatalog
        csv_path: path to CSV file with pid,subset columns
        subset: 'train', 'val', or 'test'

    Returns:
        filtered list of dataset dicts
    """
    split_map = load_split_csv(csv_path)

    filtered_dicts = []
    for d in dataset_dicts:
        file_id = d.get("file_id", "")
        case_id = os.path.basename(file_id).replace('.npz', '')

        if case_id in split_map:
            if split_map[case_id] == subset:
                filtered_dicts.append(d)
        else:
            # Try partial matching
            for pid, s in split_map.items():
                if pid in case_id or case_id in pid:
                    if s == subset:
                        filtered_dicts.append(d)
                    break

    logger.info(f"Found {len(filtered_dicts)} samples for '{subset}' subset")
    return filtered_dicts


def setup_cfg(config_file, checkpoint_path):
    """
    Setup config for inference.
    """
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoint_path
    cfg.freeze()
    return cfg


def save_segmentation(seg_array, output_path, affine=None):
    """
    Save segmentation result as NIfTI file.

    Args:
        seg_array: numpy array of segmentation
        output_path: path to save the NIfTI file
        affine: affine matrix for NIfTI (identity if None)
    """
    # Ensure binary mask
    seg_array = (seg_array > 0.5).astype(np.uint8)

    # Create NIfTI image
    if affine is None:
        affine = np.eye(4)

    nii_img = nib.Nifti1Image(seg_array, affine)
    nib.save(nii_img, output_path)


def save_centerline(verts, output_path):
    """
    Save centerline vertices as numpy file.

    Args:
        verts: numpy array of centerline vertices (N, 3)
        output_path: path to save the numpy file
    """
    np.save(output_path, verts)


def compute_dice(pred, gt, smooth=1e-5):
    """Compute Dice score."""
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class TestDataMapper:
    """
    Dataset mapper for testing that prepares data for model inference.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.class_id = cfg.MODEL.PRED_CLASS
        self.pad = np.array([20, 20, 20])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        npz_file = np.load(dataset_dict["file_name"], allow_pickle=True)

        # Load image and normalize
        image = npz_file["img"].astype(np.float32)
        original_shape = image.shape
        image = image / 1024.0

        # Load segmentation (ground truth for comparison)
        seg = npz_file["seg"].copy()

        # Load centerline
        cline = npz_file["cline"].copy()

        src_shape = np.array(image.shape)

        # Find ROI based on segmentation mask
        if self.class_id > 0:
            seg_mask = (seg == self.class_id)
        else:
            seg_mask = (seg > 0)

        if seg_mask.any():
            indices = np.array(np.where(seg_mask))
            start = np.maximum(indices.min(1) - self.pad, 0)
            end = np.minimum(indices.max(1) + 1 + self.pad, src_shape)
        else:
            start = np.array([0, 0, 0])
            end = src_shape

        # Store crop info for later reconstruction
        dataset_dict["crop_start"] = start.tolist()
        dataset_dict["crop_end"] = end.tolist()
        dataset_dict["original_shape"] = original_shape

        # Crop to ROI
        image_cropped = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        seg_cropped = seg[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        cline_cropped = cline[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

        # Filter to single class
        if self.class_id > 0:
            seg_cropped[seg_cropped != self.class_id] = 0
            cline_cropped[cline_cropped != self.class_id] = 0

        # Store ground truth for evaluation (full and cropped)
        dataset_dict["gt_seg_full"] = seg.copy()
        dataset_dict["gt_seg"] = seg_cropped.copy()
        dataset_dict["gt_cline"] = cline_cropped.copy()

        # Store in dataset dict for model input
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image_cropped[None, ...]))
        dataset_dict["cline"] = torch.as_tensor(np.ascontiguousarray(cline_cropped[None, ...]))
        dataset_dict["seg"] = torch.as_tensor(np.ascontiguousarray(seg_cropped[None, ...]))

        return dataset_dict


def run_inference(cfg, model, dataset_dicts, output_dir, device):
    """
    Run inference on all samples and save results.

    Args:
        cfg: config
        model: trained model
        dataset_dicts: list of dataset dicts to process
        output_dir: directory to save results
        device: torch device

    Returns:
        list of results dictionaries
    """
    model.eval()
    mapper = TestDataMapper(cfg)

    # Create output directories
    seg_dir = os.path.join(output_dir, "segmentations")
    cline_dir = os.path.join(output_dir, "centerlines")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(cline_dir, exist_ok=True)

    # Metrics storage
    results = []

    logger.info(f"Running inference on {len(dataset_dicts)} samples...")

    with torch.no_grad():
        for dataset_dict in tqdm(dataset_dicts, desc="Inference"):
            file_id = dataset_dict["file_id"]
            logger.info(f"Processing: {file_id}")

            # Prepare input
            data = mapper(dataset_dict)

            # Move to device
            inputs = [{
                "image": data["image"].to(device),
                "seg": data["seg"].to(device),
                "cline": data["cline"].to(device),
            }]

            try:
                # Run inference
                outputs = model(inputs)
                output = outputs[0]

                # Get predictions
                pred_seg = output["seg"]  # Predicted segmentation (cropped region)
                pred_cline = output.get("pred_cline", {})
                pred_verts = pred_cline.get("verts", None)

                # Get crop info
                original_shape = data["original_shape"]
                crop_start = data["crop_start"]
                crop_end = data["crop_end"]

                # Convert prediction to numpy
                pred_seg_np = pred_seg.cpu().numpy().astype(np.uint8)

                # Create full-size output
                full_seg = np.zeros(original_shape, dtype=np.uint8)
                h, w, d = pred_seg_np.shape
                full_seg[
                    crop_start[0]:crop_start[0]+h,
                    crop_start[1]:crop_start[1]+w,
                    crop_start[2]:crop_start[2]+d
                ] = pred_seg_np

                # Save full segmentation
                seg_path = os.path.join(seg_dir, f"{file_id}_seg.nii.gz")
                save_segmentation(full_seg, seg_path)

                # Save centerline if available
                cline_path = None
                if pred_verts is not None:
                    # Adjust vertices to original coordinates
                    pred_verts_np = pred_verts.cpu().numpy()
                    pred_verts_np = pred_verts_np + np.array(crop_start)

                    cline_path = os.path.join(cline_dir, f"{file_id}_cline.npy")
                    save_centerline(pred_verts_np, cline_path)

                # Compute Dice score on cropped region
                gt_seg = data["gt_seg"]
                dice = compute_dice(pred_seg_np, gt_seg)

                # Compute Dice score on full volume
                gt_seg_full = data["gt_seg_full"]
                if cfg.MODEL.PRED_CLASS > 0:
                    gt_seg_full = (gt_seg_full == cfg.MODEL.PRED_CLASS).astype(np.uint8)
                else:
                    gt_seg_full = (gt_seg_full > 0).astype(np.uint8)
                dice_full = compute_dice(full_seg, gt_seg_full)

                results.append({
                    "file_id": file_id,
                    "dice_cropped": dice,
                    "dice_full": dice_full,
                    "seg_path": seg_path,
                    "cline_path": cline_path if cline_path else "",
                })

                logger.info(f"  Dice (cropped): {dice:.4f}, Dice (full): {dice_full:.4f}")

            except Exception as e:
                logger.error(f"Error processing {file_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "file_id": file_id,
                    "dice_cropped": 0.0,
                    "dice_full": 0.0,
                    "error": str(e),
                })

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def save_results(results, output_dir):
    """Save results to CSV file and print summary."""
    # Save results to CSV
    results_path = os.path.join(output_dir, "results.csv")
    with open(results_path, 'w', newline='') as f:
        fieldnames = ["file_id", "dice_cropped", "dice_full", "seg_path", "cline_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    # Compute and print summary statistics
    dice_cropped = [r["dice_cropped"] for r in results if "error" not in r]
    dice_full = [r["dice_full"] for r in results if "error" not in r]

    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Successful: {len(dice_cropped)}")
    logger.info(f"Failed: {len(results) - len(dice_cropped)}")
    logger.info("-"*60)

    if dice_cropped:
        logger.info(f"Dice Score (cropped ROI):")
        logger.info(f"  Mean: {np.mean(dice_cropped):.4f}")
        logger.info(f"  Std:  {np.std(dice_cropped):.4f}")
        logger.info(f"  Min:  {np.min(dice_cropped):.4f}")
        logger.info(f"  Max:  {np.max(dice_cropped):.4f}")
        logger.info("-"*60)
        logger.info(f"Dice Score (full volume):")
        logger.info(f"  Mean: {np.mean(dice_full):.4f}")
        logger.info(f"  Std:  {np.std(dice_full):.4f}")
        logger.info(f"  Min:  {np.min(dice_full):.4f}")
        logger.info(f"  Max:  {np.max(dice_full):.4f}")

    logger.info("-"*60)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Segmentations saved to: {os.path.join(output_dir, 'segmentations')}")
    logger.info(f"Centerlines saved to: {os.path.join(output_dir, 'centerlines')}")
    logger.info("="*60)

    return {
        "mean_dice_cropped": np.mean(dice_cropped) if dice_cropped else 0,
        "mean_dice_full": np.mean(dice_full) if dice_full else 0,
        "std_dice_cropped": np.std(dice_cropped) if dice_cropped else 0,
        "std_dice_full": np.std(dice_full) if dice_full else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Vessel Segmentation Model")
    parser.add_argument(
        "--config-file",
        default="configs/vessel_mednext.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        default="./outputs/vessel_mednext/model_final.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--csv-file",
        default="impulse2_rl.csv",
        help="Path to CSV file with train/val/test split"
    )
    parser.add_argument(
        "--output-dir",
        default="inference_mednext",
        help="Directory to save inference results"
    )
    parser.add_argument(
        "--subset",
        default="test",
        choices=["train", "val", "test"],
        help="Which subset to run inference on"
    )
    parser.add_argument(
        "--data-dir",
        default="./VesselSeg_Data",
        help="Directory containing the dataset"
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to run inference on (e.g., cuda:0, cuda:1, cpu)"
    )

    args = parser.parse_args()

    # Print setup info
    logger.info("="*60)
    logger.info("Vessel Segmentation Testing")
    logger.info("="*60)
    logger.info(f"Config file: {args.config_file}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"CSV file: {args.csv_file}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Subset: {args.subset}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60)

    # Validate files exist
    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV file not found: {args.csv_file}")
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    # Setup config
    cfg = setup_cfg(args.config_file, args.checkpoint)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model.to(device)

    # Load checkpoint
    logger.info(f"Loading checkpoint...")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(args.checkpoint)
    model.eval()

    # Get dataset
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset_dicts = load_npz_cta_dataset(args.data_dir)
    logger.info(f"Found {len(dataset_dicts)} total samples")

    # Filter to specified subset
    test_dicts = get_test_samples(dataset_dicts, args.csv_file, args.subset)

    if len(test_dicts) == 0:
        logger.error(f"No samples found for subset '{args.subset}'")
        logger.info("Available PIDs in CSV:")
        split_map = load_split_csv(args.csv_file)
        for pid, subset in list(split_map.items())[:10]:
            logger.info(f"  {pid}: {subset}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference
    results = run_inference(cfg, model, test_dicts, args.output_dir, device)

    # Save and print results
    summary = save_results(results, args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
