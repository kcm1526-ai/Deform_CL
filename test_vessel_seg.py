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
import logging
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler

from vesselseg.config import add_seg3d_config
from vesselseg.data.datasets import load_npz_cta_dataset

# Import the correct dataset mapper from train_vessel_seg
from train_vessel_seg import get_dataset_mapper
from train_utils import get_dataset_dicts, load_split_csv, split_dataset_by_csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def build_test_dataloader(cfg, dataset_dicts):
    """
    Build test data loader using the correct dataset mapper.
    This uses the same mapper as evaluation to ensure correct data preparation.
    """
    dataset = DatasetFromList(dataset_dicts)

    # Use the same mapper that training/evaluation uses
    mapper = get_dataset_mapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # Use 0 for easier debugging
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: batch,
    )
    return data_loader


def get_test_samples_from_csv(dataset_dicts, csv_path, subset='test'):
    """
    Filter dataset dicts to only include samples from the specified subset.
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


def save_segmentation(seg_array, output_path, affine=None):
    """
    Save segmentation result as NIfTI file.
    """
    seg_array = (seg_array > 0.5).astype(np.uint8)

    if affine is None:
        affine = np.eye(4)

    nii_img = nib.Nifti1Image(seg_array, affine)
    nib.save(nii_img, output_path)


def save_centerline(verts, output_path):
    """
    Save centerline vertices as numpy file.
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


def run_inference(cfg, model, data_loader, dataset_dicts, output_dir, device):
    """
    Run inference on all samples and save results.
    """
    model.eval()

    # Create output directories
    seg_dir = os.path.join(output_dir, "segmentations")
    gt_dir = os.path.join(output_dir, "ground_truth")
    cline_dir = os.path.join(output_dir, "centerlines")
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(cline_dir, exist_ok=True)

    results = []

    logger.info(f"Running inference on {len(data_loader)} samples...")

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc="Inference")):
            # Get file_id from the actual input (more reliable than index-based lookup)
            file_id = inputs[0].get("file_id", None)
            if file_id is None:
                # Fallback to dataset_dicts if file_id not in input
                if idx < len(dataset_dicts):
                    file_id = dataset_dicts[idx].get("file_id", f"sample_{idx}")
                else:
                    file_id = f"sample_{idx}"

            case_id = os.path.basename(file_id).replace('.npz', '')
            logger.info(f"Processing: {case_id} (file_id: {file_id})")

            # Debug: print input shapes
            logger.info(f"  Input image shape: {inputs[0]['image'].shape}")
            logger.info(f"  Input seg shape: {inputs[0]['seg'].shape}")

            try:
                # Run inference - the model expects inputs as a list of dicts
                outputs = model(inputs)
                output = outputs[0]

                # Get predictions
                pred_seg = output.get("seg", None)
                pred_cline = output.get("pred_cline", {})
                pred_verts = pred_cline.get("verts", None)

                # Get ground truth from input
                gt_seg = inputs[0].get("seg", None)

                # Debug: print output shapes
                if pred_seg is not None:
                    logger.info(f"  Output seg shape: {pred_seg.shape}")
                if gt_seg is not None:
                    logger.info(f"  GT seg shape: {gt_seg.shape}")

                # Compute dice score
                dice_score = 0.0
                if pred_seg is not None and gt_seg is not None:
                    pred_seg_np = pred_seg.cpu().numpy()
                    gt_seg_np = gt_seg.cpu().numpy()

                    # Debug: print numpy shapes before processing
                    logger.info(f"  pred_seg_np shape before: {pred_seg_np.shape}")
                    logger.info(f"  gt_seg_np shape before: {gt_seg_np.shape}")

                    # Handle shape differences
                    if pred_seg_np.ndim == 4:
                        pred_seg_np = pred_seg_np[0]  # Remove batch/channel dim
                    if gt_seg_np.ndim == 4:
                        gt_seg_np = gt_seg_np[0]

                    # Debug: print shapes after processing
                    logger.info(f"  pred_seg_np shape after: {pred_seg_np.shape}")
                    logger.info(f"  gt_seg_np shape after: {gt_seg_np.shape}")

                    # Check if shapes match
                    if pred_seg_np.shape != gt_seg_np.shape:
                        logger.warning(f"  SHAPE MISMATCH! pred: {pred_seg_np.shape} vs gt: {gt_seg_np.shape}")
                        # Try to match shapes by taking minimum dimensions
                        min_shape = tuple(min(p, g) for p, g in zip(pred_seg_np.shape, gt_seg_np.shape))
                        pred_seg_np = pred_seg_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                        gt_seg_np = gt_seg_np[:min_shape[0], :min_shape[1], :min_shape[2]]
                        logger.info(f"  Adjusted to common shape: {min_shape}")

                    # Binarize
                    pred_binary = (pred_seg_np > 0.5).astype(np.float32)
                    gt_binary = (gt_seg_np > 0).astype(np.float32)

                    # Debug: print sum of foreground voxels
                    logger.info(f"  Pred foreground voxels: {pred_binary.sum()}")
                    logger.info(f"  GT foreground voxels: {gt_binary.sum()}")

                    dice_score = compute_dice(pred_binary, gt_binary)

                    # Save prediction
                    seg_path = os.path.join(seg_dir, f"{case_id}_pred.nii.gz")
                    save_segmentation(pred_seg_np, seg_path)

                    # Also save ground truth for comparison
                    gt_path = os.path.join(gt_dir, f"{case_id}_gt.nii.gz")
                    save_segmentation(gt_binary, gt_path)
                else:
                    seg_path = ""
                    logger.warning(f"No segmentation output for {case_id}")

                # Save centerline if available
                cline_path = ""
                if pred_verts is not None:
                    pred_verts_np = pred_verts.cpu().numpy()
                    cline_path = os.path.join(cline_dir, f"{case_id}_cline.npy")
                    save_centerline(pred_verts_np, cline_path)

                # Define gt_path at this scope level
                gt_path_result = gt_path if pred_seg is not None and gt_seg is not None else ""

                results.append({
                    "case_id": case_id,
                    "file_id": file_id,
                    "dice": dice_score,
                    "seg_path": seg_path,
                    "gt_path": gt_path_result,
                    "cline_path": cline_path,
                })

                logger.info(f"  Dice: {dice_score:.4f}")

            except Exception as e:
                logger.error(f"Error processing {case_id}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "case_id": case_id,
                    "file_id": file_id,
                    "dice": 0.0,
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
        fieldnames = ["case_id", "file_id", "dice", "seg_path", "gt_path", "cline_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    # Compute summary statistics
    dice_scores = [r["dice"] for r in results if "error" not in r]

    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Successful: {len(dice_scores)}")
    logger.info(f"Failed: {len(results) - len(dice_scores)}")
    logger.info("-"*60)

    if dice_scores:
        logger.info(f"Dice Score:")
        logger.info(f"  Mean: {np.mean(dice_scores):.4f}")
        logger.info(f"  Std:  {np.std(dice_scores):.4f}")
        logger.info(f"  Min:  {np.min(dice_scores):.4f}")
        logger.info(f"  Max:  {np.max(dice_scores):.4f}")

    logger.info("-"*60)
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Predictions saved to: {os.path.join(output_dir, 'segmentations')}")
    logger.info(f"Ground truth saved to: {os.path.join(output_dir, 'ground_truth')}")
    logger.info(f"Centerlines saved to: {os.path.join(output_dir, 'centerlines')}")
    logger.info("="*60)

    return {
        "mean_dice": np.mean(dice_scores) if dice_scores else 0,
        "std_dice": np.std(dice_scores) if dice_scores else 0,
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
    test_dicts = get_test_samples_from_csv(dataset_dicts, args.csv_file, args.subset)

    if len(test_dicts) == 0:
        logger.error(f"No samples found for subset '{args.subset}'")
        logger.info("Available PIDs in CSV:")
        split_map = load_split_csv(args.csv_file)
        for pid, subset in list(split_map.items())[:10]:
            logger.info(f"  {pid}: {subset}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build data loader with correct mapper
    logger.info("Building data loader with correct dataset mapper...")
    data_loader = build_test_dataloader(cfg, test_dicts)

    # Run inference
    results = run_inference(cfg, model, data_loader, test_dicts, args.output_dir, device)

    # Save and print results
    summary = save_results(results, args.output_dir)

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
