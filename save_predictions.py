"""
Save prediction results as NIfTI files for visualization.

This script runs inference on validation/test sets and saves predictions
as .nii.gz files that can be visualized with ITK-SNAP, 3D Slicer, etc.

Usage:
    python save_predictions.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split val test \
        --output-dir ./predictions_latest \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

Output:
    predictions_latest/
    ├── val/
    │   ├── lung3d_00025_pred.nii.gz
    │   ├── lung3d_00025_prob.nii.gz  (probability map)
    │   └── ...
    └── test/
        ├── lung3d_01928_pred.nii.gz
        ├── lung3d_01928_prob.nii.gz
        └── ...
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import nibabel as nib
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import default_setup, launch
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

from vesselseg.config import add_seg3d_config
from train_utils import get_dataset_dicts
from train_vessel_seg import VesselTrainer, get_dataset_mapper


def save_nifti(data, output_path, affine=None):
    """
    Save numpy array as NIfTI file.

    Args:
        data: numpy array (3D volume)
        output_path: path to save .nii.gz file
        affine: affine transformation matrix (optional)
    """
    if affine is None:
        affine = np.eye(4)

    # Ensure data is in correct format
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    # Remove extra dimensions
    data = np.squeeze(data)

    # Create NIfTI image
    nii_img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(nii_img, output_path)


def build_inference_loader(cfg, subset):
    """
    Build data loader for inference.

    Args:
        cfg: config
        subset: 'train', 'val', or 'test'

    Returns:
        tuple: (DataLoader, dataset_dicts)
    """
    dataset_dicts = get_dataset_dicts(cfg.DATASETS.TEST, cfg, is_train=False, subset=subset)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {len(dataset_dicts)} samples for {subset} set")

    dataset = DatasetFromList(dataset_dicts)
    mapper = get_dataset_mapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=lambda batch: batch,
    )
    return data_loader, dataset_dicts


def run_inference_and_save(cfg, model, subset, output_dir, save_prob=True):
    """
    Run inference on a subset and save predictions as NIfTI files.

    Args:
        cfg: config
        model: trained model
        subset: 'train', 'val', or 'test'
        output_dir: base output directory
        save_prob: whether to save probability maps

    Returns:
        list: saved file paths
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running inference on {subset.upper()} set")
    logger.info(f"{'='*60}")

    data_loader, dataset_dicts = build_inference_loader(cfg, subset)

    if len(data_loader) == 0:
        logger.warning(f"No samples found for {subset} set!")
        return []

    # Create output directory for this subset
    subset_output_dir = os.path.join(output_dir, subset)
    os.makedirs(subset_output_dir, exist_ok=True)

    saved_files = []
    model.eval()

    with torch.no_grad():
        for idx, inputs in enumerate(tqdm(data_loader, desc=f"Saving {subset} predictions")):
            # Get case info
            if idx < len(dataset_dicts):
                sample_info = dataset_dicts[idx]
                file_id = sample_info.get('file_id', f'sample_{idx}')
                case_id = os.path.basename(file_id).replace('.npz', '')

                # Try to get original affine from NPZ if available
                npz_path = sample_info.get('file_name', '')
                affine = None
                original_shape = None
                if npz_path and os.path.exists(npz_path):
                    try:
                        npz_data = np.load(npz_path, allow_pickle=True)
                        if 'affine' in npz_data:
                            affine = npz_data['affine']
                        if 'image' in npz_data:
                            original_shape = npz_data['image'].shape
                    except Exception:
                        pass
            else:
                case_id = f'sample_{idx}'
                affine = None
                original_shape = None

            # Run inference
            outputs = model(inputs)

            # Process each output in batch (typically batch size = 1)
            for i, output in enumerate(outputs):
                # Get segmentation prediction
                if 'seg' in output:
                    pred_seg = output['seg']
                    if isinstance(pred_seg, torch.Tensor):
                        pred_seg = pred_seg.cpu().numpy()

                    pred_seg = np.squeeze(pred_seg)

                    # Save probability map
                    if save_prob:
                        prob_path = os.path.join(subset_output_dir, f'{case_id}_prob.nii.gz')
                        save_nifti(pred_seg, prob_path, affine)
                        saved_files.append(prob_path)

                    # Save binary prediction (threshold at 0.5)
                    pred_binary = (pred_seg > 0.5).astype(np.uint8)
                    pred_path = os.path.join(subset_output_dir, f'{case_id}_pred.nii.gz')
                    save_nifti(pred_binary, pred_path, affine)
                    saved_files.append(pred_path)

                    logger.debug(f"Saved: {case_id}_pred.nii.gz (shape: {pred_binary.shape})")

                # Also save ground truth if available in input
                if 'seg' in inputs[i]:
                    gt_seg = inputs[i]['seg']
                    if isinstance(gt_seg, torch.Tensor):
                        gt_seg = gt_seg.cpu().numpy()
                    gt_seg = np.squeeze(gt_seg)

                    gt_path = os.path.join(subset_output_dir, f'{case_id}_gt.nii.gz')
                    save_nifti(gt_seg.astype(np.uint8), gt_path, affine)
                    saved_files.append(gt_path)

    logger.info(f"Saved {len(saved_files)} files to {subset_output_dir}")
    return saved_files


def main(args):
    """Main function."""
    # Setup config
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="VesselSeg3D-Pred",
        abbrev_name="pred"
    )

    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = args.output_dir
    if comm.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Predictions will be saved to: {output_dir}")

    # Check if CSV split is specified
    split_csv = getattr(cfg.DATASETS, 'SPLIT_CSV', '')
    if not split_csv or not os.path.exists(split_csv):
        logger.warning("DATASETS.SPLIT_CSV not specified or file not found!")
        logger.warning("Will use hash-based splitting (TEST_FOLDS)")

    # Build model
    logger.info("Building model...")
    model = VesselTrainer.build_model(cfg)

    # Load weights
    logger.info(f"Loading weights from: {args.weights}")
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.weights, resume=False
    )
    model.eval()

    # Run inference and save predictions for each split
    all_saved_files = OrderedDict()
    for subset in args.split:
        saved_files = run_inference_and_save(
            cfg, model, subset, output_dir,
            save_prob=args.save_prob
        )
        all_saved_files[subset] = saved_files

    # Print summary
    if comm.is_main_process():
        logger.info("\n" + "="*60)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*60)

        total_files = 0
        for subset, files in all_saved_files.items():
            n_files = len(files)
            total_files += n_files
            n_cases = n_files // (3 if args.save_prob else 2)  # pred + prob + gt or pred + gt
            logger.info(f"{subset.upper()}: {n_cases} cases, {n_files} files")

        logger.info("-"*60)
        logger.info(f"Total: {total_files} files saved to {output_dir}")
        logger.info("="*60)

        logger.info("\nOutput structure:")
        logger.info(f"  {output_dir}/")
        for subset in args.split:
            logger.info(f"  ├── {subset}/")
            logger.info(f"  │   ├── <case_id>_pred.nii.gz  (binary prediction)")
            if args.save_prob:
                logger.info(f"  │   ├── <case_id>_prob.nii.gz  (probability map)")
            logger.info(f"  │   └── <case_id>_gt.nii.gz    (ground truth)")

    return all_saved_files


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Save Vessel Segmentation Predictions")
    parser.add_argument(
        "--config-file",
        default="configs/vessel_deformcl.yaml",
        metavar="FILE",
        help="Path to config file",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="FILE",
        help="Path to model weights (.pth file)",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        default=["val", "test"],
        choices=["train", "val", "test"],
        help="Which split(s) to run inference on (default: val test)",
    )
    parser.add_argument(
        "--output-dir",
        default="./predictions_latest",
        metavar="DIR",
        help="Directory to save predictions (default: ./predictions_latest)",
    )
    parser.add_argument(
        "--save-prob",
        action="store_true",
        default=True,
        help="Save probability maps in addition to binary predictions",
    )
    parser.add_argument(
        "--no-save-prob",
        action="store_false",
        dest="save_prob",
        help="Don't save probability maps",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=1,
        help="Total number of machines",
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="Rank of this machine",
    )
    parser.add_argument(
        "--dist-url",
        default="auto",
        help="URL for distributed training",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    print("=" * 60)
    print("Save Vessel Segmentation Predictions")
    print("=" * 60)
    print(f"Config: {args.config_file}")
    print(f"Weights: {args.weights}")
    print(f"Splits: {args.split}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Save Probability Maps: {args.save_prob}")
    print("=" * 60)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
