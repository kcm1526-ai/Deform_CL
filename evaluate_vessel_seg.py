"""
Evaluation script for Vessel Segmentation.

This script evaluates trained models on validation or test sets and reports
Dice scores and other metrics.

Usage:
    # Evaluate on validation set
    python evaluate_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split val \
        --output-dir ./eval_results \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

    # Evaluate on test set
    python evaluate_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split test \
        --output-dir ./eval_results \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

    # Evaluate on both val and test
    python evaluate_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split val test \
        --output-dir ./eval_results \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

Results are saved to:
    - {output_dir}/eval_results.json  (all metrics in JSON format)
    - {output_dir}/eval_results.csv   (summary table in CSV format)
    - {output_dir}/per_sample_{split}.csv (per-sample dice scores)
"""

import argparse
import logging
import os
import sys
import json
import csv
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import default_setup, launch
from detectron2.evaluation import DatasetEvaluators, print_csv_format
from detectron2.utils.logger import setup_logger

from vesselseg.config import add_seg3d_config
from vesselseg.evaluation import CommonDiceEvaluator
from vesselseg.data import (
    VesselClineDeformDatasetMapper,
    build_vessel_transform_gen,
)
from train_utils import (
    get_dataset_dicts,
    inference_on_dataset,
    compute_dice_score,
)
from train_vessel_seg import VesselTrainer, get_dataset_mapper


def build_eval_loader(cfg, subset):
    """
    Build evaluation data loader for a specific subset.

    Args:
        cfg: config
        subset: 'train', 'val', or 'test'

    Returns:
        tuple: (DataLoader, dataset_dicts) for the specified subset
    """
    dataset_dicts = get_dataset_dicts(cfg.DATASETS.TEST, cfg, is_train=False, subset=subset)

    logger = logging.getLogger(__name__)
    logger.info(f"Loaded {len(dataset_dicts)} samples for {subset} set evaluation")

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


class PerSampleDiceEvaluator:
    """
    Evaluator that computes per-sample dice scores.
    """

    def __init__(self, dataset_name, cfg, dataset_dicts):
        self._dataset_name = dataset_name
        self._cfg = cfg
        self._dataset_dicts = dataset_dicts
        self._predictions = []
        self._sample_idx = 0

    def reset(self):
        self._predictions = []
        self._sample_idx = 0

    def process(self, inputs, outputs):
        """Process a batch of inputs and outputs."""
        for input_data, output in zip(inputs, outputs):
            # Get sample info
            if self._sample_idx < len(self._dataset_dicts):
                sample_info = self._dataset_dicts[self._sample_idx]
                file_id = sample_info.get('file_id', f'sample_{self._sample_idx}')
                case_id = os.path.basename(file_id).replace('.npz', '')
            else:
                case_id = f'sample_{self._sample_idx}'

            # Compute dice score
            dice_score = 0.0
            if 'seg' in output and 'seg' in input_data:
                pred_seg = output['seg']
                gt_seg = input_data['seg']

                if isinstance(pred_seg, torch.Tensor):
                    pred_seg = pred_seg.cpu().numpy()
                if isinstance(gt_seg, torch.Tensor):
                    gt_seg = gt_seg.cpu().numpy()

                # Binarize prediction
                pred_binary = (pred_seg > 0.5).astype(np.float32)
                gt_binary = (gt_seg > 0).astype(np.float32)

                # Compute dice
                intersection = (pred_binary * gt_binary).sum()
                union = pred_binary.sum() + gt_binary.sum()
                if union > 0:
                    dice_score = 2.0 * intersection / union
                else:
                    dice_score = 1.0 if pred_binary.sum() == 0 else 0.0

            self._predictions.append({
                'case_id': case_id,
                'dice': dice_score,
            })
            self._sample_idx += 1

    def evaluate(self):
        """Compute final metrics."""
        if not self._predictions:
            return {}

        dice_scores = [p['dice'] for p in self._predictions]
        results = {
            'mean_dice': float(np.mean(dice_scores)),
            'std_dice': float(np.std(dice_scores)),
            'min_dice': float(np.min(dice_scores)),
            'max_dice': float(np.max(dice_scores)),
            'median_dice': float(np.median(dice_scores)),
            'num_samples': len(dice_scores),
        }
        return results

    def get_per_sample_results(self):
        """Return per-sample results."""
        return self._predictions


def evaluate_on_subset(cfg, model, subset, output_dir=None):
    """
    Evaluate model on a specific subset.

    Args:
        cfg: config
        model: trained model
        subset: 'train', 'val', or 'test'
        output_dir: directory to save per-sample results

    Returns:
        dict: evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on {subset.upper()} set")
    logger.info(f"{'='*60}")

    data_loader, dataset_dicts = build_eval_loader(cfg, subset)

    if len(data_loader) == 0:
        logger.warning(f"No samples found for {subset} set!")
        return {}, []

    # Use both CommonDiceEvaluator and PerSampleDiceEvaluator
    common_evaluator = CommonDiceEvaluator(cfg.DATASETS.TEST[0], cfg)
    per_sample_evaluator = PerSampleDiceEvaluator(cfg.DATASETS.TEST[0], cfg, dataset_dicts)

    # Run inference
    results = inference_on_dataset(model, data_loader, common_evaluator, amp=False)

    # Get per-sample results by running inference again with per-sample evaluator
    # Reset data loader
    data_loader, _ = build_eval_loader(cfg, subset)
    per_sample_evaluator.reset()

    model.eval()
    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            per_sample_evaluator.process(inputs, outputs)

    per_sample_results = per_sample_evaluator.get_per_sample_results()
    per_sample_metrics = per_sample_evaluator.evaluate()

    # Merge results
    results.update(per_sample_metrics)

    if comm.is_main_process():
        logger.info(f"\n{subset.upper()} Set Results:")
        print_csv_format(results)

        # Save per-sample results to CSV
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            per_sample_csv = os.path.join(output_dir, f'per_sample_{subset}.csv')
            with open(per_sample_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['case_id', 'dice'])
                writer.writeheader()
                for row in per_sample_results:
                    writer.writerow(row)
            logger.info(f"Per-sample results saved to: {per_sample_csv}")

    return results, per_sample_results


def main(args):
    """Main evaluation function."""
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
        name="VesselSeg3D-Eval",
        abbrev_name="eval"
    )

    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = args.output_dir
    if output_dir and comm.is_main_process():
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir}")

    # Check if CSV split is specified
    split_csv = getattr(cfg.DATASETS, 'SPLIT_CSV', '')
    if not split_csv or not os.path.exists(split_csv):
        logger.warning("DATASETS.SPLIT_CSV not specified or file not found!")
        logger.warning("Evaluation will use hash-based splitting (TEST_FOLDS)")

    # Build model
    logger.info("Building model...")
    model = VesselTrainer.build_model(cfg)

    # Load weights
    logger.info(f"Loading weights from: {args.weights}")
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        args.weights, resume=False
    )
    model.eval()

    # Evaluate on each split
    all_results = OrderedDict()
    all_per_sample = OrderedDict()
    for subset in args.split:
        results, per_sample = evaluate_on_subset(cfg, model, subset, output_dir)
        all_results[subset] = results
        all_per_sample[subset] = per_sample

    # Print and save summary
    if comm.is_main_process():
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)

        for subset, results in all_results.items():
            if results:
                logger.info(f"\n{subset.upper()} Set:")
                for metric, value in results.items():
                    if isinstance(value, float):
                        logger.info(f"  {metric}: {value:.4f}")
                    else:
                        logger.info(f"  {metric}: {value}")

        # Print as table for easy copy-paste
        logger.info("\n" + "-"*60)
        logger.info("Results Table (for reporting):")
        logger.info("-"*60)
        header = ["Metric"] + [s.upper() for s in args.split]
        logger.info(" | ".join(f"{h:>12}" for h in header))
        logger.info("-" * (14 * len(header)))

        # Get all metric names
        all_metrics = set()
        for results in all_results.values():
            if results:
                all_metrics.update(results.keys())

        for metric in sorted(all_metrics):
            row = [metric[:12]]
            for subset in args.split:
                if subset in all_results and all_results[subset] and metric in all_results[subset]:
                    value = all_results[subset][metric]
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                else:
                    row.append("-")
            logger.info(" | ".join(f"{v:>12}" for v in row))

        # Save results to files
        if output_dir:
            # Save JSON with all results
            json_path = os.path.join(output_dir, 'eval_results.json')
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'weights': args.weights,
                'config': args.config_file,
                'splits': args.split,
                'results': {k: v for k, v in all_results.items()},
            }
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"\nJSON results saved to: {json_path}")

            # Save CSV summary table
            csv_path = os.path.join(output_dir, 'eval_results.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric'] + [s.upper() for s in args.split])
                for metric in sorted(all_metrics):
                    row = [metric]
                    for subset in args.split:
                        if subset in all_results and all_results[subset] and metric in all_results[subset]:
                            value = all_results[subset][metric]
                            if isinstance(value, float):
                                row.append(f"{value:.4f}")
                            else:
                                row.append(str(value))
                        else:
                            row.append("-")
                    writer.writerow(row)
            logger.info(f"CSV summary saved to: {csv_path}")

            logger.info("\n" + "="*60)
            logger.info("FILES SAVED:")
            logger.info("="*60)
            logger.info(f"  - {json_path}")
            logger.info(f"  - {csv_path}")
            for subset in args.split:
                logger.info(f"  - {os.path.join(output_dir, f'per_sample_{subset}.csv')}")
            logger.info("="*60)

    return all_results


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Evaluate Vessel Segmentation Model")
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
        default=["val"],
        choices=["train", "val", "test"],
        help="Which split(s) to evaluate on (default: val)",
    )
    parser.add_argument(
        "--output-dir",
        default="./eval_results",
        metavar="DIR",
        help="Directory to save evaluation results (default: ./eval_results)",
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
    print("Vessel Segmentation Evaluation")
    print("=" * 60)
    print(f"Config: {args.config_file}")
    print(f"Weights: {args.weights}")
    print(f"Splits: {args.split}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 60)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
