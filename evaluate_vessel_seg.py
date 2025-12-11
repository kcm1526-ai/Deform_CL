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
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

    # Evaluate on test set
    python evaluate_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split test \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

    # Evaluate on both val and test
    python evaluate_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split val test \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

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
)
from train_vessel_seg import VesselTrainer, get_dataset_mapper


def build_eval_loader(cfg, subset):
    """
    Build evaluation data loader for a specific subset.

    Args:
        cfg: config
        subset: 'train', 'val', or 'test'

    Returns:
        DataLoader for the specified subset
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
    return data_loader


def evaluate_on_subset(cfg, model, subset):
    """
    Evaluate model on a specific subset.

    Args:
        cfg: config
        model: trained model
        subset: 'train', 'val', or 'test'

    Returns:
        dict: evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating on {subset.upper()} set")
    logger.info(f"{'='*60}")

    data_loader = build_eval_loader(cfg, subset)

    if len(data_loader) == 0:
        logger.warning(f"No samples found for {subset} set!")
        return {}

    evaluator = CommonDiceEvaluator(cfg.DATASETS.TEST[0], cfg)
    results = inference_on_dataset(model, data_loader, evaluator, amp=False)

    if comm.is_main_process():
        logger.info(f"\n{subset.upper()} Set Results:")
        print_csv_format(results)

    return results


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
    for subset in args.split:
        results = evaluate_on_subset(cfg, model, subset)
        all_results[subset] = results

    # Print summary
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
    print("=" * 60)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
