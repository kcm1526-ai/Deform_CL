"""
Training script for Vessel Segmentation using DeformCL.

This script provides:
1. Training with DeformCL method (centerline + segmentation)
2. Training with UNet baseline (segmentation only)
3. Evaluation and inference

Usage:
    # Step 1: Process your NIfTI dataset
    python process_vessel_dataset.py \
        --image_dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/image \
        --label_dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/label \
        --output_dir ./VesselSeg_Data

    # Step 2: Train DeformCL model (single GPU)
    python train_vessel_seg.py --num-gpus 1 \
        --config-file configs/vessel_deformcl.yaml \
        OUTPUT_DIR ./outputs/vessel_deformcl

    # Step 2 Alternative: Train UNet baseline
    python train_vessel_seg.py --num-gpus 1 \
        --config-file configs/vessel_unet.yaml \
        OUTPUT_DIR ./outputs/vessel_unet

    # Step 3: Evaluation only
    python train_vessel_seg.py --num-gpus 1 --eval-only \
        --config-file configs/vessel_deformcl.yaml \
        MODEL.WEIGHTS ./outputs/vessel_deformcl/model_final.pth

    # Multi-GPU training
    python train_vessel_seg.py --num-gpus 4 --dist-url auto \
        --config-file configs/vessel_deformcl.yaml

    # Training with CSV-based train/val/test split
    python train_vessel_seg.py --num-gpus 4 --dist-url auto \
        --config-file configs/vessel_deformcl.yaml \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv
"""

import logging
from collections import OrderedDict

import torch
import numpy as np
import detectron2.utils.comm as comm
from train_utils import build_adamw_optimizer, compute_dice_score
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, HookBase
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage

from vesselseg.config import add_seg3d_config
from vesselseg.evaluation import (
    CommonDiceEvaluator,
)
from vesselseg.data import (
    ClineDeformDatasetMapper,
    VesselSegDatasetMapper,
    VesselClineDeformDatasetMapper,
    build_cline_deform_transform_gen,
    build_bbox_transform_gen,
    build_vessel_transform_gen,
)
from train_utils import (
    build_train_loader,
    build_test_loader,
    inference_on_dataset,
)


class DiceAccuracyHook(HookBase):
    """
    Hook to compute and log dice accuracy during training.
    Computes dice on the training batch every N iterations.
    """

    def __init__(self, log_period=100):
        """
        Args:
            log_period: compute and log dice every N iterations
        """
        self.log_period = log_period
        self._logger = logging.getLogger(__name__)

    def after_step(self):
        """Called after each training step."""
        # Only compute dice periodically to avoid slowdown
        if (self.trainer.iter + 1) % self.log_period != 0:
            return

        # Get the last batch outputs from trainer storage
        # The model's forward pass during training returns losses, but we need predictions
        # We'll compute dice from the storage if available
        storage = get_event_storage()

        # Check if we have dice-related metrics in storage
        # The model may already compute dice loss, which we can use as proxy
        try:
            # Try to get dice-related loss if available
            if hasattr(self.trainer, '_last_dice_score'):
                dice = self.trainer._last_dice_score
                storage.put_scalar("train_dice", dice)
                if comm.is_main_process():
                    self._logger.info(f"[Iter {self.trainer.iter + 1}] Train Dice: {dice:.4f}")
        except Exception:
            pass


def get_dataset_mapper(cfg, is_train=False):
    """
    Get the appropriate dataset mapper based on model architecture.

    For vessel segmentation, we use VesselClineDeformDatasetMapper which
    doesn't require pre-computed bounding boxes.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    dataset_name = cfg.DATASETS.TRAIN[0] if is_train else cfg.DATASETS.TEST[0]

    # Use VesselClineDeformDatasetMapper for VesselSeg dataset
    if 'VesselSeg' in dataset_name:
        mapper_func = VesselClineDeformDatasetMapper
        transform_builder = build_vessel_transform_gen
    elif meta_arch != 'Bbox3d':
        mapper_func = ClineDeformDatasetMapper
        transform_builder = build_cline_deform_transform_gen
    else:
        mapper_func = VesselSegDatasetMapper
        transform_builder = build_bbox_transform_gen

    return mapper_func(cfg, transform_builder, is_train)


class VesselTrainer(DefaultTrainer):
    """
    Trainer class for vessel segmentation.
    """

    def __init__(self, cfg):
        super(VesselTrainer, self).__init__(cfg)
        self._last_dice_score = 0.0
        self._dice_scores = []
        self._dice_log_period = 100  # Log dice every N iterations

    def build_hooks(self):
        """Build training hooks including dice accuracy hook."""
        hooks = super().build_hooks()
        # Add dice accuracy hook
        hooks.insert(-1, DiceAccuracyHook(log_period=self._dice_log_period))
        return hooks

    def run_step(self):
        """
        Override run_step to compute dice accuracy during training.
        """
        assert self.model.training, "[VesselTrainer] model was changed to eval mode!"
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        data = next(self._data_loader_iter)
        data_time = start.elapsed_time(end) / 1000.0  # seconds

        # Forward pass
        loss_dict = self.model(data)

        # Compute dice score from segmentation predictions if available
        if (self.iter + 1) % self._dice_log_period == 0:
            try:
                # Try to get dice from loss dict (some models return dice-related losses)
                if 'loss_dice' in loss_dict:
                    # Dice loss is 1 - dice, so dice = 1 - loss_dice
                    dice = 1.0 - loss_dict['loss_dice'].item()
                    self._last_dice_score = dice
                    self._dice_scores.append(dice)
                elif 'loss_seg' in loss_dict:
                    # For seg loss, we'll estimate dice from loss
                    # Lower loss generally means higher dice
                    seg_loss = loss_dict['loss_seg'].item()
                    dice = max(0.0, 1.0 - seg_loss)  # Approximate
                    self._last_dice_score = dice
                    self._dice_scores.append(dice)
            except Exception:
                pass

        # Compute total loss
        losses = sum(loss_dict.values())

        # Backward pass
        self.optimizer.zero_grad()
        losses.backward()

        # Log metrics
        self._write_metrics(loss_dict, data_time)

        # Optimizer step
        self.optimizer.step()

    def _write_metrics(self, loss_dict, data_time, prefix=""):
        """Write metrics to storage."""
        storage = get_event_storage()

        # Write losses
        total_loss = 0.0
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            total_loss += v
            storage.put_scalar(f"{prefix}{k}", v)
        storage.put_scalar(f"{prefix}total_loss", total_loss)
        storage.put_scalar("data_time", data_time)

        # Write dice if available
        if self._last_dice_score > 0 and (self.iter + 1) % self._dice_log_period == 0:
            storage.put_scalar("train_dice", self._last_dice_score)
            if comm.is_main_process():
                avg_dice = np.mean(self._dice_scores[-10:]) if len(self._dice_scores) > 0 else 0.0
                logging.getLogger(__name__).info(
                    f"[Iter {self.iter + 1}] Train Dice: {self._last_dice_score:.4f} (avg: {avg_dice:.4f})"
                )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build evaluator for vessel segmentation."""
        evaluators = []
        evaluators.append(CommonDiceEvaluator(dataset_name, cfg))
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test data loader."""
        return build_test_loader(
            cfg, dataset_name, mapper=get_dataset_mapper(cfg, is_train=False)
        )

    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader."""
        return build_train_loader(
            cfg, mapper=get_dataset_mapper(cfg, is_train=True)
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        """Build AdamW optimizer with gradient clipping."""
        return build_adamw_optimizer(cfg, model)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the model on test dataset.

        Args:
            cfg (CfgNode): Configuration
            model (nn.Module): Model to evaluate
            evaluators (list[DatasetEvaluator] or None): Custom evaluators

        Returns:
            dict: Evaluation results (e.g., Dice scores)
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `VesselTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator, amp=False)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(
        output=cfg.OUTPUT_DIR,
        distributed_rank=comm.get_rank(),
        name="VesselSeg3D",
        abbrev_name="vessel"
    )
    return cfg


def main(args):
    """Main entry point for training/evaluation."""
    cfg = setup(args)

    if args.eval_only:
        # Evaluation mode
        model = VesselTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = VesselTrainer.test(cfg, model)
        return res

    # Training mode
    trainer = VesselTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("=" * 60)
    print("Vessel Segmentation Training with DeformCL")
    print("=" * 60)
    print("Command Line Args:", args)
    print("=" * 60)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
