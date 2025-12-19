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
import torch.nn as nn
import numpy as np
import detectron2.utils.comm as comm
from train_utils import build_adamw_optimizer, compute_dice_score
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, HookBase
from detectron2.engine.train_loop import SimpleTrainer, AMPTrainer
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


class MemoryCleanupHook(HookBase):
    """
    Hook to periodically clear CUDA cache to prevent memory fragmentation.
    This helps prevent OOM errors during long training runs.
    """

    def __init__(self, cleanup_period=50):
        """
        Args:
            cleanup_period: clear CUDA cache every N iterations
        """
        self.cleanup_period = cleanup_period
        self._logger = logging.getLogger(__name__)

    def after_step(self):
        """Called after each training step."""
        if (self.trainer.iter + 1) % self.cleanup_period == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DiceAccuracyHook(HookBase):
    """
    Hook to compute and log dice accuracy during training.
    Estimates dice from the dice loss logged during training.
    """

    def __init__(self, log_period=100):
        """
        Args:
            log_period: compute and log dice every N iterations
        """
        self.log_period = log_period
        self._logger = logging.getLogger(__name__)
        self._dice_scores = []

    def after_step(self):
        """Called after each training step."""
        # Only log periodically to avoid slowdown
        if (self.trainer.iter + 1) % self.log_period != 0:
            return

        storage = get_event_storage()

        try:
            # Try to get dice from loss values in storage
            # The model logs loss_dice or loss_seg which we can use to estimate dice
            history = storage.histories()

            dice_score = None

            # Check for loss_dice (dice loss = 1 - dice, so dice = 1 - loss)
            if 'loss_dice' in history:
                loss_dice = history['loss_dice'].latest()
                dice_score = 1.0 - loss_dice
            # Check for loss_seg as fallback
            elif 'loss_seg' in history:
                loss_seg = history['loss_seg'].latest()
                dice_score = max(0.0, 1.0 - loss_seg)

            if dice_score is not None:
                self._dice_scores.append(dice_score)
                avg_dice = np.mean(self._dice_scores[-10:]) if len(self._dice_scores) > 0 else dice_score

                storage.put_scalar("train_dice", dice_score)
                if comm.is_main_process():
                    self._logger.info(
                        f"[Iter {self.trainer.iter + 1}] Train Dice: {dice_score:.4f} (avg: {avg_dice:.4f})"
                    )
        except Exception as e:
            # Silently ignore errors in dice computation
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


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model with find_unused_parameters=True.
    This is needed because MedNeXt has parameters (dummy_tensor, deep supervision outputs)
    that may not be used in every forward pass.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    if comm.get_world_size() == 1:
        return model

    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]

    # Enable find_unused_parameters to handle unused deep supervision / gradient checkpoint params
    kwargs["find_unused_parameters"] = True

    ddp = DDP(model, **kwargs)
    return ddp


class VesselTrainer(DefaultTrainer):
    """
    Trainer class for vessel segmentation.
    """

    def __init__(self, cfg):
        # Initialize dice log period BEFORE calling super().__init__() because
        # the parent's __init__ calls build_hooks() which needs this
        self._dice_log_period = 100  # Log dice every N iterations

        # We need custom DDP initialization with find_unused_parameters=True
        # This is required because MedNeXt backbone has parameters that may not be
        # used in every forward pass (dummy_tensor, deep supervision outputs)
        self._custom_init(cfg)

    def _custom_init(self, cfg):
        """
        Custom initialization that mimics DefaultTrainer but uses find_unused_parameters=True
        for DDP to handle MedNeXt's unused parameters.
        """
        import os
        import weakref
        from detectron2.solver import build_lr_scheduler
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.utils.events import EventStorage

        logger = logging.getLogger("detectron2")

        # Initialize base class attributes (from TrainerBase)
        self._hooks = []
        self.iter = 0
        self.start_iter = 0
        self.storage = None

        # Build model
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # Wrap model in DDP with find_unused_parameters=True
        model = create_ddp_model(model, broadcast_buffers=False)

        # Enable static graph for gradient checkpointing compatibility with DDP
        # This is needed when using torch.utils.checkpoint with DDP
        if hasattr(model, '_set_static_graph'):
            model._set_static_graph()

        # Create trainer (SimpleTrainer or AMPTrainer)
        # Note: detectron2's default uses SimpleTrainer. If AMP is needed, use AMPTrainer.
        self._trainer = SimpleTrainer(model, data_loader, optimizer)

        self.scheduler = build_lr_scheduler(cfg, optimizer)

        # Ensure output directory exists
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        # Store model reference
        self.model = model

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """Build training hooks including dice accuracy hook."""
        from detectron2.engine import hooks as d2_hooks
        from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter

        cfg = self.cfg
        ret = [
            d2_hooks.IterationTimer(),
            d2_hooks.LRScheduler(),
        ]

        # Periodic checkpointing
        if cfg.SOLVER.CHECKPOINT_PERIOD > 0:
            ret.append(
                d2_hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        # Periodic evaluation with memory cleanup
        if len(cfg.DATASETS.TEST) > 0 and cfg.TEST.EVAL_PERIOD > 0:
            def eval_with_cleanup():
                # Clear CUDA cache before evaluation to free training memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return self.test(cfg, self.model)

            ret.append(
                d2_hooks.EvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    eval_with_cleanup,
                )
            )

        # Add memory cleanup hook to prevent OOM from memory fragmentation
        ret.append(MemoryCleanupHook(cleanup_period=50))

        # Add dice accuracy hook
        ret.append(DiceAccuracyHook(log_period=self._dice_log_period))

        # Periodic writer (should be last)
        if comm.is_main_process():
            ret.append(d2_hooks.PeriodicWriter(
                [
                    CommonMetricPrinter(self.max_iter),
                    JSONWriter(f"{cfg.OUTPUT_DIR}/metrics.json"),
                    TensorboardXWriter(cfg.OUTPUT_DIR),
                ],
                period=cfg.SOLVER.LOG_PERIOD if hasattr(cfg.SOLVER, 'LOG_PERIOD') else 20
            ))

        return ret

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
    import os
    parser = default_argument_parser()
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '2,3,4,5'). "
             "If not specified, uses all available GPUs."
    )
    args = parser.parse_args()

    # Set CUDA_VISIBLE_DEVICES if --gpus is specified
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"Using GPUs: {args.gpus}")
        # Update num_gpus to match the number of specified GPUs
        specified_gpus = len(args.gpus.split(","))
        if args.num_gpus > specified_gpus:
            print(f"Warning: --num-gpus ({args.num_gpus}) > specified GPUs ({specified_gpus}). "
                  f"Using {specified_gpus} GPUs.")
            args.num_gpus = specified_gpus

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
