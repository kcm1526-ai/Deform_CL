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
"""

import logging
from collections import OrderedDict

import detectron2.utils.comm as comm
from train_utils import build_adamw_optimizer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator, print_csv_format
from detectron2.utils.logger import setup_logger

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
