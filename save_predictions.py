"""
Save prediction results as NIfTI files for visualization.

This script runs sliding window inference on ORIGINAL NIfTI files (not cropped patches)
and saves full-volume predictions that match the original image dimensions.

Usage:
    python save_predictions.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights ./outputs/vessel_deformcl/model_final.pth \
        --split val test \
        --output-dir ./predictions_latest \
        --image-dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/image \
        --label-dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/label \
        DATASETS.SPLIT_CSV /path/to/impulse2_rl.csv

Output:
    predictions_latest/
    ├── val/
    │   ├── lung3d_00025_pred.nii.gz   (binary prediction)
    │   ├── lung3d_00025_prob.nii.gz   (probability map)
    │   └── lung3d_00025_gt.nii.gz     (ground truth - copied from original)
    └── test/
        └── ...
"""

import argparse
import logging
import os
import csv
from collections import OrderedDict

import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from vesselseg.config import add_seg3d_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_split_csv(csv_path):
    """Load train/val/test split from CSV file."""
    split_map = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['pid'].strip()
            subset = row['subset'].strip().lower()
            split_map[pid] = subset
    return split_map


def get_cases_for_split(csv_path, subset):
    """Get list of case IDs for a specific split."""
    split_map = load_split_csv(csv_path)
    cases = [pid for pid, s in split_map.items() if s == subset]
    return cases


def setup_cfg(config_file, weights, opts=None):
    """Setup configuration for inference."""
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = weights
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    return cfg


def load_model(cfg, device='cuda'):
    """Load trained model for inference."""
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.to(device)
    return model


def load_nifti_image(file_path):
    """Load NIfTI image and return array with metadata."""
    sitk_img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(sitk_img)
    metadata = {
        'spacing': sitk_img.GetSpacing(),
        'origin': sitk_img.GetOrigin(),
        'direction': sitk_img.GetDirection(),
    }
    return img_array, metadata


def save_nifti_image(array, file_path, metadata, dtype=np.uint8):
    """Save numpy array as NIfTI file with metadata."""
    sitk_img = sitk.GetImageFromArray(array.astype(dtype))
    sitk_img.SetSpacing(metadata['spacing'])
    sitk_img.SetOrigin(metadata['origin'])
    sitk_img.SetDirection(metadata['direction'])
    sitk.WriteImage(sitk_img, file_path)


def preprocess_image(img_array, normalize=True, clip_range=(-1024, 3000)):
    """Preprocess image for model input."""
    img = np.clip(img_array, clip_range[0], clip_range[1]).astype(np.float32)
    if normalize:
        img = img / 1024.0
    img_tensor = torch.as_tensor(img[None, None, ...]).float()
    return img_tensor


def sliding_window_inference(model, image_tensor, window_size=(128, 128, 128),
                             overlap=0.5, device='cuda'):
    """
    Perform sliding window inference for large volumes.
    """
    _, _, D, H, W = image_tensor.shape
    wd, wh, ww = window_size

    # Calculate stride
    stride_d = int(wd * (1 - overlap))
    stride_h = int(wh * (1 - overlap))
    stride_w = int(ww * (1 - overlap))

    # Initialize output and count maps
    output_shape = (D, H, W)
    output = torch.zeros(output_shape, dtype=torch.float32, device='cpu')
    count = torch.zeros(output_shape, dtype=torch.float32, device='cpu')

    # Generate window positions
    d_positions = list(range(0, max(D - wd + 1, 1), stride_d))
    h_positions = list(range(0, max(H - wh + 1, 1), stride_h))
    w_positions = list(range(0, max(W - ww + 1, 1), stride_w))

    # Ensure we cover the entire volume
    if len(d_positions) == 0:
        d_positions = [0]
    if len(h_positions) == 0:
        h_positions = [0]
    if len(w_positions) == 0:
        w_positions = [0]

    if d_positions[-1] + wd < D:
        d_positions.append(max(0, D - wd))
    if h_positions[-1] + wh < H:
        h_positions.append(max(0, H - wh))
    if w_positions[-1] + ww < W:
        w_positions.append(max(0, W - ww))

    # Remove duplicates and sort
    d_positions = sorted(set(d_positions))
    h_positions = sorted(set(h_positions))
    w_positions = sorted(set(w_positions))

    total_windows = len(d_positions) * len(h_positions) * len(w_positions)

    with torch.no_grad():
        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    # Handle edge cases where window extends beyond volume
                    d_end = min(d + wd, D)
                    h_end = min(h + wh, H)
                    w_end = min(w + ww, W)

                    # Adjust start if window would be smaller than expected
                    d_start = max(0, d_end - wd)
                    h_start = max(0, h_end - wh)
                    w_start = max(0, w_end - ww)

                    # Extract window
                    window = image_tensor[:, :, d_start:d_start+wd, h_start:h_start+wh, w_start:w_start+ww]

                    # Pad if needed
                    pad_d = wd - window.shape[2]
                    pad_h = wh - window.shape[3]
                    pad_w = ww - window.shape[4]

                    if pad_d > 0 or pad_h > 0 or pad_w > 0:
                        window = torch.nn.functional.pad(
                            window, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0
                        )

                    window = window.to(device)

                    # Prepare input for model
                    inputs = [{
                        'image': window[0],
                        'file_id': 'inference',
                    }]

                    # Run inference
                    outputs = model(inputs)

                    # Extract segmentation prediction
                    pred = None
                    if 'seg' in outputs[0]:
                        pred = outputs[0]['seg']
                    elif 'seg_pred' in outputs[0]:
                        pred = outputs[0]['seg_pred']
                    elif 'seg_logits' in outputs[0]:
                        pred = outputs[0]['seg_logits']
                    else:
                        for key in outputs[0]:
                            if isinstance(outputs[0][key], torch.Tensor):
                                pred = outputs[0][key]
                                break

                    if pred is not None:
                        # Apply sigmoid if logits
                        if pred.min() < 0 or pred.max() > 1:
                            pred = pred.sigmoid()
                        pred = pred.cpu()

                        # Remove batch/channel dimensions if present
                        while pred.dim() > 3:
                            pred = pred.squeeze(0)

                        # Only use the valid part (without padding)
                        valid_d = min(wd, D - d_start)
                        valid_h = min(wh, H - h_start)
                        valid_w = min(ww, W - w_start)

                        # Accumulate predictions
                        output[d_start:d_start+valid_d, h_start:h_start+valid_h, w_start:w_start+valid_w] += pred[:valid_d, :valid_h, :valid_w]
                        count[d_start:d_start+valid_d, h_start:h_start+valid_h, w_start:w_start+valid_w] += 1

    # Average overlapping predictions
    output = output / count.clamp(min=1)
    return output.numpy()


def find_nifti_file(directory, case_id):
    """Find NIfTI file matching case_id in directory."""
    for ext in ['.nii.gz', '.nii']:
        # Try exact match
        path = os.path.join(directory, f'{case_id}{ext}')
        if os.path.exists(path):
            return path

        # Try with common suffixes/prefixes
        for filename in os.listdir(directory):
            if case_id in filename and filename.endswith(ext):
                return os.path.join(directory, filename)

    return None


def process_case(model, case_id, image_dir, label_dir, output_dir, cfg, device,
                 save_prob=True, threshold=0.5):
    """
    Process a single case: load image, run inference, save predictions.
    """
    # Find image file
    image_path = find_nifti_file(image_dir, case_id)
    if image_path is None:
        logger.warning(f"Image not found for case {case_id} in {image_dir}")
        return False

    # Find label file
    label_path = find_nifti_file(label_dir, case_id)
    if label_path is None:
        logger.warning(f"Label not found for case {case_id} in {label_dir}")
        # Continue without GT

    try:
        # Load image
        img_array, metadata = load_nifti_image(image_path)
        logger.debug(f"Loaded image: {image_path}, shape: {img_array.shape}")

        # Preprocess
        img_tensor = preprocess_image(img_array)

        # Get crop size from config
        crop_size = tuple(cfg.INPUT.CROP_SIZE_TRAIN)

        # Run sliding window inference
        pred_prob = sliding_window_inference(
            model, img_tensor,
            window_size=crop_size,
            overlap=0.5,
            device=device
        )

        # Resize prediction to match original image size if needed
        if pred_prob.shape != img_array.shape:
            import torch.nn.functional as F
            pred_tensor = torch.tensor(pred_prob).unsqueeze(0).unsqueeze(0)
            pred_tensor = F.interpolate(
                pred_tensor,
                size=img_array.shape,
                mode='trilinear',
                align_corners=False
            )
            pred_prob = pred_tensor.squeeze().numpy()

        # Save probability map
        if save_prob:
            prob_path = os.path.join(output_dir, f'{case_id}_prob.nii.gz')
            save_nifti_image(pred_prob, prob_path, metadata, dtype=np.float32)

        # Save binary prediction
        pred_binary = (pred_prob > threshold).astype(np.uint8)
        pred_path = os.path.join(output_dir, f'{case_id}_pred.nii.gz')
        save_nifti_image(pred_binary, pred_path, metadata, dtype=np.uint8)

        # Copy ground truth (original, unmodified)
        if label_path:
            gt_array, gt_metadata = load_nifti_image(label_path)
            gt_path = os.path.join(output_dir, f'{case_id}_gt.nii.gz')
            save_nifti_image(gt_array, gt_path, gt_metadata, dtype=np.uint8)

        return True

    except Exception as e:
        logger.error(f"Error processing {case_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main(args):
    """Main function."""
    # Setup config
    cfg = setup_cfg(args.config_file, args.weights, args.opts)

    logger.info(f"Image directory: {args.image_dir}")
    logger.info(f"Label directory: {args.label_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Get CSV split file
    split_csv = getattr(cfg.DATASETS, 'SPLIT_CSV', '')
    if not split_csv or not os.path.exists(split_csv):
        logger.error("DATASETS.SPLIT_CSV not specified or file not found!")
        logger.error("Please specify the CSV file path using DATASETS.SPLIT_CSV option")
        return

    logger.info(f"Using split CSV: {split_csv}")

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = load_model(cfg, device)
    logger.info("Model loaded successfully")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each split
    all_results = OrderedDict()
    for subset in args.split:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {subset.upper()} set")
        logger.info(f"{'='*60}")

        # Get cases for this split
        cases = get_cases_for_split(split_csv, subset)
        logger.info(f"Found {len(cases)} cases for {subset} set")

        if len(cases) == 0:
            logger.warning(f"No cases found for {subset} set!")
            continue

        # Create output directory for this subset
        subset_output_dir = os.path.join(args.output_dir, subset)
        os.makedirs(subset_output_dir, exist_ok=True)

        # Process each case
        successful = 0
        failed = 0

        for case_id in tqdm(cases, desc=f"Processing {subset}"):
            success = process_case(
                model=model,
                case_id=case_id,
                image_dir=args.image_dir,
                label_dir=args.label_dir,
                output_dir=subset_output_dir,
                cfg=cfg,
                device=device,
                save_prob=args.save_prob,
                threshold=args.threshold
            )

            if success:
                successful += 1
            else:
                failed += 1

        all_results[subset] = {'successful': successful, 'failed': failed}
        logger.info(f"{subset.upper()}: {successful} successful, {failed} failed")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    for subset, result in all_results.items():
        logger.info(f"{subset.upper()}: {result['successful']} successful, {result['failed']} failed")

    logger.info("\nOutput structure:")
    logger.info(f"  {args.output_dir}/")
    for subset in args.split:
        logger.info(f"  ├── {subset}/")
        logger.info(f"  │   ├── <case_id>_pred.nii.gz  (binary prediction)")
        if args.save_prob:
            logger.info(f"  │   ├── <case_id>_prob.nii.gz  (probability map)")
        logger.info(f"  │   └── <case_id>_gt.nii.gz    (ground truth from original)")

    logger.info("\nThe GT files are now EXACT COPIES of the original labels!")
    logger.info("You can compare predictions directly with original labels.")


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Save Vessel Segmentation Predictions (Full Volume)")
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
        "--image-dir",
        default="/raid/users/ai_kcm_0/M3DVBAV_CropLung/image",
        metavar="DIR",
        help="Directory containing original NIfTI images",
    )
    parser.add_argument(
        "--label-dir",
        default="/raid/users/ai_kcm_0/M3DVBAV_CropLung/label",
        metavar="DIR",
        help="Directory containing original NIfTI labels",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary segmentation (default: 0.5)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
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
    print("Save Vessel Segmentation Predictions (Full Volume)")
    print("=" * 60)
    print(f"Config: {args.config_file}")
    print(f"Weights: {args.weights}")
    print(f"Splits: {args.split}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Image Dir: {args.image_dir}")
    print(f"Label Dir: {args.label_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Save Probability Maps: {args.save_prob}")
    print(f"GPU: {args.gpu}")
    print("=" * 60)

    main(args)
