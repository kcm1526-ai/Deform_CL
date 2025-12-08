"""
Inference script for Vessel Segmentation using DeformCL.

This script performs inference on new CT/MRI images and saves the
predicted vessel segmentation masks as NIfTI files.

Usage:
    # Single file inference
    python inference_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights outputs/vessel_deformcl/model_final.pth \
        --input /path/to/image.nii.gz \
        --output /path/to/output_dir

    # Batch inference on a directory
    python inference_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights outputs/vessel_deformcl/model_final.pth \
        --input-dir /path/to/images \
        --output /path/to/output_dir

    # With GPU selection
    python inference_vessel_seg.py \
        --config-file configs/vessel_deformcl.yaml \
        --weights outputs/vessel_deformcl/model_final.pth \
        --input-dir /path/to/images \
        --output /path/to/output_dir \
        --gpu 0
"""

import os
import argparse
import logging
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from vesselseg.config import add_seg3d_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_cfg(config_file, weights, opts=None):
    """
    Setup configuration for inference.
    """
    cfg = get_cfg()
    add_seg3d_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)

    # Set model weights
    cfg.MODEL.WEIGHTS = weights

    # Set to single GPU inference mode
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1

    cfg.freeze()
    return cfg


def load_model(cfg, device='cuda'):
    """
    Load trained model for inference.
    """
    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    model.to(device)
    return model


def load_nifti_image(file_path):
    """
    Load NIfTI image and return array with metadata.
    """
    sitk_img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(sitk_img)

    metadata = {
        'spacing': sitk_img.GetSpacing(),
        'origin': sitk_img.GetOrigin(),
        'direction': sitk_img.GetDirection(),
    }

    return img_array, metadata


def save_nifti_image(array, file_path, metadata):
    """
    Save numpy array as NIfTI file with metadata.
    """
    sitk_img = sitk.GetImageFromArray(array.astype(np.uint8))
    sitk_img.SetSpacing(metadata['spacing'])
    sitk_img.SetOrigin(metadata['origin'])
    sitk_img.SetDirection(metadata['direction'])
    sitk.WriteImage(sitk_img, file_path)


def preprocess_image(img_array, normalize=True, clip_range=(-1024, 3000)):
    """
    Preprocess image for model input.
    """
    # Clip and normalize
    img = np.clip(img_array, clip_range[0], clip_range[1]).astype(np.float32)

    if normalize:
        img = img / 1024.0

    # Add batch and channel dimensions
    img_tensor = torch.as_tensor(img[None, None, ...]).float()

    return img_tensor


def sliding_window_inference(model, image_tensor, window_size=(128, 128, 128),
                             overlap=0.5, device='cuda'):
    """
    Perform sliding window inference for large volumes.

    Args:
        model: Trained model
        image_tensor: Input image tensor (B, C, D, H, W)
        window_size: Size of sliding window
        overlap: Overlap ratio between windows
        device: Device to run inference on

    Returns:
        Predicted segmentation mask
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
    if d_positions[-1] + wd < D:
        d_positions.append(D - wd)
    if h_positions[-1] + wh < H:
        h_positions.append(H - wh)
    if w_positions[-1] + ww < W:
        w_positions.append(W - ww)

    total_windows = len(d_positions) * len(h_positions) * len(w_positions)
    logger.info(f"Processing {total_windows} windows...")

    with torch.no_grad():
        for d in d_positions:
            for h in h_positions:
                for w in w_positions:
                    # Extract window
                    window = image_tensor[:, :, d:d+wd, h:h+wh, w:w+ww].to(device)

                    # Prepare input for model
                    inputs = [{
                        'image': window[0],  # Remove batch dimension for model
                        'file_id': 'inference',
                    }]

                    # Run inference
                    outputs = model(inputs)

                    # Extract segmentation prediction
                    if 'seg_pred' in outputs[0]:
                        pred = outputs[0]['seg_pred'].sigmoid().cpu()
                    elif 'seg_logits' in outputs[0]:
                        pred = outputs[0]['seg_logits'].sigmoid().cpu()
                    else:
                        # Fallback: try to get any prediction tensor
                        for key in outputs[0]:
                            if isinstance(outputs[0][key], torch.Tensor):
                                pred = outputs[0][key].sigmoid().cpu()
                                break

                    # Remove batch/channel dimensions if present
                    while pred.dim() > 3:
                        pred = pred.squeeze(0)

                    # Accumulate predictions
                    output[d:d+wd, h:h+wh, w:w+ww] += pred
                    count[d:d+wd, h:h+wh, w:w+ww] += 1

    # Average overlapping predictions
    output = output / count.clamp(min=1)

    return output.numpy()


def simple_inference(model, image_tensor, device='cuda'):
    """
    Simple inference for small volumes that fit in memory.

    Args:
        model: Trained model
        image_tensor: Input image tensor (B, C, D, H, W)
        device: Device to run inference on

    Returns:
        Predicted segmentation mask
    """
    with torch.no_grad():
        # Prepare input for model
        inputs = [{
            'image': image_tensor[0].to(device),  # Remove batch dimension
            'file_id': 'inference',
        }]

        # Run inference
        outputs = model(inputs)

        # Extract segmentation prediction
        if 'seg_pred' in outputs[0]:
            pred = outputs[0]['seg_pred'].sigmoid().cpu()
        elif 'seg_logits' in outputs[0]:
            pred = outputs[0]['seg_logits'].sigmoid().cpu()
        else:
            # Fallback: try to get any prediction tensor
            for key in outputs[0]:
                if isinstance(outputs[0][key], torch.Tensor):
                    pred = outputs[0][key].sigmoid().cpu()
                    break

        # Remove batch/channel dimensions if present
        while pred.dim() > 3:
            pred = pred.squeeze(0)

    return pred.numpy()


def infer_single_image(model, image_path, output_path, cfg, device='cuda',
                       use_sliding_window=True, threshold=0.5):
    """
    Perform inference on a single image.

    Args:
        model: Trained model
        image_path: Path to input NIfTI file
        output_path: Path to save output NIfTI file
        cfg: Configuration
        device: Device for inference
        use_sliding_window: Whether to use sliding window inference
        threshold: Threshold for binary segmentation

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        img_array, metadata = load_nifti_image(image_path)
        logger.info(f"Loaded image: {image_path}, shape: {img_array.shape}")

        # Preprocess
        img_tensor = preprocess_image(img_array)

        # Get crop size from config
        crop_size = cfg.INPUT.CROP_SIZE_TRAIN

        # Decide on inference method based on image size
        volume_size = img_array.shape
        use_sliding = use_sliding_window and any(
            v > c * 1.5 for v, c in zip(volume_size, crop_size)
        )

        # Run inference
        if use_sliding:
            logger.info("Using sliding window inference...")
            pred_prob = sliding_window_inference(
                model, img_tensor,
                window_size=crop_size,
                overlap=0.5,
                device=device
            )
        else:
            logger.info("Using simple inference...")
            pred_prob = simple_inference(model, img_tensor, device=device)

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

        # Apply threshold
        pred_mask = (pred_prob > threshold).astype(np.uint8)

        # Save prediction
        save_nifti_image(pred_mask, output_path, metadata)
        logger.info(f"Saved prediction to: {output_path}")

        # Also save probability map if requested
        prob_path = output_path.replace('.nii.gz', '_prob.nii.gz')
        prob_array = (pred_prob * 255).astype(np.uint8)
        save_nifti_image(prob_array, prob_path, metadata)
        logger.info(f"Saved probability map to: {prob_path}")

        return True

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Vessel Segmentation Inference')
    parser.add_argument('--config-file', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input NIfTI file')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Directory containing input NIfTI files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for predictions')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary segmentation')
    parser.add_argument('--no-sliding-window', action='store_true',
                       help='Disable sliding window inference')
    parser.add_argument('--opts', nargs='*', default=[],
                       help='Additional config options')

    args = parser.parse_args()

    # Validate input arguments
    if args.input is None and args.input_dir is None:
        parser.error("Either --input or --input-dir must be specified")

    # Setup device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Setup configuration
    cfg = setup_cfg(args.config_file, args.weights, args.opts)

    # Load model
    logger.info("Loading model...")
    model = load_model(cfg, device)
    logger.info("Model loaded successfully")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Gather input files
    if args.input:
        input_files = [args.input]
    else:
        input_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.endswith('.nii.gz') or f.endswith('.nii')
        ])

    logger.info(f"Found {len(input_files)} files to process")

    # Process each file
    successful = 0
    failed = 0

    for input_file in tqdm(input_files, desc="Processing"):
        # Generate output filename
        basename = os.path.basename(input_file)
        if not basename.endswith('.nii.gz'):
            basename = basename.replace('.nii', '.nii.gz')
        output_file = os.path.join(args.output, basename.replace('.nii.gz', '_seg.nii.gz'))

        # Run inference
        success = infer_single_image(
            model=model,
            image_path=input_file,
            output_path=output_file,
            cfg=cfg,
            device=device,
            use_sliding_window=not args.no_sliding_window,
            threshold=args.threshold
        )

        if success:
            successful += 1
        else:
            failed += 1

    logger.info(f"Inference complete: {successful} successful, {failed} failed")


if __name__ == "__main__":
    main()
