"""
Process Vessel Segmentation Dataset from NIfTI files.
Converts NIfTI images and labels to NPZ format with centerline extraction.

Dataset structure:
    - images: /raid/users/ai_kcm_0/M3DVBAV_CropLung/image/*.nii.gz
    - labels: /raid/users/ai_kcm_0/M3DVBAV_CropLung/label/*.nii.gz
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from tqdm import tqdm
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_nifti(file_path):
    """Load NIfTI file and return numpy array with spacing info."""
    sitk_img = sitk.ReadImage(file_path)
    img_array = sitk.GetArrayFromImage(sitk_img)
    spacing = sitk_img.GetSpacing()
    origin = sitk_img.GetOrigin()
    direction = sitk_img.GetDirection()
    return img_array, spacing, origin, direction


def extract_centerline(seg_mask):
    """
    Extract centerline from segmentation mask using skeletonization.

    Args:
        seg_mask: Binary segmentation mask (numpy array)

    Returns:
        Centerline mask with same labels as input segmentation
    """
    # Create binary mask
    binary_mask = (seg_mask > 0).astype(np.uint8)

    # Apply 3D skeletonization
    skeleton = morphology.skeletonize_3d(binary_mask).astype(np.uint8)

    # Create centerline map preserving original labels
    cline_map = seg_mask.copy()
    cline_map[skeleton == 0] = 0

    return cline_map


def normalize_image(img, clip_range=(-1024, 3000)):
    """
    Normalize image intensity for CT/MRI.

    Args:
        img: Input image array
        clip_range: (min, max) values for clipping

    Returns:
        Normalized image
    """
    img = np.clip(img, clip_range[0], clip_range[1])
    return img.astype(np.float32)


def process_single_case(img_path, label_path, output_path, case_id):
    """
    Process a single case and save as NPZ file.

    Args:
        img_path: Path to image NIfTI file
        label_path: Path to label NIfTI file
        output_path: Output directory
        case_id: Case identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image and label
        img_array, spacing, origin, direction = load_nifti(img_path)
        label_array, _, _, _ = load_nifti(label_path)

        # Ensure label is integer type
        label_array = label_array.astype(np.uint8)

        # Normalize image (do not divide by 1024 here, do it in mapper)
        img_normalized = normalize_image(img_array)

        # Extract centerline from segmentation
        cline_map = extract_centerline(label_array)

        # Save as NPZ file
        output_file = os.path.join(output_path, f'{case_id}.npz')
        np.savez_compressed(
            output_file,
            img=img_normalized,
            seg=label_array,
            cline=cline_map,
            spacing=np.array(spacing),
            origin=np.array(origin),
            direction=np.array(direction)
        )

        logger.info(f"Processed {case_id}: img shape={img_array.shape}, "
                   f"seg unique values={np.unique(label_array)}, "
                   f"cline points={np.sum(cline_map > 0)}")
        return True

    except Exception as e:
        logger.error(f"Error processing {case_id}: {str(e)}")
        return False


def process_dataset(
    image_dir='/raid/users/ai_kcm_0/M3DVBAV_CropLung/image',
    label_dir='/raid/users/ai_kcm_0/M3DVBAV_CropLung/label',
    output_dir='./VesselSeg_Data',
    num_workers=1
):
    """
    Process entire vessel segmentation dataset.

    Args:
        image_dir: Directory containing image NIfTI files
        label_dir: Directory containing label NIfTI files
        output_dir: Output directory for NPZ files
        num_workers: Number of parallel workers (not used currently)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

    logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")

    # Match images with labels
    # Assuming same filename for image and label
    successful = 0
    failed = 0

    for img_file in tqdm(image_files, desc="Processing dataset"):
        case_id = img_file.replace('.nii.gz', '')

        img_path = os.path.join(image_dir, img_file)

        # Try different label naming conventions
        label_candidates = [
            img_file,  # Same name
            img_file.replace('_0000', ''),  # nnUNet naming convention
            case_id + '_seg.nii.gz',  # _seg suffix
            case_id + '_label.nii.gz',  # _label suffix
        ]

        label_path = None
        for candidate in label_candidates:
            candidate_path = os.path.join(label_dir, candidate)
            if os.path.exists(candidate_path):
                label_path = candidate_path
                break

        if label_path is None:
            logger.warning(f"No label found for {img_file}, skipping...")
            failed += 1
            continue

        success = process_single_case(img_path, label_path, output_dir, case_id)
        if success:
            successful += 1
        else:
            failed += 1

    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    return successful, failed


def verify_dataset(data_dir='./VesselSeg_Data'):
    """
    Verify the processed dataset.

    Args:
        data_dir: Directory containing NPZ files
    """
    npz_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    logger.info(f"Found {len(npz_files)} NPZ files")

    for npz_file in npz_files[:5]:  # Check first 5 files
        data = np.load(os.path.join(data_dir, npz_file), allow_pickle=True)
        logger.info(f"\n{npz_file}:")
        logger.info(f"  Image shape: {data['img'].shape}")
        logger.info(f"  Image dtype: {data['img'].dtype}")
        logger.info(f"  Image range: [{data['img'].min():.2f}, {data['img'].max():.2f}]")
        logger.info(f"  Seg shape: {data['seg'].shape}")
        logger.info(f"  Seg unique: {np.unique(data['seg'])}")
        logger.info(f"  Cline points: {np.sum(data['cline'] > 0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Vessel Segmentation Dataset')
    parser.add_argument('--image_dir', type=str,
                       default='/raid/users/ai_kcm_0/M3DVBAV_CropLung/image',
                       help='Directory containing image NIfTI files')
    parser.add_argument('--label_dir', type=str,
                       default='/raid/users/ai_kcm_0/M3DVBAV_CropLung/label',
                       help='Directory containing label NIfTI files')
    parser.add_argument('--output_dir', type=str,
                       default='./VesselSeg_Data',
                       help='Output directory for NPZ files')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing dataset')

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
    else:
        process_dataset(
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir
        )
        # Verify after processing
        verify_dataset(args.output_dir)
