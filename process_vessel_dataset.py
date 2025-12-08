"""
Process Vessel Segmentation Dataset from NIfTI files.
Converts NIfTI images and labels to NPZ format with centerline extraction.

Dataset structure:
    - images: /raid/users/ai_kcm_0/M3DVBAV_CropLung/image/*.nii.gz
    - labels: /raid/users/ai_kcm_0/M3DVBAV_CropLung/label/*.nii.gz

Usage:
    # Use all CPU cores (recommended)
    python process_vessel_dataset.py --num_workers -1

    # Use 8 CPU cores
    python process_vessel_dataset.py --num_workers 8

    # Skip centerline extraction (much faster, for testing)
    python process_vessel_dataset.py --skip_centerline
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import morphology
from tqdm import tqdm
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time

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


def extract_centerline_fast(seg_mask):
    """
    Fast centerline extraction using morphological thinning.
    Less accurate but much faster than full skeletonization.

    Args:
        seg_mask: Binary segmentation mask (numpy array)

    Returns:
        Centerline mask with same labels as input segmentation
    """
    from scipy import ndimage

    # Create binary mask
    binary_mask = (seg_mask > 0).astype(np.uint8)

    # Use distance transform to find centerline approximation
    dist = ndimage.distance_transform_edt(binary_mask)

    # Find local maxima along the distance transform (approximate centerline)
    from skimage.feature import peak_local_max
    coordinates = peak_local_max(dist, min_distance=1, threshold_abs=1)

    # Create centerline map
    cline_map = np.zeros_like(seg_mask)
    if len(coordinates) > 0:
        cline_map[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
        # Preserve original labels
        cline_map = cline_map * seg_mask

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


def process_single_case_wrapper(args):
    """Wrapper for multiprocessing."""
    return process_single_case(*args)


def process_single_case(img_path, label_path, output_path, case_id,
                       skip_centerline=False, use_fast_centerline=False):
    """
    Process a single case and save as NPZ file.

    Args:
        img_path: Path to image NIfTI file
        label_path: Path to label NIfTI file
        output_path: Output directory
        case_id: Case identifier
        skip_centerline: Skip centerline extraction (faster)
        use_fast_centerline: Use fast approximation instead of full skeletonization

    Returns:
        Tuple of (case_id, success, message)
    """
    try:
        start_time = time.time()

        # Load image and label
        img_array, spacing, origin, direction = load_nifti(img_path)
        label_array, _, _, _ = load_nifti(label_path)

        # Ensure label is integer type
        label_array = label_array.astype(np.uint8)

        # Normalize image (do not divide by 1024 here, do it in mapper)
        img_normalized = normalize_image(img_array)

        # Extract centerline from segmentation
        if skip_centerline:
            # Just copy segmentation as centerline placeholder
            cline_map = label_array.copy()
        elif use_fast_centerline:
            cline_map = extract_centerline_fast(label_array)
        else:
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

        elapsed = time.time() - start_time
        msg = (f"Processed {case_id}: shape={img_array.shape}, "
               f"seg_vals={np.unique(label_array).tolist()}, "
               f"cline_pts={np.sum(cline_map > 0)}, time={elapsed:.1f}s")
        return (case_id, True, msg)

    except Exception as e:
        return (case_id, False, f"Error processing {case_id}: {str(e)}")


def find_label_file(img_file, image_dir, label_dir):
    """Find corresponding label file for an image."""
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

    return img_path, label_path, case_id


def process_dataset(
    image_dir='/raid/users/ai_kcm_0/M3DVBAV_CropLung/image',
    label_dir='/raid/users/ai_kcm_0/M3DVBAV_CropLung/label',
    output_dir='./VesselSeg_Data',
    num_workers=1,
    skip_centerline=False,
    use_fast_centerline=False
):
    """
    Process entire vessel segmentation dataset with multiprocessing support.

    Args:
        image_dir: Directory containing image NIfTI files
        label_dir: Directory containing label NIfTI files
        output_dir: Output directory for NPZ files
        num_workers: Number of parallel workers (-1 for all cores)
        skip_centerline: Skip centerline extraction
        use_fast_centerline: Use fast centerline approximation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of image files
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

    logger.info(f"Found {len(image_files)} images and {len(label_files)} labels")

    # Prepare task list
    tasks = []
    skipped = 0
    for img_file in image_files:
        img_path, label_path, case_id = find_label_file(img_file, image_dir, label_dir)

        if label_path is None:
            logger.warning(f"No label found for {img_file}, skipping...")
            skipped += 1
            continue

        tasks.append((img_path, label_path, output_dir, case_id,
                     skip_centerline, use_fast_centerline))

    logger.info(f"Prepared {len(tasks)} tasks ({skipped} skipped due to missing labels)")

    # Determine number of workers
    if num_workers == -1:
        num_workers = cpu_count()
    elif num_workers == 0 or num_workers == 1:
        num_workers = 1

    logger.info(f"Using {num_workers} worker(s)")

    # Process with multiprocessing or single process
    successful = 0
    failed = 0
    start_time = time.time()

    if num_workers > 1:
        # Multiprocessing
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_case_wrapper, tasks),
                total=len(tasks),
                desc="Processing dataset"
            ))

        for case_id, success, msg in results:
            if success:
                successful += 1
                logger.info(msg)
            else:
                failed += 1
                logger.error(msg)
    else:
        # Single process
        for task in tqdm(tasks, desc="Processing dataset"):
            case_id, success, msg = process_single_case(*task)
            if success:
                successful += 1
                logger.info(msg)
            else:
                failed += 1
                logger.error(msg)

    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Skipped: {skipped}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Avg time per case: {total_time/max(successful,1):.1f}s")
    logger.info(f"{'='*60}")

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
    parser.add_argument('--num_workers', type=int, default=-1,
                       help='Number of parallel workers (-1 for all CPU cores, default: -1)')
    parser.add_argument('--skip_centerline', action='store_true',
                       help='Skip centerline extraction (much faster, use for testing)')
    parser.add_argument('--fast_centerline', action='store_true',
                       help='Use fast centerline approximation instead of full skeletonization')
    parser.add_argument('--verify', action='store_true',
                       help='Only verify existing dataset')

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
    else:
        logger.info(f"\nSettings:")
        logger.info(f"  Image dir: {args.image_dir}")
        logger.info(f"  Label dir: {args.label_dir}")
        logger.info(f"  Output dir: {args.output_dir}")
        logger.info(f"  Workers: {args.num_workers} (-1 = all cores = {cpu_count()})")
        logger.info(f"  Skip centerline: {args.skip_centerline}")
        logger.info(f"  Fast centerline: {args.fast_centerline}")
        logger.info("")

        process_dataset(
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            num_workers=args.num_workers,
            skip_centerline=args.skip_centerline,
            use_fast_centerline=args.fast_centerline
        )
        # Verify after processing
        verify_dataset(args.output_dir)
