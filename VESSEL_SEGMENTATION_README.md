# Vessel Segmentation using DeformCL

This implementation provides vessel segmentation from CT/MRI images using the **DeformCL** method (Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image - CVPR 2025).

## Overview

DeformCL is a novel approach to vessel segmentation that:
- Learns deformable centerline representations instead of traditional mask-based segmentation
- Provides natural connectivity and noise robustness
- Is clinically useful for curved planar reformation
- Combines centerline deformation with segmentation for improved accuracy

## Dataset Structure

Your dataset should be organized as follows:
```
/raid/users/ai_kcm_0/M3DVBAV_CropLung/
├── image/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── label/
    ├── case_001.nii.gz
    ├── case_002.nii.gz
    └── ...
```

## Quick Start

### Step 1: Process the Dataset

Convert your NIfTI files to the required NPZ format with centerline extraction:

```bash
python process_vessel_dataset.py \
    --image_dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/image \
    --label_dir /raid/users/ai_kcm_0/M3DVBAV_CropLung/label \
    --output_dir ./VesselSeg_Data
```

This will:
- Load all NIfTI images and labels
- Extract vessel centerlines via 3D skeletonization
- Save processed data as NPZ files

### Step 2: Train the Model

**Option A: DeformCL (Recommended)**
```bash
# Single GPU
python train_vessel_seg.py --num-gpus 1 \
    --config-file configs/vessel_deformcl.yaml \
    OUTPUT_DIR ./outputs/vessel_deformcl

# Multi-GPU (4 GPUs)
python train_vessel_seg.py --num-gpus 4 --dist-url auto \
    --config-file configs/vessel_deformcl.yaml \
    OUTPUT_DIR ./outputs/vessel_deformcl

# Or use the shell script
./run_vessel_deformcl.sh 1  # Single GPU
./run_vessel_deformcl.sh 4  # 4 GPUs
```

**Option B: UNet Baseline**
```bash
python train_vessel_seg.py --num-gpus 1 \
    --config-file configs/vessel_unet.yaml \
    OUTPUT_DIR ./outputs/vessel_unet

# Or use the shell script
./run_vessel_unet.sh 1
```

### Step 3: Evaluate the Model

```bash
python train_vessel_seg.py --num-gpus 1 --eval-only \
    --config-file configs/vessel_deformcl.yaml \
    MODEL.WEIGHTS ./outputs/vessel_deformcl/model_final.pth \
    OUTPUT_DIR ./outputs/vessel_deformcl_eval
```

### Step 4: Run Inference on New Images

```bash
# Single image
python inference_vessel_seg.py \
    --config-file configs/vessel_deformcl.yaml \
    --weights outputs/vessel_deformcl/model_final.pth \
    --input /path/to/new_image.nii.gz \
    --output /path/to/output_dir

# Batch inference
python inference_vessel_seg.py \
    --config-file configs/vessel_deformcl.yaml \
    --weights outputs/vessel_deformcl/model_final.pth \
    --input-dir /path/to/images \
    --output /path/to/output_dir \
    --gpu 0
```

## Configuration

### Key Parameters in `configs/vessel_deformcl.yaml`

```yaml
MODEL:
  PRED_CLASS: 1          # Class ID for vessels (adjust based on your labels)

INPUT:
  CROP_SIZE_TRAIN: (128, 128, 128)  # Training patch size

SOLVER:
  IMS_PER_BATCH: 2       # Batch size (reduce if OOM)
  BASE_LR: 1e-3          # Learning rate
  MAX_ITER: 10000        # Total training iterations

DATASETS:
  NUM_FOLDS: 5           # Cross-validation folds
  TEST_FOLDS: (0,)       # Test fold ID
```

### Adjusting for Your Dataset

1. **If your labels have different class IDs**: Modify `MODEL.PRED_CLASS`
2. **If you have limited GPU memory**: Reduce `SOLVER.IMS_PER_BATCH` or `INPUT.CROP_SIZE_TRAIN`
3. **If training is unstable**: Reduce `SOLVER.BASE_LR` to 5e-4 or 1e-4
4. **For larger datasets**: Increase `SOLVER.MAX_ITER`

## File Structure

```
Deform_CL/
├── configs/
│   ├── vessel_deformcl.yaml    # DeformCL config for vessel seg
│   └── vessel_unet.yaml        # UNet baseline config
├── process_vessel_dataset.py   # Dataset preprocessing script
├── train_vessel_seg.py         # Training script
├── inference_vessel_seg.py     # Inference script
├── run_vessel_deformcl.sh      # Training shell script (DeformCL)
├── run_vessel_unet.sh          # Training shell script (UNet)
└── vesselseg/
    ├── data/
    │   ├── datasets.py         # Dataset registration
    │   └── dataset_mapper.py   # Data loading and augmentation
    ├── modeling/
    │   └── ...                 # Model architecture
    └── evaluation/
        └── seg_evaluation.py   # Dice evaluation metrics
```

## Expected Results

After training, you should see:
- **Dice Score**: Measure of segmentation accuracy (higher is better)
- **Centerline Accuracy**: For DeformCL, measures quality of extracted centerlines

Typical results on vessel segmentation tasks:
- UNet baseline: ~0.70-0.80 Dice
- DeformCL: ~0.75-0.85 Dice (with better connectivity)

## Troubleshooting

### Out of Memory (OOM) Error
```yaml
# Reduce batch size
SOLVER:
  IMS_PER_BATCH: 1

# Or reduce crop size
INPUT:
  CROP_SIZE_TRAIN: (96, 96, 96)
```

### No Segmentation Mask Found
- Check your label files contain non-zero vessel annotations
- Verify `MODEL.PRED_CLASS` matches your label values

### Training Diverges
```yaml
# Reduce learning rate
SOLVER:
  BASE_LR: 5e-4

# Enable gradient clipping (already enabled by default)
SOLVER:
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_VALUE: 0.01
```

### Missing Dependencies
```bash
pip install SimpleITK scikit-image tqdm
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{deformcl2025cvpr,
  title={Learning Deformable Centerline Representation for Vessel Extraction in 3D Medical Image},
  author={...},
  booktitle={CVPR},
  year={2025}
}
```

## License

This code is released under the same license as the original DeformCL repository.
