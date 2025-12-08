#!/bin/bash
# Training script for Vessel Segmentation using UNet baseline
#
# Usage:
#   ./run_vessel_unet.sh [NUM_GPUS] [OUTPUT_DIR]
#
# Examples:
#   ./run_vessel_unet.sh 1                    # Single GPU training
#   ./run_vessel_unet.sh 4                    # 4 GPU training
#   ./run_vessel_unet.sh 1 ./outputs/exp1    # Custom output directory

NUM_GPUS=${1:-1}
OUTPUT_DIR=${2:-"./outputs/vessel_unet"}

echo "======================================"
echo "Vessel Segmentation Training (UNet)"
echo "======================================"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "======================================"

# Make sure the dataset is processed
if [ ! -d "./VesselSeg_Data" ]; then
    echo "WARNING: VesselSeg_Data directory not found!"
    echo "Please run: python process_vessel_dataset.py first"
    echo "======================================"
fi

# Run training
python train_vessel_seg.py \
    --num-gpus ${NUM_GPUS} \
    --dist-url auto \
    --config-file configs/vessel_unet.yaml \
    OUTPUT_DIR "${OUTPUT_DIR}"
