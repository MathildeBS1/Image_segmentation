#!/bin/bash

# Make sure we're in the right directory
cd ~/project/Image_segmentation

module load python3/3.13.9

uv sync

echo ""
echo "=== Encoder-Decoder ==="

echo ""
echo "Testing: DRIVE U-Net + DRIVE"
uv run -m scripts.test_enc_dec \
    --checkpoint checkpoints/DRIVE_encdec_dice_20251116_104807/best_model.pth \
    --dataset DRIVE \
    --loss dice \
    --batch_size 2 \
    --num_workers 4 \
    --device cuda