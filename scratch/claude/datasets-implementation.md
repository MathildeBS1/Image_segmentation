# Dataloader Implementation Summary

## Overview

Implemented PyTorch dataloaders for two medical image segmentation datasets: DRIVE (retinal vessel segmentation) and PH2 (skin lesion segmentation).

## Dataset Exploration

### DRIVE Dataset
- **Location**: `/dtu/datasets1/02516/DRIVE`
- **Total samples**: 20 labeled images (training folder only)
- **Key finding**: Test set contains no ground truth vessel masks - only FOV masks
- **Native resolution**: 565×584 (consistent across all images)
- **Structure**: Images in `.tif`, vessel masks in `.gif` (1st_manual folder), FOV masks in `.gif` (mask folder)
- **Image type**: Retinal fundus photographs with blood vessel annotations

### PH2 Dataset
- **Location**: `/dtu/datasets1/02516/PH2_Dataset_images`
- **Total samples**: 200 labeled images
- **Native resolution**: Variable (~765×573 with slight variations)
- **Structure**: Nested folders (IMD###/IMD###_Dermoscopic_Image, IMD###_lesion)
- **Image type**: Dermoscopic skin lesion images with lesion boundary annotations
- **Non-consecutive naming**: IMD002-IMD437 with gaps

## Key Design Decisions

### 1. Data Splits
Since DRIVE test set lacks ground truth, we created custom splits from available labeled data:
- **DRIVE**: 12 train / 4 val / 4 test (from 20 images)
- **PH2**: 120 train / 40 val / 40 test (from 200 images)
- **Random seed**: 67 for reproducibility

### 2. Image Resizing
All images resized to **512×512** for:
- U-Net architectural compatibility (clean 2× downsampling: 512→256→128→64→32→16)
- Consistency across both datasets
- Avoiding shape mismatch issues in skip connections
- Standard practice in segmentation literature

**Interpolation methods**:
- Images: `Image.BILINEAR` (smooth, natural appearance)
- Masks: `Image.NEAREST` (preserves binary values, prevents interpolation artifacts)

### 3. Normalization Strategy
Per-channel standardization (mean and std) to handle different imaging characteristics:
- Computed **only from training split** to prevent information leakage
- Applied to all splits (train/val/test) using training statistics
- Stored on dataset instance and shared across subsets created by `random_split`

### 4. Return Values
Each sample returns:
- DRIVE: `(image, vessel_mask, fov_mask, case_id)`
- PH2: `(image, mask, case_id)`
- Case IDs enable per-sample tracking and visualization

## Implementation Details

### Architecture Pattern
- Single dataset class per dataset
- PyTorch's `random_split` for train/val/test division
- Static method `get_dataloaders()` returns all three loaders at once
- Lazy normalization: computed after split, applied via shared dataset reference

### Key Methods
- `__init__`: Discovers and validates all image-mask pairs
- `__getitem__`: Loads, resizes, normalizes, and returns samples
- `set_normalization()`: Stores mean/std for standardization
- `_compute_normalization_stats()`: Calculates statistics from training indices only
- `get_dataloaders()`: Creates splits, computes stats, returns dataloaders

### FOV Masks (DRIVE)
Field-of-view masks identify valid retinal region vs black background. Decision: use for evaluation only (not training) since black border is valid background class.

## Technical Challenges & Solutions

1. **Variable image sizes (PH2)**: Solved by resizing to fixed 512×512
2. **Non-consecutive sample IDs**: Handled by explicit validation during loading
3. **Information leakage risk**: Prevented by computing normalization stats from training split only
4. **Missing test labels (DRIVE)**: Created custom splits from training data only
5. **Binary mask preservation**: Used nearest neighbor interpolation for masks

## Code Quality
- Clean, consistent structure across both datasets
- Minimal comments (self-documenting code)
- One-line docstrings
- No user-configurable parameters for core design decisions (512×512, splits, seed)
- Includes `__main__` block for quick validation

## Validation
Both dataloaders tested and confirmed:
- Correct batch shapes: `[batch_size, channels, 512, 512]`
- Proper normalization statistics computed
- Case IDs correctly extracted and returned
- No dimension mismatch errors