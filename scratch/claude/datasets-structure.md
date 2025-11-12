# Dataset Summary

## DRIVE (Retinal Vessel Segmentation)

**Location**: `/dtu/datasets1/02516/DRIVE`

**Structure**:
- Training: 20 images with vessel masks and FOV masks
- Test: 20 images with FOV masks only (no ground truth)

**Split**: 12 train / 4 val / 4 test (from 20 labeled images, seed=67)

**Native size**: 565×584 → Resized to 512×512

## PH2 (Skin Lesion Segmentation)

**Location**: `/dtu/datasets1/02516/PH2_Dataset_images`

**Structure**: 200 samples (IMD002-IMD437), each with image and lesion mask

**Split**: 120 train / 40 val / 40 test (seed=67)

**Native size**: ~765×573 (variable) → Resized to 512×512

## Preprocessing

- All images resized to 512×512 (U-Net compatibility)
- Per-channel standardization using training set statistics only
- Images: bilinear interpolation
- Masks: nearest neighbor interpolation