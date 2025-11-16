# Medical Image Segmentation Project - Full Context Summary

## Project Overview

This is a medical image segmentation project comparing two neural network architectures (U-Net vs Encoder-Decoder) across two datasets (PH2 and DRIVE) with multiple loss functions. The goal is to segment lesions/vessels in medical images and analyze which architecture and loss function performs best.

---

## Project Structure

```
Image_segmentation/
├── src/
│   ├── datasets/
│   │   ├── PH2.py              # Skin lesion dataset (200 images)
│   │   └── DRIVE.py            # Retinal vessel dataset (40 images)
│   ├── models/
│   │   ├── u_net.py            # U-Net architecture (with skip connections)
│   │   ├── enc_dec.py          # Encoder-Decoder (no skip connections)
│   │   └── losses.py           # BCE, Dice, Focal, Weighted BCE losses
│   ├── metrics/
│   │   └── segmentation.py     # Dice, IoU, Accuracy, Sensitivity, Specificity
│   ├── training/
│   │   ├── train.py            # Training loop and train_model()
│   │   ├── evaluate.py         # Evaluation function
│   │   ├── early_stopping.py   # Early stopping callback
│   │   └── helpers.py          # Checkpoint saving, CSV export
│   └── utils/
│       ├── logger.py           # Colored console logging
│       └── seed.py             # Reproducibility (seed=67)
├── scripts/
│   ├── common.py               # Shared functions for all scripts
│   ├── train_u_net.py          # U-Net training script
│   ├── train_enc_dec.py        # Encoder-Decoder training script
│   ├── test_u_net.py           # U-Net testing script
│   ├── test_enc_dec.py         # Encoder-Decoder testing script
│   └── test_all_models.py      # Batch test all trained models
├── scratch/
│   └── plotting/
│       └── training_history.py # Plot training curves
├── jobs/
│   ├── train_u_net.sh          # HPC job script for U-Net
│   ├── train_enc_dec.sh        # HPC job script for Encoder-Decoder
│   └── submit_all_part_1.sh    # Batch submit all 16 experiments
├── checkpoints/                # Model checkpoints (auto-created)
├── outputs/                    # HPC job logs (auto-created)
└── pyproject.toml             # uv project config
```

---

## Datasets

### 1. PH2 Dataset (Skin Lesions)
- **Images**: 200 dermoscopic images of skin lesions
- **Task**: Segment lesion from skin
- **Image size**: Variable (resized to 256x256)
- **Splits**: 70% train, 15% val, 15% test
- **No FOV mask needed**

### 2. DRIVE Dataset (Retinal Vessels)
- **Images**: 40 retinal fundus images
- **Task**: Segment blood vessels
- **Image size**: 565x584 (resized to 256x256)
- **Splits**: 70% train, 15% val, 15% test
- **FOV mask**: Used during evaluation (masks out non-retina regions)

**Data splits use seed=67 for reproducibility across all experiments.**

---

## Model Architectures

### 1. U-Net
- **File**: `src/models/u_net.py`
- **Architecture**: 
  - Encoder: 4 down-sampling blocks (conv → conv → max pool)
  - Bottleneck: 2 conv layers
  - Decoder: 4 up-sampling blocks with **skip connections**
  - Output: 1 channel (binary segmentation logits)
- **Init features**: 64 (grows to 512 in bottleneck)
- **Key feature**: Skip connections preserve spatial information

### 2. Encoder-Decoder (EncDec)
- **File**: `src/models/enc_dec.py`
- **Architecture**:
  - Encoder: 4 down-sampling blocks (same as U-Net)
  - Bottleneck: 2 conv layers
  - Decoder: 4 up-sampling blocks **without skip connections**
  - Output: 1 channel (binary segmentation logits)
- **Init features**: 64
- **Key difference**: No skip connections (simpler architecture)

---

## Loss Functions

All implemented in `src/models/losses.py`:

### 1. Binary Cross-Entropy (BCE)
```python
BCE = -[y * log(sigmoid(ŷ)) + (1-y) * log(1-sigmoid(ŷ))]
```
- Standard binary classification loss
- Works directly on logits (uses BCEWithLogitsLoss)

### 2. Dice Loss
```python
Dice = 1 - (2 * |X ∩ Y|) / (|X| + |Y|)
```
- Directly optimizes Dice coefficient
- Good for imbalanced datasets
- Smooth parameter: 1e-6

### 3. Focal Loss
```python
FL = -α(1-pt)^γ * BCE
```
- Focuses on hard examples
- Parameters: α=0.25, γ=2.0
- Good for extreme class imbalance

### 4. Weighted BCE
```python
WeightedBCE = BCE with pos_weight
```
- Automatically computes `pos_weight = neg_pixels / pos_pixels` from training data
- Balances classes by weighting positive examples

---

## Metrics

All computed in `src/metrics/segmentation.py`:

1. **Dice Coefficient**: `2 * |pred ∩ target| / (|pred| + |target|)`
2. **IoU (Jaccard)**: `|pred ∩ target| / |pred ∪ target|`
3. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
4. **Sensitivity (Recall)**: `TP / (TP + FN)`
5. **Specificity**: `TN / (TN + FP)`

**All metrics use threshold=0.5 to convert probabilities to binary predictions.**

---

## Training Pipeline

### Training Loop (`src/training/train.py`)

**Function**: `train_model()`

**Process**:
1. Loop through epochs
2. **Training phase**:
   - `train_one_epoch()`: Forward pass, compute loss, backward pass, update weights
   - Track training loss
3. **Validation phase**:
   - `evaluate()`: Compute all 5 metrics on validation set
   - Track val loss, dice, iou, accuracy, sensitivity, specificity
4. **Checkpointing**:
   - Save `best_model.pth` when validation Dice improves
   - Save `latest_model.pth` every epoch
5. **Early stopping**:
   - Monitor validation Dice
   - Stop if no improvement for `patience` epochs (default: 10)
   - Mode: maximize Dice score
6. **Return**: `TrainingHistory` dataclass with all metrics per epoch

**Output files per experiment**:
- `checkpoints/{DATASET}_{MODEL}_{LOSS}_{TIMESTAMP}/`
  - `best_model.pth` - Best model by validation Dice
  - `latest_model.pth` - Model from last epoch
  - `training_history.csv` - All metrics per epoch

---

## Dataclasses

### TrainingHistory (`src/training/train.py`)
```python
@dataclass
class TrainingHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_dice: List[float] = field(default_factory=list)
    val_iou: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    val_sensitivity: List[float] = field(default_factory=list)
    val_specificity: List[float] = field(default_factory=list)
```

### PerformanceMetrics (`src/training/evaluate.py`)
```python
@dataclass
class PerformanceMetrics:
    dice: float = 0.0
    iou: float = 0.0
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    loss: float = 0.0
```

---

## Training Scripts

### Individual Training Scripts

**U-Net**: `scripts/train_u_net.py`
```bash
uv run -m scripts.train_u_net \
    --dataset PH2 \
    --loss dice \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --patience 10
```

**Encoder-Decoder**: `scripts/train_enc_dec.py`
```bash
uv run -m scripts.train_enc_dec \
    --dataset DRIVE \
    --loss focal \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --patience 10
```

### Common Functions (`scripts/common.py`)

All training scripts share these helper functions:
- `create_dataloaders(dataset_name, batch_size, num_workers)` - Creates train/val/test loaders
- `create_model(model_name, device)` - Instantiates U-Net or EncDec
- `create_loss(loss_name, train_loader, device)` - Creates loss function
- `setup_checkpoint_dir(base_dir, dataset, model, loss)` - Creates checkpoint directory
- `log_training_config(args)` - Logs training configuration

---

## HPC Execution

### Job Scripts

**U-Net Job**: `jobs/train_u_net.sh`
```bash
#!/bin/sh
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_unet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=20GB]"
#BSUB -W 12:00
#BSUB -o outputs/output_unet_%J.out
#BSUB -e outputs/output_unet_%J.err

module load python3/3.13.9
cd ~/projects/Image_segmentation
uv sync

uv run -m scripts.train_u_net \
    --dataset "${DATASET:-PH2}" \
    --loss "${LOSS:-dice}" \
    --epochs ${EPOCHS:-50} \
    --batch_size ${BATCH_SIZE:-4} \
    --lr ${LR:-1e-4} \
    --patience ${PATIENCE:-10} \
    --num_workers 4 \
    --device cuda \
    --checkpoint_base checkpoints
```

**Submission**:
```bash
# Submit single job
DATASET=PH2 LOSS=dice bash jobs/submit_all_part_1.sh

# Submit all 16 experiments (2 models × 2 datasets × 4 losses)
bash jobs/submit_all_part_1.sh
```

### All 16 Experiments

The batch submission script `jobs/submit_all_part_1.sh` submits:

**U-Net (8 experiments)**:
1. PH2_unet_bce
2. PH2_unet_dice
3. PH2_unet_focal
4. PH2_unet_weighted_bce
5. DRIVE_unet_bce
6. DRIVE_unet_dice
7. DRIVE_unet_focal
8. DRIVE_unet_weighted_bce

**Encoder-Decoder (8 experiments)**:
9. PH2_encdec_bce
10. PH2_encdec_dice
11. PH2_encdec_focal
12. PH2_encdec_weighted_bce
13. DRIVE_encdec_bce
14. DRIVE_encdec_dice
15. DRIVE_encdec_focal
16. DRIVE_encdec_weighted_bce

**HPC Commands**:
```bash
bstat                           # Check job status
bjobs                           # List all jobs
bpeek <job_id>                  # View live output
tail -f outputs/output_unet_*.out  # View job logs
```

---

## Testing Scripts

### Individual Model Testing

**Test U-Net**: `scripts/test_u_net.py`
```bash
uv run -m scripts.test_u_net \
    --checkpoint checkpoints/PH2_unet_dice_20251112_183723/best_model.pth \
    --dataset PH2 \
    --loss dice \
    --batch_size 4 \
    --gpu_id 0
```

**Output**: Evaluates on **train, validation, and test** sets, prints comprehensive metrics table.

**Test Encoder-Decoder**: `scripts/test_enc_dec.py`
- Same interface as `test_u_net.py`
- Also evaluates on all 3 splits

### Batch Testing All Models

**Script**: `scripts/test_all_models.py`

```bash
uv run -m scripts.test_all_models \
    --checkpoint_dir checkpoints \
    --output_csv test_results.csv \
    --batch_size 4 \
    --gpu_id 0
```

**What it does**:
1. Finds all `checkpoints/*/best_model.pth` files
2. Parses directory names to extract dataset, model, loss
3. Tests each model on train/val/test splits
4. Saves results to CSV with columns:
   - `dataset`, `model`, `loss`, `epoch`, `checkpoint`
   - `train_dice`, `train_iou`, `train_accuracy`, `train_sensitivity`, `train_specificity`, `train_loss`
   - `val_dice`, `val_iou`, `val_accuracy`, `val_sensitivity`, `val_specificity`, `val_loss`
   - `test_dice`, `test_iou`, `test_accuracy`, `test_sensitivity`, `test_specificity`, `test_loss`
5. Prints summary table sorted by test Dice score

**Output**: `test_results.csv` with all 16 experiments' results

---

## Visualization

### Training History Plots

**Script**: `scratch/plotting/training_history.py`

```bash
uv run -m scratch.plotting.training_history \
    --unet checkpoints/PH2_unet_dice_20251112_183723 \
    --encdec checkpoints/PH2_encdec_dice_20251112_175528 \
    --dataset PH2 \
    --output plots/PH2_comparison.png
```

**Generates**:
- **Left plot**: Training loss vs Validation loss (U-Net and EncDec)
- **Right plot**: Validation Dice score (U-Net and EncDec)
- **Style**: Seaborn muted palette, whitegrid
- **Purpose**: Visualize overfitting (train/val divergence)

---

## Key Design Decisions

### 1. Reproducibility
- **Fixed seed**: 67 everywhere (data splits, model initialization, training)
- **Deterministic PyTorch**: `torch.backends.cudnn.deterministic = True`

### 2. Dataclasses Over Dicts
- Use `@dataclass` for `TrainingHistory` and `PerformanceMetrics`
- Leverage `dataclasses.asdict()` for automatic CSV conversion
- Type-safe, auto-complete friendly

### 3. Checkpoint Naming Convention
```
checkpoints/{DATASET}_{MODEL}_{LOSS}_{TIMESTAMP}/
```
- Example: `PH2_unet_dice_20251112_183723/`
- Self-documenting: immediately see what experiment it is

### 4. Early Stopping
- **Metric**: Validation Dice coefficient
- **Mode**: Maximize
- **Patience**: 10 epochs (configurable)
- **Delta**: 0.001 minimum improvement

### 5. FOV Masking
- DRIVE dataset uses Field-of-View (FOV) mask
- Masks out non-retina regions during evaluation
- PH2 does not use FOV masking

### 6. Loss Function Selection
- **Weighted BCE**: Automatically computes `pos_weight` from training data
- Useful for highly imbalanced datasets

---

## Current Status

### Completed ✅
1. ✅ All datasets implemented (PH2, DRIVE)
2. ✅ Both models implemented (U-Net, EncDec)
3. ✅ All 4 loss functions implemented (BCE, Dice, Focal, Weighted BCE)
4. ✅ All 5 metrics implemented (Dice, IoU, Acc, Sens, Spec)
5. ✅ Training pipeline with early stopping
6. ✅ Checkpoint saving and CSV export
7. ✅ HPC job scripts for batch submission
8. ✅ Testing scripts for evaluation
9. ✅ Training history visualization
10. ✅ 16 experiments submitted to HPC

### In Progress 🔄
- ⏳ Waiting for all 16 HPC jobs to complete
- ⏳ Testing all trained models on test sets

### Next Steps 🎯
1. **Run batch testing** once jobs complete:
   ```bash
   uv run -m scripts.test_all_models --output_csv results/all_test_results.csv
   ```

2. **Generate comparison plots** for each dataset:
   ```bash
   uv run -m scratch.plotting.training_history \
       --unet checkpoints/PH2_unet_dice_* \
       --encdec checkpoints/PH2_encdec_dice_* \
       --dataset PH2 \
       --output plots/PH2_comparison.png
   ```

3. **Analyze results**:
   - Which architecture performs better? (U-Net vs EncDec)
   - Which loss function is best? (BCE vs Dice vs Focal vs Weighted BCE)
   - Do skip connections help? (U-Net vs EncDec comparison)
   - Is there overfitting? (train vs val curves)
   - Dataset differences? (PH2 vs DRIVE performance)

4. **Create final report** with:
   - Results table (all 16 experiments)
   - Learning curves (2 plots: PH2 and DRIVE)
   - Best model analysis
   - Conclusions and recommendations

---

## Important Notes

### Environment Variables in Job Submission
- **Issue**: Environment variables must be quoted in job scripts
- **Fix**: Use `"${DATASET:-PH2}"` not `${DATASET:-PH2}` in job scripts
- **Reason**: `bsub` doesn't pass environment variables by default

### GPU Usage
- All jobs request 1 GPU with exclusive process mode
- Training uses `--device cuda` by default
- Can specify GPU ID with `--gpu_id` for testing

### Memory and Time Limits
- **Memory**: 20GB per job
- **Time**: 12 hours per job
- **CPUs**: 4 cores per job
- Adjust in job scripts if needed

### Common Commands
```bash
# Training
uv run -m scripts.train_u_net --dataset PH2 --loss dice

# Testing one model
uv run -m scripts.test_u_net --checkpoint checkpoints/.../best_model.pth --dataset PH2 --loss dice

# Testing all models
uv run -m scripts.test_all_models --output_csv results.csv

# Plotting
uv run -m scratch.plotting.training_history --unet ... --encdec ... --dataset PH2

# HPC job management
bstat                    # Check status
bjobs                    # List jobs
bpeek <job_id>          # Live output
bkill <job_id>          # Kill job
```

---

## Research Questions

This project aims to answer:

1. **Architecture Comparison**: Does U-Net (with skip connections) outperform simple Encoder-Decoder?
2. **Loss Function Analysis**: Which loss function works best for medical image segmentation?
3. **Dataset Differences**: Are results consistent across different medical imaging modalities (skin vs retinal)?
4. **Generalization**: How much do models overfit? (train vs validation performance)
5. **Class Imbalance**: Does weighted BCE or focal loss help with imbalanced datasets?

---

## File Locations Summary

**Core Implementation**:
- Models: `src/models/u_net.py`, `src/models/enc_dec.py`
- Losses: `src/models/losses.py`
- Metrics: `src/metrics/segmentation.py`
- Training: `src/training/train.py`, `src/training/evaluate.py`
- Datasets: `src/datasets/PH2.py`, `src/datasets/DRIVE.py`

**Executable Scripts**:
- Training: `scripts/train_u_net.py`, `scripts/train_enc_dec.py`
- Testing: `scripts/test_u_net.py`, `scripts/test_enc_dec.py`, `scripts/test_all_models.py`
- Plotting: `scratch/plotting/training_history.py`

**HPC Scripts**:
- Jobs: `jobs/train_u_net.sh`, `jobs/train_enc_dec.sh`
- Submission: `jobs/submit_all_part_1.sh`

**Outputs**:
- Checkpoints: `checkpoints/{DATASET}_{MODEL}_{LOSS}_{TIMESTAMP}/`
- Job logs: `outputs/output_{model}_{JOB_ID}.out`
- Results: `test_results.csv` (generated by `test_all_models.py`)
- Plots: `plots/*.png` (generated by plotting scripts)

---

## End of Summary

This document provides complete context for resuming work on this medical image segmentation project. All code is functional and ready for analysis once the 16 HPC training jobs complete.