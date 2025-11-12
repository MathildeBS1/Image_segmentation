import torch
from src.metrics.segmentation import SegmentationMetrics
from src.utils.logger import get_logger
from dataclasses import dataclass

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Segmentation performance metrics."""

    dice: float
    iou: float
    accuracy: float
    sensitivity: float
    specificity: float
    loss: float


def evaluate(model, dataloader, criterion, device, use_fov_mask=False):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0

    with torch.no_grad():
        for batch in dataloader:
            # Handle both PH2 and DRIVE datasets
            if len(batch) == 3:  # PH2: (image, mask, case_id)
                images, masks, _ = batch
                fov_masks = None
            else:  # DRIVE: (image, vessel_mask, fov_mask, case_id)
                images, masks, fov_masks, _ = batch

            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Compute metrics
            probs = torch.sigmoid(logits)

            # Apply FOV mask if requested (DRIVE only)
            if use_fov_mask and fov_masks is not None:
                fov_masks = fov_masks.to(device)
                probs = probs * fov_masks
                masks = masks * fov_masks

            # Accumulate metrics
            total_dice += SegmentationMetrics.dice_coefficient(probs, masks)
            total_iou += SegmentationMetrics.iou(probs, masks)
            total_accuracy += SegmentationMetrics.accuracy(probs, masks)
            total_sensitivity += SegmentationMetrics.sensitivity(probs, masks)
            total_specificity += SegmentationMetrics.specificity(probs, masks)

    # Average across all batches
    num_batches = len(dataloader)

    return PerformanceMetrics(
        dice=total_dice / num_batches,
        iou=total_iou / num_batches,
        accuracy=total_accuracy / num_batches,
        sensitivity=total_sensitivity / num_batches,
        specificity=total_specificity / num_batches,
        loss=total_loss / num_batches,
    )
