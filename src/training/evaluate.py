import torch
from src.metrics.segmentation import SegmentationMetrics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(model, dataloader, criterion, device, use_fov_mask=False):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_metrics = {
        "dice": 0,
        "iou": 0,
        "accuracy": 0,
        "sensitivity": 0,
        "specificity": 0,
    }

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
                # Mask out predictions and targets outside FOV
                probs = probs * fov_masks
                masks = masks * fov_masks

            # Compute all metrics
            batch_metrics = SegmentationMetrics.compute_all_metrics(probs, masks)
            for key in all_metrics:
                all_metrics[key] += batch_metrics[key]

    # Average across all batches
    avg_loss = total_loss / len(dataloader)
    for key in all_metrics:
        all_metrics[key] /= len(dataloader)

    return avg_loss, all_metrics
