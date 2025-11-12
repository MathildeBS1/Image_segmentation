import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss for binary segmentation.

    Formula:
        BCE = -[y * log(sigmoid(ŷ)) + (1-y) * log(1-sigmoid(ŷ))]

    where:
        ŷ = logits (model output)
        sigmoid = sigmoid function
        y = ground truth (0 or 1)
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation. Differentiable approximation of Dice coefficient.

    Formula:
        DiceLoss = 1 - (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)

    where:
        X = predicted segmentation (after sigmoid)
        Y = ground truth segmentation
        ε = smoothing constant to avoid division by zero
        |X ∩ Y| = intersection (element-wise product sum)
        |X|, |Y| = cardinality (sum of elements)
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Apply sigmoid to convert logits to probabilities
        y_pred_prob = torch.sigmoid(y_pred)

        # Flatten the tensors for computing intersection and sums
        y_pred_flat = y_pred_prob.view(-1)
        y_true_flat = y_true.view(-1)

        # Compute Dice coefficient components
        intersection = (y_pred_flat * y_true_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            y_pred_flat.sum() + y_true_flat.sum() + self.smooth
        )

        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice_coeff


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance by down-weighting easy examples.

    Formula:
        FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where:
        p_t = p if y=1, else (1-p)  [probability of correct class]
        p = sigmoid(ŷ)  [predicted probability after sigmoid]
        alpha_t = alpha if y=1, else (1-alpha)  [class balancing weight]
        gamma = focusing parameter (default: 2.0)
        alpha = positive class weight (default: 0.25)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        # Apply sigmoid to get probabilities
        y_pred_prob = torch.sigmoid(y_pred)

        # Compute BCE loss element-wise (no reduction)
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

        # Compute focal weight: (1 - p_t)^gamma
        # p_t is the probability of the correct class
        p_t = y_true * y_pred_prob + (1 - y_true) * (1 - y_pred_prob)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha balancing (weight positive class)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)

        # Focal loss with alpha balancing
        focal_loss = alpha_t * focal_weight * bce

        return torch.mean(focal_loss)


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with automatic positive class weighting.

    Formula:
        WeightedBCE = -[w * y * log(sigmoid(ŷ)) + (1-y) * log(1-sigmoid(ŷ))]

    where:
        w = pos_weight (weight for positive class)
        ŷ = logits (model output)
        sigmoid = sigmoid function
        y = ground truth (0 or 1)

    The pos_weight is typically computed as:
        pos_weight = num_negative_pixels / num_positive_pixels

    This helps handle class imbalance by up-weighting the minority class.
    """

    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight if pos_weight is not None else torch.tensor(1.0)

    def forward(self, y_pred, y_true):
        return F.binary_cross_entropy_with_logits(
            y_pred, y_true, pos_weight=self.pos_weight
        )

    @staticmethod
    def compute_pos_weight(train_loader, device="cuda"):
        """Compute pos_weight for WeightedBCELoss from training data."""
        num_positive = 0
        num_negative = 0

        # Check the dataset class to determine the batch structure
        dataset_class = train_loader.dataset.dataset.__class__.__name__

        for batch in train_loader:
            if dataset_class == "PH2":  # PH2: (image, mask, case_id)
                _, mask, _ = batch
            elif (
                dataset_class == "DRIVE"
            ):  # DRIVE: (image, vessel_mask, fov_mask, case_id)
                _, mask, _, _ = batch
            else:
                raise ValueError(f"Unsupported dataset class: {dataset_class}")

            num_positive += mask.sum().item()
            num_negative += (1 - mask).sum().item()

        pos_weight = torch.tensor(num_negative / num_positive, device=device)
        logger.info(
            f"Computed pos_weight: {pos_weight.item():.4f} "
            f"(neg: {num_negative:,}, pos: {num_positive:,})"
        )
        return pos_weight
