

class SegmentationMetrics:
    """
    Compute segmentation evaluation metrics.
    
    All metrics expect:
    - pred: model predictions (logits or probabilities), shape (B, 1, H, W) or (B, H, W)
    - target: ground truth labels, shape (B, 1, H, W) or (B, H, W), values in {0, 1}
    """
    
    @staticmethod
    def _prepare_tensors(pred, target, threshold=0.5):
        """Convert to binary predictions and ensure same shape."""
        # Convert to float
        pred = pred.float()
        target = target.float()
        
        # Apply threshold if needed
        pred = (pred > threshold).float()
        
        # Flatten batch and spatial dimensions for easier computation
        pred = pred.reshape(-1)
        target = target.reshape(-1)
        
        return pred, target
    
    @staticmethod
    def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
        """
        Dice Coefficient (F1 score for segmentation).
        
        Range: [0, 1], where 1 is perfect overlap.
        Formula: 2 * |X ∩ Y| / (|X| + |Y|)
        
        """
        pred, target = SegmentationMetrics._prepare_tensors(pred, target, threshold)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return dice.item()
    
    @staticmethod
    def iou(pred, target, threshold=0.5, smooth=1e-6):
        """
        Intersection over Union (Jaccard Index).
        
        Range: [0, 1], where 1 is perfect overlap.
        Formula: |X ∩ Y| / |X ∪ Y|
        """
        pred, target = SegmentationMetrics._prepare_tensors(pred, target, threshold)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou_score = (intersection + smooth) / (union + smooth)
        
        return iou_score.item()
    
    @staticmethod
    def accuracy(pred, target, threshold=0.5):
        """
        Pixel-wise Accuracy.
        
        Range: [0, 1], where 1 means all pixels correctly classified.
        Formula: (TP + TN) / (TP + TN + FP + FN)
    
        """
        pred, target = SegmentationMetrics._prepare_tensors(pred, target, threshold)
        
        correct = (pred == target).float().sum()
        total = target.numel()
        
        return (correct / total).item()
    
    @staticmethod
    def sensitivity(pred, target, threshold=0.5, smooth=1e-6):
        """
        Sensitivity / Recall / True Positive Rate (TPR).
        
        Range: [0, 1], where 1 means all positive pixels are found.
        Formula: TP / (TP + FN)
        """
        pred, target = SegmentationMetrics._prepare_tensors(pred, target, threshold)
        
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        
        return sensitivity.item()
    
    @staticmethod
    def specificity(pred, target, threshold=0.5, smooth=1e-6):
        """
        Specificity / True Negative Rate (TNR).
        
        Range: [0, 1], where 1 means all negative pixels are correctly identified.
        Formula: TN / (TN + FP)
        """
        pred, target = SegmentationMetrics._prepare_tensors(pred, target, threshold)
        
        tn = ((1 - pred) * (1 - target)).sum()
        fp = (pred * (1 - target)).sum()
        specificity = (tn + smooth) / (tn + fp + smooth)
        
        return specificity.item()
    
    @staticmethod
    def compute_all_metrics(pred, target, threshold=0.5):
        """
        Compute all metrics at once.
        
        Args:
            pred: Model predictions
            target: Ground truth labels
            threshold: Threshold for binarizing predictions
            
        Returns:
            Dictionary with all metric values
        """
        return {
            'dice': SegmentationMetrics.dice_coefficient(pred, target, threshold),
            'iou': SegmentationMetrics.iou(pred, target, threshold),
            'accuracy': SegmentationMetrics.accuracy(pred, target, threshold),
            'sensitivity': SegmentationMetrics.sensitivity(pred, target, threshold),
            'specificity': SegmentationMetrics.specificity(pred, target, threshold),
        }
    
    @staticmethod
    def print_metrics(metrics, stage='Validation'):
        """
        Pretty please print all metrics.
        
        Args:
            metrics: Dictionary of metrics (from compute_all_metrics)
            stage: Name of the stage (e.g., 'Validation', 'Test')
        """
        print(f"\n{'='*50}")
        print(f"{stage} Metrics:")
        print(f"{'='*50}")
        print(f"  Dice Coefficient:  {metrics['dice']:.4f}")
        print(f"  IoU (Jaccard):     {metrics['iou']:.4f}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Sensitivity (TPR): {metrics['sensitivity']:.4f}")
        print(f"  Specificity (TNR): {metrics['specificity']:.4f}")
        print(f"{'='*50}\n")