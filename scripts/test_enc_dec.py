import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from src.training.evaluate import evaluate
from src.metrics.segmentation import SegmentationMetrics
from src.utils.logger import get_logger
from scripts.common import create_dataloaders, create_model, create_loss

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["PH2", "DRIVE"])
    parser.add_argument("--loss", type=str, required=True, choices=["bce", "dice", "focal", "weighted_bce"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    args.model = "encdec"
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Load data (only need test loader)
    _, _, test_loader = create_dataloaders(args.dataset, args.batch_size, args.num_workers)
    
    # Create model and load checkpoint
    model = create_model(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create loss function
    criterion = create_loss(args.loss, test_loader, device)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 80)
    
    use_fov_mask = args.dataset == "DRIVE"
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        use_fov_mask=use_fov_mask
    )
    
    # Print segmentation metrics only
    metrics_dict = {
        'dice': test_metrics.dice,
        'iou': test_metrics.iou,
        'accuracy': test_metrics.accuracy,
        'sensitivity': test_metrics.sensitivity,
        'specificity': test_metrics.specificity,
    }
    
    SegmentationMetrics.print_metrics(metrics_dict, stage="Test Set")
    
    # Also log them
    logger.info(f"Test Dice:        {test_metrics.dice:.4f}")
    logger.info(f"Test IoU:         {test_metrics.iou:.4f}")
    logger.info(f"Test Accuracy:    {test_metrics.accuracy:.4f}")
    logger.info(f"Test Sensitivity: {test_metrics.sensitivity:.4f}")
    logger.info(f"Test Specificity: {test_metrics.specificity:.4f}")


if __name__ == "__main__":
    main()