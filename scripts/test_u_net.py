import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import os
from src.training.evaluate import evaluate
from src.metrics.segmentation import SegmentationMetrics
from src.utils.logger import get_logger
from scripts.common import create_dataloaders, create_model, create_loss

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test trained U-Net model on all splits")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["PH2", "DRIVE"])
    parser.add_argument("--loss", type=str, required=True, choices=["bce", "dice", "focal", "weighted_bce"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:1", help="Device (cuda:0, cuda:1, or cpu)")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU ID to use (default: 1)")
    
    args = parser.parse_args()
    args.model = "unet"
    
    # Set GPU device
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device("cuda:0")  # After setting CUDA_VISIBLE_DEVICES, it becomes cuda:0
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: cuda (GPU {args.gpu_id})")
    logger.info(f"Model: U-Net")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Load all three dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, args.num_workers
    )
    
    # Create model and load checkpoint
    model = create_model(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create loss function
    criterion = create_loss(args.loss, train_loader, device)
    
    # Determine if FOV mask should be used
    use_fov_mask = args.dataset == "DRIVE"
    
    # Define all splits to evaluate
    splits = [
        ("TRAIN", train_loader),
        ("VALIDATION", val_loader),
        ("TEST", test_loader)
    ]
    
    all_results = {}
    
    # Evaluate on each split
    for split_name, dataloader in splits:
        # Clear cache before each evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"EVALUATING ON {split_name} SET")
        logger.info("=" * 80)
        
        metrics = evaluate(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            use_fov_mask=use_fov_mask
        )
        
        # Store results for this split
        metrics_dict = {
            'dice': metrics.dice,
            'iou': metrics.iou,
            'accuracy': metrics.accuracy,
            'sensitivity': metrics.sensitivity,
            'specificity': metrics.specificity,
        }
        all_results[split_name] = metrics_dict
        
        # Print metrics for this split
        SegmentationMetrics.print_metrics(metrics_dict, stage=f"{split_name.title()} Set")
        
        # Log individual metrics
        logger.info(f"{split_name} Dice:        {metrics.dice:.4f}")
        logger.info(f"{split_name} IoU:         {metrics.iou:.4f}")
        logger.info(f"{split_name} Accuracy:    {metrics.accuracy:.4f}")
        logger.info(f"{split_name} Sensitivity: {metrics.sensitivity:.4f}")
        logger.info(f"{split_name} Specificity: {metrics.specificity:.4f}")
    
    # Print comprehensive summary table
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: ALL FIVE PERFORMANCE MEASURES ON ALL THREE SPLITS")
    logger.info("=" * 80)
    logger.info(f"{'Metric':<15} {'Train':<12} {'Validation':<12} {'Test':<12}")
    logger.info("-" * 80)
    
    metric_names = ['dice', 'iou', 'accuracy', 'sensitivity', 'specificity']
    metric_display = {
        'dice': 'Dice',
        'iou': 'IoU',
        'accuracy': 'Accuracy',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity'
    }
    
    for metric in metric_names:
        logger.info(
            f"{metric_display[metric]:<15} "
            f"{all_results['TRAIN'][metric]:<12.4f} "
            f"{all_results['VALIDATION'][metric]:<12.4f} "
            f"{all_results['TEST'][metric]:<12.4f}"
        )
    
    logger.info("=" * 80)
    
    return all_results


if __name__ == "__main__":
    main()