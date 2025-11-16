"""Test all trained models and save results to CSV."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import pandas as pd
from datetime import datetime
from src.training.evaluate import evaluate
from src.utils.logger import get_logger
from scripts.common import create_dataloaders, create_model, create_loss

logger = get_logger(__name__)


def parse_checkpoint_name(checkpoint_dir):
    """
    Parse checkpoint directory name to extract dataset, model, loss.
    Expected format: {DATASET}_{MODEL}_{LOSS}_{TIMESTAMP}
    Example: PH2_unet_dice_20251112_183723
    """
    dirname = Path(checkpoint_dir).name
    parts = dirname.split('_')
    
    if len(parts) >= 4:
        dataset = parts[0]
        model = parts[1]
        loss = parts[2]
        return dataset, model, loss
    else:
        logger.warning(f"Could not parse directory name: {dirname}")
        return None, None, None


def test_single_model(checkpoint_path, dataset, model, loss, device, batch_size=4, num_workers=4):
    """Test a single model on all three splits."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Testing: {dataset}_{model}_{loss}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info("=" * 80)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size, num_workers
    )
    
    # Create model and load checkpoint
    model_obj = create_model(model, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_obj.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"Loaded checkpoint from epoch: {epoch}")
    
    # Create loss function
    criterion = create_loss(loss, train_loader, device)
    
    # FOV mask for DRIVE
    use_fov_mask = dataset == "DRIVE"
    
    # Evaluate on all splits
    results = {
        'dataset': dataset,
        'model': model,
        'loss': loss,
        'epoch': epoch,
        'checkpoint': str(checkpoint_path)
    }
    
    for split_name, dataloader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Evaluating on {split_name} set...")
        
        metrics = evaluate(
            model=model_obj,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            use_fov_mask=use_fov_mask
        )
        
        # Store metrics with split prefix
        results[f'{split_name}_dice'] = metrics.dice
        results[f'{split_name}_iou'] = metrics.iou
        results[f'{split_name}_accuracy'] = metrics.accuracy
        results[f'{split_name}_sensitivity'] = metrics.sensitivity
        results[f'{split_name}_specificity'] = metrics.specificity
        results[f'{split_name}_loss'] = metrics.loss
        
        logger.info(f"  Dice: {metrics.dice:.4f}, IoU: {metrics.iou:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test all trained models")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Base directory containing checkpoints (default: checkpoints)")
    parser.add_argument("--output_csv", type=str, default="test_results.csv",
                        help="Output CSV file (default: test_results.csv)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers (default: 4)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (default: cuda)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    
    args = parser.parse_args()
    
    # Set GPU
    if torch.cuda.is_available() and args.device == "cuda":
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device("cuda:0")
        logger.info(f"Using GPU {args.gpu_id}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Find all checkpoint directories
    checkpoint_base = Path(args.checkpoint_dir)
    checkpoint_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    
    if not checkpoint_dirs:
        logger.error(f"No checkpoint directories found in {checkpoint_base}")
        return
    
    logger.info(f"Found {len(checkpoint_dirs)} checkpoint directories")
    
    # Test all models
    all_results = []
    
    for checkpoint_dir in sorted(checkpoint_dirs):
        checkpoint_path = checkpoint_dir / "best_model.pth"
        
        if not checkpoint_path.exists():
            logger.warning(f"No best_model.pth found in {checkpoint_dir}, skipping...")
            continue
        
        # Parse checkpoint name
        dataset, model, loss = parse_checkpoint_name(checkpoint_dir)
        
        if dataset is None or model is None or loss is None:
            logger.warning(f"Skipping {checkpoint_dir} (could not parse name)")
            continue
        
        try:
            results = test_single_model(
                checkpoint_path=checkpoint_path,
                dataset=dataset,
                model=model,
                loss=loss,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            all_results.append(results)
            
        except Exception as e:
            logger.error(f"Error testing {checkpoint_dir}: {e}")
            continue
    
    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Reorder columns for clarity
        column_order = [
            'dataset', 'model', 'loss', 'epoch', 'checkpoint',
            'train_dice', 'train_iou', 'train_accuracy', 'train_sensitivity', 'train_specificity', 'train_loss',
            'val_dice', 'val_iou', 'val_accuracy', 'val_sensitivity', 'val_specificity', 'val_loss',
            'test_dice', 'test_iou', 'test_accuracy', 'test_sensitivity', 'test_specificity', 'test_loss',
        ]
        df = df[column_order]
        
        df.to_csv(args.output_csv, index=False)
        logger.info(f"\n✓ Results saved to {args.output_csv}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY OF ALL MODELS (TEST SET DICE SCORES)")
        logger.info("=" * 80)
        summary = df[['dataset', 'model', 'loss', 'test_dice']].sort_values('test_dice', ascending=False)
        logger.info("\n" + summary.to_string(index=False))
        logger.info("=" * 80)
        
    else:
        logger.error("No results to save!")


if __name__ == "__main__":
    main()