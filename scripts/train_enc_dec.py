import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.optim import Adam

from src.training.train import train_model
from src.training.early_stopping import EarlyStopping
from src.training.helpers import save_history_to_csv
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from scripts.common import (
    create_dataloaders,
    create_model,
    create_loss,
    setup_checkpoint_dir,
    log_training_config,
)

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train Encoder-Decoder segmentation model"
    )

    # Experiment configuration
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["PH2", "DRIVE"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        choices=["bce", "dice", "focal", "weighted_bce"],
        help="Loss function",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs (default: 50)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )

    # System configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
    )

    # Output configuration
    parser.add_argument(
        "--checkpoint_base",
        type=str,
        default="checkpoints",
        help="Base directory for checkpoints (default: checkpoints)",
    )

    args = parser.parse_args()
    args.model = "encdec"  # Fixed model

    # Set seed for reproducibility
    set_seed(67)
    log_training_config(args)

    # Setup
    checkpoint_dir = setup_checkpoint_dir(
        args.checkpoint_base, args.dataset, args.model
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create components
    train_loader, val_loader, test_loader = create_dataloaders(
        args.dataset, args.batch_size, args.num_workers
    )
    model = create_model(args.model, device)
    criterion = create_loss(args.loss, train_loader, device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    logger.info(f"Optimizer: Adam (lr={args.lr})")

    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.patience, mode="max", delta=0.001, verbose=True
    )
    logger.info(f"Early stopping: patience={args.patience}, mode=max, delta=0.001")

    # Train
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    use_fov_mask = args.dataset == "DRIVE"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        early_stopping=early_stopping,
        use_fov_mask=use_fov_mask,
    )

    # Save results
    history_path = checkpoint_dir / "training_history.csv"
    save_history_to_csv(history, str(history_path))

    logger.info("=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation Dice: {max(history.val_dice):.4f}")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
