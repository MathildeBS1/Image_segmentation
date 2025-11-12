from pathlib import Path
from src.datasets.PH2 import PH2
from src.datasets.DRIVE import DRIVE
from src.models.u_net import UNet
from src.models.enc_dec import EncDec
from src.models.losses import BCELoss, DiceLoss, FocalLoss, WeightedBCELoss
from src.utils.logger import get_logger
from datetime import datetime

logger = get_logger(__name__)


def create_dataloaders(dataset_name, batch_size, num_workers=4):
    """Create dataloaders for specified dataset."""
    if dataset_name == "PH2":
        train_loader, val_loader, test_loader = PH2.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=67,
        )
    elif dataset_name == "DRIVE":
        train_loader, val_loader, test_loader = DRIVE.get_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=67,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def create_model(model_name, device):
    """Create model."""
    if model_name == "unet":
        model = UNet(in_channels=3, out_channels=1, init_features=64)
    elif model_name == "encdec":
        model = EncDec(in_channels=3, out_channels=1, init_features=64)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name}")
    logger.info(f"  Parameters: {num_params:,}")

    return model


def create_loss(loss_name, train_loader, device):
    """Create loss function."""
    if loss_name == "bce":
        criterion = BCELoss()
    elif loss_name == "dice":
        criterion = DiceLoss()
    elif loss_name == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    elif loss_name == "weighted_bce":
        logger.info("Computing pos_weight from training data...")
        pos_weight = WeightedBCELoss.compute_pos_weight(train_loader, device=device)
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    logger.info(f"Loss: {loss_name}")
    return criterion


def setup_checkpoint_dir(base_dir, dataset, model, loss, timestamp=None):
    """Create checkpoint directory with naming convention."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Format: {dataset}_{model}_{loss}_{timestamp}
    checkpoint_dir = Path(base_dir) / f"{dataset}_{model}_{loss}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    return checkpoint_dir


def log_training_config(args):
    """Log training configuration."""
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Loss: {args.loss}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
