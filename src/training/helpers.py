import torch
import os
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path, verbose=True):
    """Save model checkpoint."""

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, checkpoint_path)

    if verbose:
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(
            f"Checkpoint saved: {checkpoint_path} (epoch {epoch}, {metric_str})"
        )


def load_checkpoint(
    model, optimizer=None, checkpoint_path=None, device="cuda", verbose=True
):
    """Load model checkpoint."""
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        if verbose:
            logger.warning(f"✗ Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if verbose:
        epoch = checkpoint.get("epoch", "unknown")
        metrics = checkpoint.get("metrics", {})
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(
            f"✓ Checkpoint loaded: {checkpoint_path} (epoch {epoch}, {metric_str})"
        )

    return checkpoint


def save_training_state(
    model, optimizer, scheduler, early_stopping, epoch, checkpoint_dir
):
    """Save complete training state (for resuming training)."""
    checkpoint_path = os.path.join(checkpoint_dir, "training_state.pth")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "early_stopping": {
            "best_metric": early_stopping.best_metric,
            "counter": early_stopping.counter,
            "best_epoch": early_stopping.best_epoch,
        },
    }

    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(state, checkpoint_path)
    logger.info(f"✓ Training state saved: {checkpoint_path}")
