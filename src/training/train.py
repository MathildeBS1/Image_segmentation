from src.training.evaluate import evaluate
from src.training.helpers import save_checkpoint
from src.utils.logger import get_logger
from dataclasses import dataclass, field

logger = get_logger(__name__)


@dataclass
class TrainingHistory:
    """Training history across epochs."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_dice: list[float] = field(default_factory=list)
    val_iou: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    val_sensitivity: list[float] = field(default_factory=list)
    val_specificity: list[float] = field(default_factory=list)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        if len(batch) == 3:  # PH2
            images, masks, _ = batch
        else:  # DRIVE
            images, masks, _, _ = batch

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    checkpoint_dir,
    early_stopping=None,
    use_fov_mask=False,
):
    """Train model with validation and checkpointing."""
    history = TrainingHistory()
    best_val_dice = 0

    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, use_fov_mask=use_fov_mask
        )

        logger.info(f"  Val Loss: {val_metrics.loss:.4f}")
        logger.info(f"  Val Dice: {val_metrics.dice:.4f}")

        # Store history
        history.train_loss.append(train_loss)
        history.val_loss.append(val_metrics.loss)
        history.val_dice.append(val_metrics.dice)
        history.val_iou.append(val_metrics.iou)
        history.val_accuracy.append(val_metrics.accuracy)
        history.val_sensitivity.append(val_metrics.sensitivity)
        history.val_specificity.append(val_metrics.specificity)

        # Save best model
        if val_metrics.dice > best_val_dice:
            best_val_dice = val_metrics.dice
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics,
                f"{checkpoint_dir}/best_model.pth",
            )

        # Save last model
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_metrics,
            f"{checkpoint_dir}/last_model.pth",
            verbose=False,
        )

        # Early stopping
        if early_stopping:
            early_stopping(val_metrics.dice, epoch=epoch)
            if early_stopping.stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    return history
