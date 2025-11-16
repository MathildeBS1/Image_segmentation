import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
sns.set_palette("muted")


def load_training_history(checkpoint_dir):
    """Load training_history.csv from a checkpoint directory."""
    history_path = Path(checkpoint_dir) / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"No training_history.csv found in {checkpoint_dir}")
    return pd.read_csv(history_path)


def plot_comparison(unet_checkpoint, encdec_checkpoint, dataset_name, output_path="training_comparison.png"):
    """
    Plot training vs validation curves comparing U-Net and EncDec.
    
    Args:
        unet_checkpoint: Path to U-Net checkpoint directory
        encdec_checkpoint: Path to EncDec checkpoint directory
        dataset_name: Name of dataset (PH2 or DRIVE)
        output_path: Where to save the plot
    """
    logger.info(f"Loading U-Net history from: {unet_checkpoint}")
    unet_history = load_training_history(unet_checkpoint)
    
    logger.info(f"Loading EncDec history from: {encdec_checkpoint}")
    encdec_history = load_training_history(encdec_checkpoint)
    
    # Create figure with 2 subplots (loss and dice)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    ax = axes[0]
    ax.plot(unet_history['epoch'], unet_history['train_loss'], 
            label='U-Net Train', linewidth=2, alpha=0.8)
    ax.plot(unet_history['epoch'], unet_history['val_loss'], 
            label='U-Net Val', linewidth=2, alpha=0.8, linestyle='--')
    ax.plot(encdec_history['epoch'], encdec_history['train_loss'], 
            label='EncDec Train', linewidth=2, alpha=0.8)
    ax.plot(encdec_history['epoch'], encdec_history['val_loss'], 
            label='EncDec Val', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training vs Validation Loss ({dataset_name})', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Dice score curves
    ax = axes[1]
    ax.plot(unet_history['epoch'], unet_history['val_dice'], 
            label='U-Net Val', linewidth=2, alpha=0.8)
    ax.plot(encdec_history['epoch'], encdec_history['val_dice'], 
            label='EncDec Val', linewidth=2, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(f'Validation Dice Score ({dataset_name})', fontsize=14, fontweight='bold')
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=800, bbox_inches='tight')
    logger.info(f"✓ Plot saved to: {output_path}")
    
    plt.savefig(Path(output_path).with_suffix('.pdf'), bbox_inches='tight')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare U-Net vs EncDec training histories")
    parser.add_argument("--unet", type=str, required=True, 
                        help="Path to U-Net checkpoint directory")
    parser.add_argument("--encdec", type=str, required=True,
                        help="Path to EncDec checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True, choices=["PH2", "DRIVE"],
                        help="Dataset name for plot title")
    parser.add_argument("--output", type=str, default="plots/training_comparison.png",
                        help="Output path for plot (default: plots/training_comparison.png)")
    
    args = parser.parse_args()
    
    plot_comparison(
        unet_checkpoint=args.unet,
        encdec_checkpoint=args.encdec,
        dataset_name=args.dataset,
        output_path=args.output
    )


if __name__ == "__main__":
    main()