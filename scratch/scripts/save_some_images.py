import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.datasets.DRIVE import DRIVE
from src.datasets.PH2 import PH2


def denormalize(image, mean, std):
    """Denormalize image for visualization."""
    image = image.clone()
    for i in range(3):
        image[i] = image[i] * std[i] + mean[i]
    return image


def visualize_samples(dataset_name, loader, num_samples=3, has_fov=False):
    """Visualize samples from a dataloader."""
    dataset = loader.dataset.dataset
    mean = dataset.mean
    std = dataset.std
    
    samples = []
    for batch in loader:
        if has_fov:
            images, masks, fov_masks, case_ids = batch
        else:
            images, masks, case_ids = batch
        
        for i in range(len(images)):
            samples.append((images[i], masks[i], case_ids[i]))
            if len(samples) >= num_samples:
                break
        if len(samples) >= num_samples:
            break
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (image, mask, case_id) in enumerate(samples):
        image_denorm = denormalize(image, mean, std)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np, 0, 1)
        mask_np = mask[0].cpu().numpy()
        
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f'{dataset_name} - {case_id}\nOriginal Image')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(mask_np, cmap='gray')
        axes[idx, 1].set_title('Mask')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(image_np)
        axes[idx, 2].imshow(mask_np, cmap='Reds', alpha=0.4)
        axes[idx, 2].set_title('Image + Mask Overlay')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path(f'{dataset_name.lower()}_samples.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved {output_path}')
    plt.close()


if __name__ == "__main__":
    print("Loading DRIVE dataset...")
    drive_train, _, _ = DRIVE.get_dataloaders(batch_size=4, num_workers=0)
    visualize_samples('DRIVE', drive_train, num_samples=3, has_fov=True)
    
    print("Loading PH2 dataset...")
    ph2_train, _, _ = PH2.get_dataloaders(batch_size=4, num_workers=0)
    visualize_samples('PH2', ph2_train, num_samples=3, has_fov=False)
    
    print("\nDone! Check drive_samples.png and ph2_samples.png in the project root.")