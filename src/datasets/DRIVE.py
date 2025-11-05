import torch
import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import random_split

DATA_PATH = '/dtu/datasets1/02516/DRIVE'


class DRIVE(torch.utils.data.Dataset):
    """DRIVE retinal vessel segmentation dataset."""
    
    def __init__(self, data_path=DATA_PATH, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        self.images = []
        self.vessel_masks = []
        self.fov_masks = []
        
        self.mean = None
        self.std = None
        
        self.images = sorted(glob.glob(os.path.join(data_path, 'training/images/*.tif')))
        self.vessel_masks = sorted(glob.glob(os.path.join(data_path, 'training/1st_manual/*.gif')))
        self.fov_masks = sorted(glob.glob(os.path.join(data_path, 'training/mask/*.gif')))
        
        assert len(self.images) == len(self.vessel_masks) == len(self.fov_masks), \
            "Mismatch in number of images and masks"
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        vessel_mask = np.array(Image.open(self.vessel_masks[idx]))
        fov_mask = np.array(Image.open(self.fov_masks[idx]))
        
        vessel_mask = (vessel_mask > 0).astype(np.float32)
        fov_mask = (fov_mask > 0).astype(np.float32)
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if self.mean is not None and self.std is not None:
            image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        
        vessel_mask = torch.from_numpy(vessel_mask).unsqueeze(0)
        fov_mask = torch.from_numpy(fov_mask).unsqueeze(0)
        
        if self.transform:
            image, vessel_mask, fov_mask = self.transform(image, vessel_mask, fov_mask)
        
        return image, vessel_mask, fov_mask
    
    def set_normalization(self, mean, std):
        """Set normalization parameters (mean and std per channel)."""
        self.mean = mean
        self.std = std
    
    def _compute_normalization_stats(self, train_indices):
        """Compute mean and std per channel from training indices."""
        all_pixels = []
        
        for idx in train_indices:
            image = np.array(Image.open(self.images[idx]))
            image = image.astype(np.float32) / 255.0
            all_pixels.append(image.reshape(-1, 3))
        
        all_pixels = np.concatenate(all_pixels, axis=0)
        mean = torch.tensor(all_pixels.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(all_pixels.std(axis=0), dtype=torch.float32)
        
        return mean, std
    
    @staticmethod
    def get_dataloaders(batch_size=4, num_workers=2, seed=67):
        """Create train, validation, and test dataloaders with 60/20/20 split."""
        
        dataset = DRIVE()
        
        total_size = len(dataset)
        train_size = int(0.6 * total_size)
        val_size = int(0.2 * total_size)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        mean, std = dataset._compute_normalization_stats(train_dataset.indices)
        dataset.set_normalization(mean, std)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = DRIVE.get_dataloaders(batch_size=4, num_workers=2)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    dataset = train_loader.dataset.dataset
    print(f"\nNormalization stats:")
    print(f"Mean: {dataset.mean}")
    print(f"Std: {dataset.std}")
    
    image, vessel_mask, fov_mask = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Image: {image.shape}")
    print(f"Vessel mask: {vessel_mask.shape}")
    print(f"FOV mask: {fov_mask.shape}")