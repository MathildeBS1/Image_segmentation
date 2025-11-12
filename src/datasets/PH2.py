import torch
import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import random_split

DATA_PATH = "/dtu/datasets1/02516/PH2_Dataset_images"


class PH2(torch.utils.data.Dataset):
    """PH2 skin lesion segmentation dataset."""

    def __init__(self, data_path=DATA_PATH, transform=None):
        self.data_path = data_path
        self.transform = transform

        self.images = []
        self.masks = []
        self.case_ids = []

        self.mean = None
        self.std = None

        sample_dirs = sorted(glob.glob(os.path.join(data_path, "IMD*")))

        for sample_dir in sample_dirs:
            sample_id = os.path.basename(sample_dir)

            image_path = os.path.join(
                sample_dir, f"{sample_id}_Dermoscopic_Image", f"{sample_id}.bmp"
            )
            mask_path = os.path.join(
                sample_dir, f"{sample_id}_lesion", f"{sample_id}_lesion.bmp"
            )

            if os.path.exists(image_path) and os.path.exists(mask_path):
                self.images.append(image_path)
                self.masks.append(mask_path)
                self.case_ids.append(sample_id)

        assert len(self.images) == len(self.masks), (
            "Mismatch in number of images and masks"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        mask = Image.open(self.masks[idx])
        case_id = self.case_ids[idx]

        image = image.resize((512, 512), Image.BILINEAR)
        mask = mask.resize((512, 512), Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.mean is not None and self.std is not None:
            image = (image - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)

        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, case_id

    def set_normalization(self, mean, std):
        """Set normalization parameters (mean and std per channel)."""
        self.mean = mean
        self.std = std

    def _compute_normalization_stats(self, train_indices):
        """Compute mean and std per channel from training indices."""
        all_pixels = []

        for idx in train_indices:
            image = Image.open(self.images[idx]).resize((512, 512), Image.BILINEAR)
            image = np.array(image).astype(np.float32) / 255.0
            all_pixels.append(image.reshape(-1, 3))

        all_pixels = np.concatenate(all_pixels, axis=0)
        mean = torch.tensor(all_pixels.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(all_pixels.std(axis=0), dtype=torch.float32)

        return mean, std

    @staticmethod
    def get_dataloaders(batch_size=4, num_workers=2, seed=67):
        """Create train, validation, and test dataloaders with 60/20/20 split."""

        dataset = PH2()

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
    train_loader, val_loader, test_loader = PH2.get_dataloaders(
        batch_size=4, num_workers=2
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    dataset = train_loader.dataset.dataset
    print("\nNormalization stats:")
    print(f"Mean: {dataset.mean}")
    print(f"Std: {dataset.std}")

    image, mask, case_ids = next(iter(train_loader))
    print("\nBatch shapes:")
    print(f"Image: {image.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Case IDs: {case_ids}")
