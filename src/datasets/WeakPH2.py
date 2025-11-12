import numpy as np
import torch
from torch.utils.data import DataLoader

from .PH2 import PH2
from .click_sampling import generate_clicks  # (mask_np, N, M) -> (pos_list, neg_list)

class WeakPH2(torch.utils.data.Dataset):
    """
    Wrapper around a PH2 subset that returns click annotations instead of full masks.
    """
    def __init__(self, base_dataset, n_positive=5, n_negative=5, rng_seed=None):
        super().__init__()
        self.base = base_dataset           
        self.n_pos = int(n_positive)
        self.n_neg = int(n_negative)
        if rng_seed is not None:
            np.random.seed(int(rng_seed))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, mask, case_id = self.base[idx]

        # [H,W] binary mask -> numpy
        if isinstance(mask, torch.Tensor):
            mask_np = (mask.squeeze().detach().cpu().numpy() > 0).astype(np.uint8)
        else:
            mask_np = (np.array(mask) > 0).astype(np.uint8)

        N, M = self.n_pos, self.n_neg
        try:
            pos_clicks, neg_clicks = generate_clicks(mask_np, N, M)
        except ValueError:
            # fallback if fewer pixels than requested (replacement)
            pos_idx = np.column_stack(np.where(mask_np == 1))
            neg_idx = np.column_stack(np.where(mask_np == 0))
            if pos_idx.size == 0: pos_idx = np.array([[0, 0]])
            if neg_idx.size == 0: neg_idx = np.array([[0, 0]])
            pos_sel = pos_idx[np.random.choice(pos_idx.shape[0], N, replace=True)]
            neg_sel = neg_idx[np.random.choice(neg_idx.shape[0], M, replace=True)]
            pos_clicks = [tuple(c) for c in pos_sel]
            neg_clicks = [tuple(c) for c in neg_sel]

        clicks = pos_clicks + neg_clicks
        labels = [1] * len(pos_clicks) + [0] * len(neg_clicks)

        clicks_coords = torch.tensor(clicks, dtype=torch.long)   # [K,2] (y,x)
        clicks_labels = torch.tensor(labels, dtype=torch.long)   # [K]
        return image, clicks_coords, clicks_labels, case_id

    @staticmethod
    def get_dataloaders(
        n_positive=5, n_negative=5,
        batch_size=4, num_workers=2, seed=67, rng_seed=None
    ):
        """
        Same signature/behavior style as PH2.get_dataloaders, but returns WeakPH2 loaders.
        1) Build PH2 loaders (computes mean/std on train, creates splits)
        2) Wrap each PH2 subset with WeakPH2
        3) Return new DataLoaders yielding clicks instead of masks
        """

        base_train_loader, base_val_loader, base_test_loader = PH2.get_dataloaders(
            batch_size=1, num_workers=num_workers, seed=seed
        )

        base_train_subset = base_train_loader.dataset
        base_val_subset   = base_val_loader.dataset
        base_test_subset  = base_test_loader.dataset

        weak_train = WeakPH2(base_train_subset, n_positive, n_negative, rng_seed=rng_seed)
        weak_val   = WeakPH2(base_val_subset,   n_positive, n_negative, rng_seed=rng_seed)
        weak_test  = WeakPH2(base_test_subset,  n_positive, n_negative, rng_seed=rng_seed)

        train_loader = DataLoader(weak_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
        val_loader   = DataLoader(weak_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader  = DataLoader(weak_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader
