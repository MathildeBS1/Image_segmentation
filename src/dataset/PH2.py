import os
import glob
import random
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'

class PH2(torch.utils.data.Dataset):

    _SPLIT_CACHE: Dict[Tuple[str, int, Tuple[float, float, float]], Dict[str, List[str]]] = {}
    _TRAIN_STATS_CACHE: Dict[Tuple[str, int, Tuple[float, float, float]], Tuple[torch.Tensor, torch.Tensor]] = {}

    def __init__(
        self,
        split: 'train' | 'val' | 'test',                               # 'train' | 'val' | 'test'
        joint_transform=None,                          
        image_transform=None,                    
        mask_transform=None,  
    ):
        assert split in ('train','val','test')
        self.split = split
        self.data_path = DATA_PATH
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.seed = 67
        self.ratios = (0.6, 0.2, 0.2)

      # Build/reuse split
        key = (self.data_path, self.seed, self.ratios)
        if key not in PH2._SPLIT_CACHE:
            cases = self._find_cases()
            PH2._SPLIT_CACHE[key] = self._split_cases(cases)
        split_dict = PH2._SPLIT_CACHE[key]

        # Collect samples
        self.samples: List[Tuple[str, str, str]] = []
        for cid in split_dict[self.split]:
            img = os.path.join(self.data_path, cid, f'{cid}_Dermoscopic_Image', f'{cid}.bmp')
            msk = os.path.join(self.data_path, cid, f'{cid}_lesion', f'{cid}_lesion.bmp')
            if os.path.isfile(img) and os.path.isfile(msk):
                self.samples.append((img, msk, cid))
        if not self.samples:
            raise RuntimeError(f"No samples found for split '{split}' under {self.data_path}")

        if self.split == 'train':
            if key in PH2._TRAIN_STATS_CACHE:
                self._mean, self._std = PH2._TRAIN_STATS_CACHE[key]
            else:
                train_ids = split_dict['train']
                train_imgs = [
                    os.path.join(self.data_path, cid, f'{cid}_Dermoscopic_Image', f'{cid}.bmp')
                    for cid in train_ids
                ]
                self._mean, self._std = self._compute_channel_stats(train_imgs)
                PH2._TRAIN_STATS_CACHE[key] = (self._mean, self._std)

    def _find_cases(self) -> List[str]:
        cases = []
        for name in os.listdir(self.data_path):
            if not name.startswith('IMD'):
                continue
            case_dir = os.path.join(self.data_path, name)
            if not os.path.isdir(case_dir):
                continue
            derm_dir = os.path.join(case_dir, f'{name}_Dermoscopic_Image')
            lesion_dir = os.path.join(case_dir, f'{name}_lesion')
            img_path = os.path.join(derm_dir, f'{name}.bmp')
            mask_path = os.path.join(lesion_dir, f'{name}_lesion.bmp')
            if os.path.isfile(img_path) and os.path.isfile(mask_path):
                cases.append(name)
        cases.sort()
        return cases

    def _split_cases(
        cases: List[str],
    ) -> Dict[str, List[str]]:
        """
        Deterministic split of case IDs into train/val/test using given ratios and seed.
        """
        train_ratio, val_ratio, test_ratio = self.ratios
        rng = random.Random(self.seed)
        shuffled = cases[:]
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val   = int(round(n * val_ratio))
        # ensure all samples used
        n_test  = n - n_train - n_val

        train_ids = shuffled[:n_train]
        val_ids   = shuffled[n_train:n_train+n_val]
        test_ids  = shuffled[n_train+n_val:]

        return {"train": train_ids, "val": val_ids, "test": test_ids}


    def _pil_to_tensor_image(self, img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0  # H,W,3
        arr = np.transpose(arr, (2, 0, 1))                            # 3,H,W
        return torch.from_numpy(arr)

    def _pil_to_tensor_mask(self, mask: Image.Image) -> torch.Tensor:
        m = np.array(mask.convert('L'), dtype=np.uint8)
        m = (m > 0).astype(np.uint8)[None, ...]  # 1,H,W
        return torch.from_numpy(m).long()

    def _compute_channel_stats(self, img_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming per-channel mean/std over a list of image paths.
        Images are read at native size and converted to [0,1] floats.
        """
        sum_c = torch.zeros(3, dtype=torch.float64)
        sumsq_c = torch.zeros(3, dtype=torch.float64)
        count = 0

        for p in img_paths:
            x = self._pil_to_tensor_image(Image.open(p)).to(dtype=torch.float64)  # [3,H,W]
            c, h, w = x.shape
            n = h * w
            sum_c += x.view(c, -1).sum(dim=1)
            sumsq_c += (x * x).view(c, -1).sum(dim=1)
            count += n

        mean = (sum_c / count).to(dtype=torch.float32)                     # [3]
        var = (sumsq_c / count - (sum_c / count) ** 2).to(dtype=torch.float32)
        std = torch.sqrt(torch.clamp(var, min=1e-12))                      # [3]
        return mean, std


    def __len__(self):
        'Returns the total number of samples'
        return len(self.samples)

    def __getitem__(self, idx: int):
        'Generates one sample of data'
        img_path, mask_path, cid = self.samples[idx]

        img = Image.open(img_path)
        msk = Image.open(mask_path)

        # joint transform first (keeps geometry aligned)
        if self.joint_transform is not None:
            img, msk = self.transform(img, msk)

        # per-modality transforms / default conversions
        if self.image_transform is not None:
            img = self.image_transform(img)
        else:
            img = self._pil_to_tensor_image(img)

        if self.mask_transform is not None:
            msk = self.mask_transform(msk)
        else:
            msk = self._pil_to_tensor_mask(msk)

        if self.split == 'train':
            mean = self._mean.view(-1, 1, 1)
            std = self._std.view(-1, 1, 1)
            img = (img - mean) / std

        return img, msk, cid
