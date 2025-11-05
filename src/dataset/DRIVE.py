import torch
import os
import glob
from PIL import Image 

DATA_PATH = '/dtu/datasets1/02516/DRIVE'


class DRIVE(torch.utils.data.Dataset):
    ... # (implementation of the DRIVE dataset class)