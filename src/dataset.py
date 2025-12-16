# Zelle: PyTorch Dataset (flache Listen)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FramesDataset(Dataset):
    def __init__(self, frames, masks, crop):
        """
        frames: list or array of np.ndarray images, shape (H,W) (values 0..1 float)
        masks:  list or array of np.ndarray masks, shape (H,W) (values 0/1)
        """
        assert len(frames) == len(masks)
        self.frames = frames
        self.masks = masks

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = np.asarray(self.frames[idx])
        mask = np.asarray(self.masks[idx])

        # Normalize image to 0..1 float
        img = img.astype('float32') / 255.0

        # to torch tensors: (C,H,W) and mask (H,W)
        img_t = torch.from_numpy(img.transpose(2,0,1)).float() # C,H,W (For Conv2d)
        mask_t = torch.from_numpy(mask[...,0]).long()          # H,W (Long for CrossEntropy)
        return img_t, mask_t