from torch.utils.data import Dataset
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

class FramesDataset(Dataset):
    def __init__(self, data, augmentation=False, target_size=(256, 256)):
        """
        data: List of dicts with 'name', 'video', 'frames', 'label' keys
        augmentation: Whether to apply data augmentation
        """
        self.data = data
        self.augmentation = augmentation
        self.samples = []
        self.target_size = target_size
        for sample in data:
            video = sample['video']  # (H, W, num_frames)
            mask = sample['label']    # (H, W, num_frames)
            for frame_idx in sample['frames']:
                self.samples.append({
                    'frame': video[:, :, frame_idx],
                    'label': mask[:, :, frame_idx]
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = self.samples[idx]['frame'].astype(np.uint8)
        mask = self.samples[idx]['label'].astype(np.uint8)

        if self.augmentation:
            img, mask = self.augment_sample(img, mask)

        # Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        # Normalize image to 0..1 float
        img = img.astype('float32') / 255.0

        # Give img 1 channel dimension
        img = img[np.newaxis, :, :]  # (1, H, W)
        #convert mask to long tensor for loss
        mask = mask.astype('long')    # (H, W)
        return img, mask
    
    def augment_sample(self, img, mask):
        h, w = img.shape
        # Apply augmentations with given probabilities
    
        # 2. Small rotation (±10°)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        
        # Kleine Verschiebungen (±5-10%)
        # Grund: ROI-Position variiert leicht
        if np.random.rand() > 0.5:
            tx, ty = np.random.randint(-h//10, h//10, 2)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w, h))
            mask = cv2.warpAffine(mask, M, (w, h))
        # 4. Contrast (70%)
        if np.random.rand() > 0.3:
            factor = np.random.uniform(0.7, 1.3)
            mean = img.mean()
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

        # 5. Elastic Deformation (50%)
        if np.random.rand() > 0.5:
            img, mask = self.elastic_transform(img, mask, alpha=10, sigma=3)

        return img, mask

    def elastic_transform(self, img, mask, alpha=10, sigma=3):
        shape = img.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (y + dy).reshape(-1), (x + dx).reshape(-1)
        
        img_warped = map_coordinates(img, indices, order=1).reshape(shape)
        mask_warped = map_coordinates(mask, indices, order=0).reshape(shape)
        return img_warped, mask_warped