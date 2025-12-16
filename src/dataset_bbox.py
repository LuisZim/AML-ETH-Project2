import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class BBoxDataset(Dataset):
    """
    Dataset for Bounding Box Regression.
    
    Expects data in format:
    {
        'video': (H, W, num_frames),
        'box': (H, W),  # Boolean mask indicating box region (same for all frames)
        'name': 'video_name'
    }
    
    Generates samples: one sample per frame, all frames share the same bbox.
    """
    def __init__(self, data, target_size=(256, 256), augment=True):
        self.data = data
        self.target_size = target_size
        self.augment = augment
        self.samples = []
        
        # Build sample list: each frame becomes one sample with the same video-level bbox
        for sample in data:
            video = np.asarray(sample['video'])
            box_mask = np.asarray(sample['box'])  # (H, W) - same for all frames
            name = sample.get('name', 'unknown')
            
            num_frames = video.shape[-1]
            for frame_idx in range(num_frames):
                frame = video[:, :, frame_idx]
                self.samples.append({
                    'frame': frame,
                    'box_mask': box_mask,  # (H, W) - stored once per video
                    'name': name
                })
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_bbox_from_mask(self, mask):
        """
        Extract normalized bbox coordinates [x_min, y_min, x_max, y_max] from boolean mask.
        
        Args:
            mask: Boolean array (H, W)
        
        Returns:
            Normalized bbox as [x_min, y_min, x_max, y_max] in [0, 1]
        """
        H, W = mask.shape
        coords = np.where(mask)
        
        if len(coords[0]) == 0:
            # Empty mask fallback
            return np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Normalize to [0, 1]
        bbox = np.array([
            x_min / W,
            y_min / H,
            x_max / W,
            y_max / H
        ], dtype=np.float32)
        
        return bbox
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        frame = sample_info['frame']
        box_mask = sample_info['box_mask']
        
        # Get frame and ensure it's float32
        frame = np.asarray(frame).astype(np.float32)
        
        # Extract bbox from original mask (before resizing)
        bbox_orig = self._extract_bbox_from_mask(box_mask)

        # === AUGMENTATION ===
        if self.augment:
            # Random rotation (small angles)
            if np.random.rand() > 0.7:
                angle = np.random.uniform(-15, 15)
                h, w = frame.shape
                center = (w // 2, h // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                frame = cv2.warpAffine(frame, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Random horizontal flip
            if np.random.rand() > 0.5:
                frame = cv2.flip(frame, 1)
                # Adjust bbox for flip: x_min, y_min, x_max, y_max -> (1-x_max), y_min, (1-x_min), y_max
                bbox_orig = np.array([1 - bbox_orig[2], bbox_orig[1], 1 - bbox_orig[0], bbox_orig[3]])
            
            # Random brightness
            if np.random.rand() > 0.6:
                brightness = np.random.uniform(0.7, 1.3)
                frame = np.clip(frame * brightness, 0, 255)
            
            # Random contrast
            if np.random.rand() > 0.6:
                contrast = np.random.uniform(0.7, 1.3)
                frame = np.clip((frame - 128) * contrast + 128, 0, 255)
            
            # Random Gaussian blur
            if np.random.rand() > 0.7:
                frame = cv2.GaussianBlur(frame, (5, 5), 0)
            
            # Random noise
            if np.random.rand() > 0.6:
                noise = np.random.normal(0, 8, frame.shape)
                frame = np.clip(frame + noise, 0, 255)

        
        # Resize frame to target size
        frame_resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize frame to [0, 1] (assuming input is grayscale [0, 255])
        frame_resized = frame_resized / 255.0
        
        # Convert to 3-channel (RGB) by replicating grayscale
        frame_rgb = np.stack([frame_resized, frame_resized, frame_resized], axis=0)  # (3, H, W)
        
        return {
            'image': torch.from_numpy(frame_rgb).float(),
            'bbox': torch.from_numpy(bbox_orig).float(),
            'name': sample_info['name']
        }