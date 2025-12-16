import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class ROIDataset(Dataset):
    """
    Dataset for Segmentation using Region of Interest (ROI) cropping.
    
    Expects data in format:
    {
        'video': (H, W, num_frames),
        'box': (H, W),  # Boolean mask indicating ROI region (same for all frames)
        'label': (H, W, num_frames),  # Segmentation labels
        'name': 'video_name'
    }
    
    Returns cropped images and labels within the ROI bounding box.
    """
    def __init__(self, data, target_size=(256, 256), augment=False):
        """
        Args:
            data: List of video samples with 'video', 'box', 'label', 'name' keys
            target_size: (H, W) tuple for resizing cropped ROI
            augment: Whether to apply data augmentation
        """
        self.data = data
        self.target_size = target_size
        self.augment = augment
        self.samples = []
        
        # Build sample list: each frame that has a label becomes one sample
        # The indeces of labeled frames in video are given in sample['frames]
        for sample in data:
            video = np.asarray(sample['video'])
            box_mask = np.asarray(sample['box'])  # (H, W) - same for all frames
            label = np.asarray(sample['label'])   # (H, W, num_frames)
            name = sample.get('name', 'unknown')
            
            # Extract bbox coordinates from mask
            bbox = self._extract_bbox_from_mask(box_mask.astype(bool))
            
            for frame_idx in sample['frames']:
                self.samples.append({
                    'img': video[:, :, frame_idx],
                    'label': label[:, :, frame_idx],
                    'bbox': bbox,  # (4,) normalized [x_min, y_min, x_max, y_max]
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
            # Empty mask fallback - use full image
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
    
    def _crop_roi(self, image, bbox, orig_h, orig_w):
        """
        Crop image to ROI defined by bbox.
        
        Args:
            image: 2D array (H, W)
            bbox: normalized [x_min, y_min, x_max, y_max] in [0, 1]
            orig_h, orig_w: original image dimensions
        
        Returns:
            Cropped image array
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to pixel coordinates
        x1 = int(x_min * orig_w)
        y1 = int(y_min * orig_h)
        x2 = int(x_max * orig_w)
        y2 = int(y_max * orig_h)
        
        # Clamp to image bounds
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(x1 + 1, min(x2, orig_w))
        y2 = max(y1 + 1, min(y2, orig_h))
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        return cropped
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        image = sample_info['img']
        label = sample_info['label']
        bbox = sample_info['bbox']
        
        # Get frame and label mask
        frame = image.astype(np.float32)
        label_mask = label.astype(np.float32)
        
        orig_h, orig_w = frame.shape
        
        # Crop to ROI
        frame_cropped = self._crop_roi(frame, bbox, orig_h, orig_w)
        label_cropped = self._crop_roi(label_mask, bbox, orig_h, orig_w)
        
        # === AUGMENTATION (only if enabled) ===
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                frame_cropped = cv2.flip(frame_cropped, 1)
                label_cropped = cv2.flip(label_cropped, 1)
            
            # Random brightness (only on image, not label)
            if np.random.rand() > 0.6:
                brightness = np.random.uniform(0.7, 1.3)
                frame_cropped = np.clip(frame_cropped * brightness, 0, 255)
            
            # Random contrast
            if np.random.rand() > 0.6:
                contrast = np.random.uniform(0.7, 1.3)
                frame_cropped = np.clip((frame_cropped - 128) * contrast + 128, 0, 255)
            
            # Random Gaussian blur
            if np.random.rand() > 0.7:
                frame_cropped = cv2.GaussianBlur(frame_cropped, (5, 5), 0)
            
            # Random noise
            if np.random.rand() > 0.6:
                noise = np.random.normal(0, 8, frame_cropped.shape)
                frame_cropped = np.clip(frame_cropped + noise, 0, 255)
        
        # Resize cropped ROI to target size
        frame_resized = cv2.resize(frame_cropped, self.target_size, interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize frame to [0, 1]
        frame_resized = frame_resized / 255.0
        
        # Convert to tensor format
        # Image: (1, H, W) for grayscale
        image_tensor = torch.from_numpy(frame_resized[np.newaxis, ...]).float()
        
        # Label: (H, W) as long tensor for segmentation
        label_tensor = torch.from_numpy(label_resized).long()
        
        return image_tensor, label_tensor
