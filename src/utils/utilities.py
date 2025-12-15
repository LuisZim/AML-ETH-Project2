import numpy as np
import torch
from tqdm import tqdm
import cv2
import gzip
import pickle

# Extracts corresponding names, frames and masks from all the labeled frames.
# Also rescales frames and masks to fixed size HxW.
# returns video_frames (N, H, W, 1), mask_frames (N, H, W, 1) and names (N,)
def preprocess_train_data(data, H=256, W=256):
    video_frames = []
    mask_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=np.bool)
        # for each annotated frame, extract the video frame and corresponding mask
        for frame_idx in item['frames']:
            mask[:, :, frame_idx] = item['label'][:, :, frame_idx]
            video_frame = video[:, :, frame_idx]
            mask_frame = mask[:, :, frame_idx]
                
            # rescale video to (H,W)
            video_frame = cv2.resize(video_frame, (H, W), interpolation=cv2.INTER_LINEAR)
            mask_frame = cv2.resize(mask_frame.astype(np.uint8), (H, W), interpolation=cv2.INTER_NEAREST)
            
            video_frame = np.expand_dims(video_frame, axis=2).astype(np.float32)
            mask_frame = np.expand_dims(mask_frame, axis=2).astype(np.int32)    
            
            video_frames.append(video_frame)
            mask_frames.append(mask_frame)
            names.append(name)
    return names, video_frames, mask_frames

def preprocess_test_data(data):
    video_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        video = video.astype(np.float32).transpose((2, 0, 1))
        video = np.expand_dims(video, axis=3)
        video_frames += list(video)
        names += [item['name'] for _ in video]
    return names, video_frames

# Split data into training and testing sets, but take only videos from expert labeled data into test set.
# Test ratio is the fraction of expert labeled data to be used as test set.
def make_train_test_split(data, test_ratio=0.2, random_seed=42):
    np.random.seed(random_seed)
    n_samples = len(data)
    expert_indices = [i for i, item in enumerate(data) if item['dataset'] == 'expert']
    n_test = int(len(expert_indices) * test_ratio)
    test_indices = np.random.choice(expert_indices, size=n_test, replace=False)
    train_indices = [i for i in range(n_samples) if i not in test_indices]
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    return train_data, test_data

def get_mask_from_image(image, model, H, W):
    device = next(model.parameters()).device
    # image has shape (H_orig, W_orig)
    H_orig, W_orig = image.shape
    
    image_resized = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
    
    model.eval()
    with torch.no_grad():
        image_tensor = torch.tensor(image_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Check logits before predict_mask
        logits = model(image_tensor)
        
        # logits shape: (B, 2, H, W)
        # Softmax über die Kanal-Dimension, dann argmax
        probs = torch.softmax(logits, dim=1)  # (B, 2, H, W) - Wahrscheinlichkeiten pro Klasse
        mask = torch.argmax(probs, dim=1)     # (B, H, W) - Werte 0 (background) oder 1 (foreground)
        
        mask_np = mask.squeeze().cpu().numpy().astype(np.uint8)
        
        mask_resized = cv2.resize(mask_np, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
    return mask_resized

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object