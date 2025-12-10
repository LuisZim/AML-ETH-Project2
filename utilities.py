import numpy as np
from tqdm import tqdm
# Extracts corresponding names, frames and masks from all the labeled frames.
def preprocess_train_data(data):
    video_frames = []
    mask_frames = []
    names = []
    for item in tqdm(data):
        video = item['video']
        name = item['name']
        height, width, n_frames = video.shape
        mask = np.zeros((height, width, n_frames), dtype=np.bool)
        # for each annotated frame, extract the video frame and corresponding mask
        for frame in item['frames']:
            mask[:, :, frame] = item['label'][:, :, frame]
            video_frame = video[:, :, frame]
            mask_frame = mask[:, :, frame]
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