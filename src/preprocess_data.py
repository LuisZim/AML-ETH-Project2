import numpy as np
import cv2 as cv
import gzip
import pickle

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

if __name__ == "__main__":
    # Example usage
    data = load_zipped_pickle('./data/raw/train.pkl')
    sample = data[55]

    vid = sample.get('video')
    idx = 5

    frames = [vid[..., i] for i in range(vid.shape[-1])]
    
    temporal_window = min(5, len(frames))  # Ungerade Zahl, z.B. 5 oder 3
    if temporal_window % 2 == 0:
        temporal_window -= 1  # Muss ungerade sein
    
    dst = cv.fastNlMeansDenoisingMulti(frames, idx, temporal_window, h=10, templateWindowSize=7, searchWindowSize=21)
    
    img = frames[idx]
    cv.imshow('denoised', dst)
    cv.imshow('original', img)
    cv.waitKey(0)
    cv.destroyAllWindows()