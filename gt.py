import os
import cv2
import numpy as np

def load_mask(path, color=[255, 0, 0]):
    mask = cv2.imread(path)
    mask2d = mask.sum(axis=2)
    mask[mask2d != 0] = np.array(color)
    return mask

def mask_geometry(mask):
    indices = np.indices((2048, 1536)).transpose()
    selected_indices = indices[mask.sum(axis=2) != 0]
    geometry = {
        'x_min': int(selected_indices[:, 0].min()),
        'x_mean': int(selected_indices[:, 0].mean()),
        'x_max': int(selected_indices[:, 0].max()),
        'y_min': int(selected_indices[:, 1].min()),
        'y_mean': int(selected_indices[:, 1].mean()),
        'y_max': int(selected_indices[:, 1].max())
    }
    geometry['width'] = geometry['x_max'] - geometry['x_min']
    geometry['height'] = geometry['y_max'] - geometry['y_min']
    
    return geometry

track_names = ['track' + str(i) for i in range(1, 12)]

from tqdm import tqdm
with open('gt.txt', 'w') as file:
    for track_id in range(1, 12):
        print(track_id)
        for frame_id in tqdm(range(451, 502)):
            if(os.path.exists(f'tracks/track{track_id}/masks/0000000{frame_id}_mask.tiff.png')):
                gmask = mask_geometry(load_mask(f'tracks/track{track_id}/masks/0000000{frame_id}_mask.tiff.png'))
                row_data = [frame_id - 450, track_id, gmask['x_min'], gmask['y_min'], gmask['width'], gmask['height'], -1, -1, -1, -1]
                row = ', '.join(map(str, row_data))
                file.write(row + '\n')