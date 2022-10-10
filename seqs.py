import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from geometry import distance, Geometry, MaskGeometry

def add_track(img, mask, title, color=(255, 0, 0)):
    img = img.copy()
    img = (img + mask * 0.25).astype(int)
    indices = np.indices((2048, 1536)).transpose()
    selected_indices = indices[mask.sum(axis=2) != 0]
    img = cv2.rectangle(img, selected_indices.min(axis=0), selected_indices.max(axis=0), color, 2)
    img = cv2.putText(img, title, (selected_indices[:, 0].min(), selected_indices[:, 1].min() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return img

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

mask_base_path = 'C:\\#Projects\\Lanit\\TRACKS\\'

def intersection(gmask, row):
    xmin = max(gmask['x_min'], row['xmin'])
    xmax = min(gmask['x_max'], row['xmax'])
    ymin = max(gmask['y_min'], row['ymin'])
    ymax = min(gmask['y_max'], row['xmax'])
    if xmax > xmin and ymax > ymin:
        return (xmax - xmin) * (ymax - ymin)
    return 0

def union(gmask, row):
    s1 = (gmask['x_max'] - gmask['x_min']) * (gmask['y_max'] - gmask['y_min'])
    s2 = (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])
    return s1 * s2 - intersection(gmask, row)

for model_name in tqdm(['yolov3', 'yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']):
    df = pd.read_csv(f'{model_name}.csv')
    for track_id in range(1, 5):
        answer = pd.DataFrame()
        with open(f'gt/gt_{track_id}.txt', 'w') as file:
            for frame_id in range(451, 600):
                if(os.path.exists(mask_base_path + f'track{track_id}/masks/0000000{frame_id}_mask.tiff.png')):
                    gmask = mask_geometry(load_mask(mask_base_path + f'track{track_id}/masks/0000000{frame_id}_mask.tiff.png'))
                    row_data = [frame_id - 450, track_id, gmask['x_min'], gmask['y_min'], gmask['width'], gmask['height'], -1, -1, -1, -1]
                    row = ', '.join(map(str, row_data))
                    file.write(row + '\n')

                    sdf = df[df['frame'] == f'0000000{frame_id}.tiff'].copy()
                    sdf['m'] = sdf.apply(lambda row: intersection(gmask, row), axis=1)
                    track = sdf[(sdf['m'] == sdf['m'].max()) & sdf['m'] != 0]
                    answer = pd.concat([answer, track])
        answer['frame_id'] = answer['frame'].apply(lambda s: int(s.split('.')[0]))
        with open(f'seqs/{model_name}_{track_id}.txt', 'w') as file:
            for index, row in answer.iterrows():
                width = row['xmax'] - row['xmin']
                height = row['ymax'] - row['ymin']
                file.write(', '.join(map(str, [row['frame_id'] - 450, track_id, row['xmin'], row['ymin'], width, height, -1, -1, -1, -1])))
                file.write('\n')