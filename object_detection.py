import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time

#model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or yolov3-spp, yolov3-tiny, custom
model_name = 'yolov5x'
model = torch.hub.load('ultralytics/yolov5', model_name)
model.cpu()

base_path = "C:\\#Projects\\Lanit\\SS2022_ITS\\Project2\\dataset_20220504_1445_cut\\17371610\\"
images_names = [f"{i:010d}.tiff" for i in range(451, 600)]


all_data = pd.DataFrame()

times = []
for image_name in tqdm(images_names):
    
    start = time()
    results = model(base_path + image_name)
    delta_time = time() - start
    times.append(delta_time)
    
    r = results.pandas()
    data = r.xyxy[0]
    columns = data.columns.values
    data = data[(data['name'] == 'car') | (data['name'] == 'bus') | (data['name'] == 'truck')].copy()
    data.xmin = data.xmin.astype(int)
    data.ymin = data.ymin.astype(int)
    data.xmax = data.xmax.astype(int)
    data.ymax = data.ymax.astype(int)
    data['frame'] = image_name
    all_data = pd.concat([all_data, data])

all_data = all_data.reset_index()
all_data.to_csv(f'data/{model_name}.csv')

print('mean time:', np.array(times).mean())