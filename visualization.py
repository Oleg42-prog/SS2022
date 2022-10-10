import os
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv('data/yolov3.csv')

base_path = "C:\\#Projects\\Lanit\\SS2022_ITS\\Project2\\dataset_20220504_1445_cut\\17371610\\"
images_names = sorted(os.listdir(base_path))[:10]
#images_paths = list(map(lambda s: base_path + s, images_names))
#print('images_names:', images_names)
#print('images_paths:', images_paths)

def in_yolo_area(x, y, df):
    return ((x > df['xmin']) & (x < df['xmax']) & (y > df['ymin']) & (y < df['ymax'])).any()

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (15000, 3))

image = cv2.imread(base_path + images_names[0])
sift = cv2.SIFT_create()
kp = sift.detect(image, None)
p0 = np.array(list(map(lambda p: np.round(p.pt).astype("float32"), kp)))
for x, y in p0:
    image = cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), 2)

for index, row in df[df['frame'] == images_names[0]].iterrows():
    image = cv2.rectangle(image, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255, 0, 0), 2)

for x, y in p0:
    if in_yolo_area(x, y, df[df['frame'] == images_names[0]]):
        image = cv2.circle(image, (int(x), int(y)), 1, (255, 0, 255), 2)

while True:
    cv2.imshow('', image)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()