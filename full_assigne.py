import os
import cv2
import numpy as np
import pandas as pd
from collections import Counter

###################################################

def in_yolo_area(x, y, df):
    return ((x > df['xmin']) & (x < df['xmax']) & (y > df['ymin']) & (y < df['ymax'])).any()

def in_yolo_area_index(x, y, df):
    return df[((x > df['xmin']) & (x < df['xmax']) & (y > df['ymin']) & (y < df['ymax']))]
        
def sift_keypoints_search_space(image, df):
    sift = cv2.SIFT_create()
    kp = sift.detect(image, None)
    p0 = np.array(list(map(lambda p: np.round(p.pt).astype("float32"), kp)))
    p0 = np.array(list(filter(lambda point: in_yolo_area(point[0], point[1], df), p0)))
    return p0

def fast_keypoints_search_space(image, df):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(image, None)
    p0 = np.array(list(map(lambda p: np.round(p.pt).astype("float32"), kp)))
    p0 = np.array(list(filter(lambda point: in_yolo_area(point[0], point[1], df), p0)))
    return p0

def good_keypoints_search_space(image, df):
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray, mask = None, **feature_params).reshape(-1, 2)
    p0 = np.array(list(filter(lambda point: in_yolo_area(point[0], point[1], df), p0)))
    return p0

def orb_keypoints_search_space(image, df):
    orb = cv2.ORB_create()
    kp = orb.detect(image, None)
    p0 = np.array(list(map(lambda p: np.round(p.pt).astype("float32"), kp)))
    p0 = np.array(list(filter(lambda point: in_yolo_area(point[0], point[1], df), p0)))
    return p0

def assigne(points1, points2, df1, df2):
    possible = {}
    assigne_sets = list(map(lambda point: set(in_yolo_area_index(point[0], point[1], df1).index), points1))
    for index, row in df2.iterrows():
        for i in range(len(points2)):
            x, y = points2[i]
            assigne = assigne_sets[i]
            if x >= row.xmin and x <= row.xmax and y >= row.ymin and y <= row.ymax:
                possible[index] = possible.get(index, []) + list(assigne)
    emptys = list(filter(lambda key: possible[key] == [], possible))
    return {key: Counter(value).most_common(1)[0][0] for key, value in possible.items() if key not in emptys}

###################################################

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (15000, 3))

###################################################

df = pd.read_csv('data/yolov5s.csv')
base_path = "C:\\#Projects\\Lanit\\SS2022_ITS\\Project2\\dataset_20220504_1445_cut\\17371610\\"
search_space = sift_keypoints_search_space

#images_names = sorted(os.listdir(base_path))
#print(images_names)
###################################################

from time import time
times = []
with open('sift_output.txt', 'w') as file:
    file.write('')

for i in range(451, 601):
    start_time = time()
    print(i, '/', 500)
    image_name1, image_name2 = f'{i:010}.tiff', f'{i+1:010}.tiff'

    selected_df1 = df[df['frame'] == image_name1]
    selected_df2 = df[df['frame'] == image_name2]

    image1 = cv2.imread(base_path + image_name1)
    image2 = cv2.imread(base_path + image_name2)

    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    p0 = search_space(image1, selected_df1)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray_image1, gray_image2, p0, None, **lk_params)
    st = st.reshape(-1)
    points0 = p1[st==1]
    points1 = p0[st==1]

    with open('sift_output.txt', 'a') as file:
        for key, value in assigne(points0, points1, selected_df1, selected_df2).items():
            file.write(str(key) + ' ' + str(value) + '\n')
    
    times.append(time() - start_time)
    print('elapsed time', times[-1])
    print()
print('End: ', np.array(times).mean(), 's')