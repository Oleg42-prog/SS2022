import os
import cv2
import numpy as np
import pandas as pd

###################################################

def in_yolo_area(x, y, df):
    return ((x > df['xmin']) & (x < df['xmax']) & (y > df['ymin']) & (y < df['ymax'])).any()

def sift_keypoints_search_space(image, df):
    sift = cv2.SIFT_create()
    kp = sift.detect(image, None)
    p0 = np.array(list(map(lambda p: np.round(p.pt).astype("float32"), kp)))
    p0 = np.array(list(filter(lambda point: in_yolo_area(point[0], point[1], df), p0)))
    return p0

###################################################

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (15000, 3))

###################################################

df = pd.read_csv('data/yolov5s.csv')
base_path = "C:\\#Projects\\Lanit\\SS2022_ITS\\Project2\\dataset_20220504_1445_cut\\17371610\\"
images_names = [f'{i:010}.tiff' for i in range(451, 600)]
selected_df = df[df['frame'] == images_names[0]]

###################################################

image = cv2.imread(base_path + images_names[0])
image2 = cv2.imread(base_path + images_names[1])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

###################################################

for index, row in selected_df.iterrows():
    image = cv2.rectangle(image, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (255, 0, 0), 2)

#for x, y in sift_keypoints_search_space(image, selected_df):
#    image = cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), 2)
print(images_names[0])
###################################################
p0 = sift_keypoints_search_space(image, selected_df)
p1, st, err = cv2.calcOpticalFlowPyrLK(gray_image, gray_image2, p0, None, **lk_params)
st = st.reshape(-1)
print(p0.shape, p1.shape, st.shape)
print((st==1).shape)
good_new = p1[st==1]
good_old = p0[st==1]

for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    image2 = cv2.line(image2, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 2)
    image2 = cv2.circle(image2, (int(a), int(b)), 4, (0, 0, 255), -1)
    #image2 = cv2.circle(image2, (int(c), int(d)), 2, (255, 0, 0), -1)

scale_percent = 100
width = int(image2.shape[1] * scale_percent / 100)
height = int(image2.shape[0] * scale_percent / 100)
dim = (width, height)
image2 = cv2.resize(image2, dim, interpolation = cv2.INTER_AREA)
while True:
    cv2.imshow('', image2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cv2.destroyAllWindows()