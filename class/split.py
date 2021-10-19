import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
df=pd.read_csv("./data/train_label.csv")
data_path = "./data/train_org_image/"
data = "./data/media/train/"
for i in tqdm(range(len(df['image_name']))):
    if df['label'][i] == 1:
        img = cv2.imread(data_path+df['image_name'][i])
        cv2.imwrite(data+"1/"+df['image_name'][i],img)
    if df['label'][i] == 2:
        img = cv2.imread(data_path+df['image_name'][i])
        cv2.imwrite(data+"2/"+df['image_name'][i],img)
    if df['label'][i] == 3:
        img = cv2.imread(data_path+df['image_name'][i])
        cv2.imwrite(data+"3/"+df['image_name'][i],img)
    if df['label'][i] == 0:
        img = cv2.imread(data_path+df['image_name'][i])
        cv2.imwrite(data+"0/"+df['image_name'][i],img)