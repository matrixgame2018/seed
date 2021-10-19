import cv2
import torch.nn as nn
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from tqdm import tqdm
import torch
from PIL import Image
train_transformA = A.Compose([
    A.Resize(1536, 1536),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
train_transform = A.Compose([
    A.Resize(768, 768),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
list = os.listdir("D:/stom_media/data/train/train_org_image/")
#
for i in tqdm(list):
    img = cv2.imread("D:/stom_media/data/train/train_org_image/"+i,cv2.IMREAD_COLOR)
    #A,B图像互补增强模块
    transformedA = train_transformA(image=img, mask=img)
    transform = train_transform(image=img,mask=img)
    img = transformedA['image']
    imgB = transform['image']
    img = img.reshape(1,3,1536,1536)
    pool = nn.MaxPool2d(2,2)
    e_lambda = 1e-4
    activaton = nn.Sigmoid()
    e_lambda = e_lambda
    b, c, h, w = img.size()
    n = w * h - 1
    x_minus_mu_square = (img - img.mean(dim=[2, 3], keepdim=True)).pow(2)
    y = activaton(x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + e_lambda)) + 0.5)
    img = pool(img * y)
    imgC = img  + imgB





