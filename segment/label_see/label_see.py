import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2


def create_pascal_label_colormap():
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap

    返回:
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    """
    输入图片和分割 mask 的可视化.
    """
    plt.figure(figsize=(25, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

LABEL_NAMES = np.asarray(['background', 'stick']) # 假设只有两类
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

imgfile = 'D:/stom_media/data/train/train_org_image/0aGT6gN7.png'
pngfile = 'D:/stom_media/data/train/train_mask/0aGT6gN7.png'
img = cv2.imread(imgfile, 1)
img = img[:,:,::-1]
seg_map = cv2.imread(pngfile, -1)
seg_map = seg_map//255
vis_segmentation(img, seg_map)

print('Done.')
#训练集mask的路径
path = "D:/stom_media/data/train/train_mask/"
#训练集原图的路径
path_img = "D:/stom_media/data/train/train_org_image/"
list = os.listdir(path)
for i in list:
    imgfile = path_img+i
    pngfile = path + i
    img = cv2.imread(imgfile, 1)
    img = img[:, :, ::-1]
    seg_map = cv2.imread(pngfile, -1)
    seg_map = seg_map // 255
    vis_segmentation(img, seg_map)
