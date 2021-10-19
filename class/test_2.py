import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from media_stom.utils import accuracy, ProgressMeter, AverageMeter
#from media_stom.models.map_se_resnet import se_resnet32
#from media_stom.models.se_simam_resnext import resnext101
from media_stom.args import args
from media_stom.build_net import make_model
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import os
import pandas as pd

state = {k: v for k, v in args._get_kwargs()}
def test():
    model = make_model(args)
    model = torch.nn.DataParallel(model).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    checkpoint = torch.load('D:/pycharm_object/media_stom/media_stom/checkpoint/data0/res_16_288_last1/model_64_9994_9994.pth')
    ckpt = {'module.'+k: v for k, v in checkpoint.items()}
    model.load_state_dict(ckpt)
    #data_dir = 'D:/weiai_dataset/train/'
    #test_folder = os.path.join(data_dir, "train_org_image/")
    #test_folder = 'D:/weiai_dataset/test/test_org_image/'
    test_folder = 'D:/weiai_dataset/train/train_org_image/'
    print(test_folder)
    #train_folder = os.path.john(data_dir, "train")

    transform_test = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #test_set = datasets.ImageFolder(test_folder, transform_test)
    #test_loader = torch.utils.data.DataLoader(
        #test_set,
        #batch_size=6, shuffle=False)
    use_gpu = True
    validate(model, criterion, transform_test,use_gpu)

def validate(model, criterion, transform_test, use_gpu):
    # switch to evaluate mode
    model.eval()
    list = []
    with torch.no_grad():
        end = time.time()
        #for i, (images, target) in tqdm(enumerate(val_loader)):
        test_path = "D:/weiai_dataset/test/test_org_image/"
        #test_path = 'D:/weiai_dataset/train/train_org_image/'
        list_test_img = os.listdir(test_path)
        for i in range(len(list_test_img)):
            #print("准备到========>",list_test_img[i])
            images = Image.open(test_path+list_test_img[i]).convert('RGB')
            images = transform_test(images)
            images = images.unsqueeze(0)
            if use_gpu:
                images = images.cuda(non_blocking=True)
                #target = target.cuda(non_blocking=True)
            # compute output
            output = model(images)
            _, predicted = torch.max(output, 1)
            a = predicted.cpu().numpy().tolist()
            print(list_test_img[i] +"的结果是=====>"+ str(a))
            for i in a:
                list.append(i)
        list_1 = list
        #data = "D:/weiai_dataset/train/train_org_image/train_org_image/"
        df_up = pd.DataFrame({'image_name': list_test_img,
                              'label': list_1})
        df_up.to_csv('result.csv', encoding='utf_8_sig', index=None)
        #print(df_up)


if __name__ == '__main__':
    test()
