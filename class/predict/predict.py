#!usr/bin/env python
#-*- coding:utf-8 _*-
import torchvision.transforms as transforms
import torch
from PIL import Image
from collections import OrderedDict
import torch.nn.functional as F
from media_stom.build_net import make_model
from efficientnet_pytorch import EfficientNet
from torch import nn
import os, time
import torchvision.models as models
from media_stom.args import args
from media_stom.models.resnetxt_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
args = {}
args['arch'] = 'resnext101_32x16d_wsl'
args['pretrained'] = False
args['num_classes'] = 3
args['image_size'] = 728
'''

class classfication_service():
    def __init__(self, model_path):
        self.model = self.build_model(model_path)
        self.pre_img = self.preprocess_img()
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_id_name_dict = \
            {
                "0"
                "1"
                "2"
            }

    def build_model(self, model_path):
        #if args.arch == 'resnext101_32x16d_wsl':
        model = make_model(args)
        model = torch.nn.DataParallel(model).cuda()
        #if args['arch'] == 'resnext101_32x8d':
            #model = models.__dict__[args['arch']]()
        #elif args['arch'] == 'efficientnet-b7':
            #model = EfficientNet.from_name(args['arch'])

        layerName, layer = list(model.named_children())[-1]
        #exec("model." + layerName + "=nn.Linear(layer.in_features," + str(args['num_classes']) + ")")

        #if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            #model.features = torch.nn.DataParallel(model.features)
            #model.cuda()
        #else:
            #model = torch.nn.DataParallel(model).cuda()

        #if torch.cuda.is_available():
            #modelState = torch.load(model_path)
            #model.load_state_dict(modelState)
            #model = model.cuda()
        #else:
            #modelState = torch.load(model_path, map_location='cpu')
            #model.load_state_dict(modelState)
        return model

    def preprocess_img(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        infer_transformation = transforms.Compose([
            Resize((288, 288)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return infer_transformation

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                img = self.pre_img(img)
                preprocessed_data[k] = img
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img = data['input_img']
        img = img.unsqueeze(0)
        img = img.to(self.device)
        with torch.no_grad():
            pred_score = self.model(img)

        if pred_score is not None:
            _, pred_label = torch.max(pred_score.data, 1)
            result = {'result': self.label_id_name_dict[str(pred_label[0].item())]}
        else:
            result = {'result': 'predict score is None'}

        return result

    def _postprocess(self, data):
        return data


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img


if __name__ == '__main__':
    model_path = 'D:/pycharm_object/media_stom/media_stom/checkpoint/data0/res_16_288_last1/model_20_9880_10000.pth'
    infer = classfication_service(model_path)
    input_dir = 'D:/weiai_dataset/train/train_org_image/'
    files = os.listdir(input_dir)
    t1 = int(time.time()*1000)
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        img = Image.open(file_path)
        img = infer.pre_img(img)
        tt1 = int(time.time() * 1000)
        result = infer._inference({'input_img': img})
        tt2 = int(time.time() * 1000)
        print((tt2 - tt1) / 100)
    t2 = int(time.time()*1000)
    print((t2 - t1)/100)

