import os
import time
import copy
import torch
import torch.optim as optim

from PIL import Image
from torch.optim.swa_utils import AveragedModel,SWALR
from torch.autograd import Variable
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from pytorch_toolbelt import losses as L
#from utils import lossesjoin as L
from utils.util import AverageMeter, inial_logger
from torchcontrib.optim import SWA
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from segmentation_models_pytorch.segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss,SoftBCEWithLogitsLoss
Image.MAX_IMAGE_PIXELS = 1000000000000000
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def smooth(v, w=0.85):
    last = v[0]
    smoothed = []
    for point in v:
        smoothed_val = last * w + (1 - w) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed



# training loop

def train_net(param, model, train_data, valid_data, plot=False,device='cuda'):
    # 初始化参数
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    lr              = param['lr']
    gamma           = param['gamma']
    step_size       = param['step_size']
    momentum        = param['momentum']
    weight_decay    = param['weight_decay']

    disp_inter      = param['disp_inter']
    save_inter      = param['save_inter']
    min_inter       = param['min_inter']
    iter_inter      = param['iter_inter']

    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']

    #
    scaler = GradScaler() 

    # 网络参数
    train_data_size = train_data.__len__()
    valid_data_size = valid_data.__len__()
    c, y, x = train_data.__getitem__(0)['image'].shape
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4 ,weight_decay=weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    #SWA
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=momentum, weight_decay=weight_decay)
    #scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    swa_model = AveragedModel(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-5, last_epoch=-1)
    swa_start = 0
    swa_scheduler = SWALR(optimizer,swa_lr=1e-5)
    DiceLoss_fn=DiceLoss(mode='binary')
    SoftBCEWithLogitsLoss_fn = SoftBCEWithLogitsLoss(smooth_factor=0.1)
    criterion = L.JointLoss(first=SoftBCEWithLogitsLoss_fn, second=DiceLoss_fn,
                              first_weight=0.4, second_weight=0.6).cuda()
    logger = inial_logger(os.path.join(save_log_dir, time.strftime('_'+model_name+ '.log')))

    # 主循环
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    train_loader_size = train_loader.__len__()
    valid_loader_size = valid_loader.__len__()
    best_acc = 0
    best_epoch=0
    best_mode = copy.deepcopy(model)
    epoch_start = 0
    if load_ckpt_dir is not None:
        ckpt = torch.load(load_ckpt_dir)
        epoch_start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info('Total Epoch:{} Image_size:({}, {}) Training num:{}  Validation num:{}'.format(epochs, x, y, train_data_size, valid_data_size))
    #
    for epoch in range(epoch_start, epochs):
        epoch_start = time.time()
        # 训练阶段
        model.train()
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx, batch_samples in enumerate(train_loader):
            data, target = batch_samples['image'], batch_samples['label']
            data, target = Variable(data.to(device)), Variable(target.to(device))
            #data = data.unsqueeze(0)
            with autocast(): #need pytorch>1.6
                pred = model(data)
                pred = pred.squeeze()
                optimizer.zero_grad()
                loss = criterion(pred, target.float())
                scaler.scale(loss).backward()
            if epoch > swa_start:
                #print("start swa")
                swa_model.update_parameters(model)
                swa_scheduler.step()
                #scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
            #optimizer.swap_swa_sgd()
            scheduler.step(epoch + batch_idx / train_loader_size) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, train_loader_size, batch_idx/train_loader_size*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * train_loader_size // 60 - spend_time // 60))
                train_iter_loss.reset()

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        #iou=IOUMetric(1)
        acc_list = []
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(valid_loader):
                data, target = batch_samples['image'], batch_samples['label']
                data, target = Variable(data.to(device)), Variable(target.to(device))
                pred = model(data)
                pred = pred.squeeze()
                loss = criterion(pred,target.float())
                #pred= pred.squeeze().cpu().data.numpy()
                #target = target.cpu().data.numpy()
                pred = pred > 0.7
                #以IoU为分类指标的框架
                #pred= np.argmax(pred,axis=1)
                #iou.add_batch(pred,target.cpu().data.numpy())
                #以acc为指标的二分类框架
                TP = (pred[target == 1] == 1).sum().type(torch.cuda.FloatTensor).item()
                TN = (pred[target == 0] == 0).sum().type(torch.cuda.FloatTensor).item()

                #调节验证集的尺寸
                acc = (TP+TN) / (2*data.shape[2]*data.shape[3])

                acc_list.append(acc)
                #
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
                val_loss = valid_iter_loss.avg
                if batch_idx % iter_inter == 0:
                     logger.info('[val] epoch:{} iter:{}/{} {:.2f}% loss:{:.6f}'.format(
                        epoch, batch_idx, valid_loader_size, batch_idx / valid_loader_size * 100, val_loss))
            #acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            acc = sum(acc_list)/len(acc_list)
            logger.info('[val] epoch:{} acc:{:.2f} '.format(epoch,acc))
                

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        if epoch % save_inter == 0 and epoch > min_inter:
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型 #修改为对准acc版本
        if acc > best_acc:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_acc = acc
            best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        #scheduler.step()
        # 显示loss
    # 训练loss曲线
    if plot:
        x = [i for i in range(epochs)]
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.plot(x, smooth(train_loss_total_epochs, 0.6), label='train loss')
        ax.plot(x, smooth(valid_loss_total_epochs, 0.6), label='val loss')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('CrossEntropy', fontsize=15)
        ax.set_title('train curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        ax = fig.add_subplot(1, 2, 2)
        ax.plot(x, epoch_lr,  label='Learning Rate')
        ax.set_xlabel('Epoch', fontsize=15)
        ax.set_ylabel('Learning Rate', fontsize=15)
        ax.set_title('lr curve', fontsize=15)
        ax.grid(True)
        plt.legend(loc='upper right', fontsize=15)
        plt.show()
            
    return best_mode, model
#
