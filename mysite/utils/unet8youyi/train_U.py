from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time
from numpy import *
from models.u_net import UNet
import numpy as np
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from datasets_tif import ImageDataset
from torch.utils.data import DataLoader
import datetime
from evaluation import SegmentationMetric
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Training a UNet model')
parser.add_argument('--batch_size', type=int, default=8, help='equivalent to instance normalization with batch_size=1')
parser.add_argument('--input_nc', type=int, default=4)
parser.add_argument('--output_nc', type=int, default=7, help='equivalent to numclass')
parser.add_argument('--niter', type=int, default=101, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
# parser.add_argument('--cuda', type=bool, default=False, help='enables cuda. default=True')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda. default=True')
parser.add_argument('--manual_seed', type=int, help='manual seed')
parser.add_argument('--num_workers', type=int, default=0, help='how many threads of cpu to use while loading data')
parser.add_argument('--size_w', type=int, default=256, help='scale image to this size')
parser.add_argument('--size_h', type=int, default=256, help='scale image to this size')
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')

parser.add_argument("--image", default=r"image.txt", help="train label")
parser.add_argument("--label", default=r"label.txt", help="train label")
parser.add_argument('--out_models', default='./checkpoint/Unet_models', help='folder to output model')

parser.add_argument('--save_epoch', default=20, help='path to save model')
parser.add_argument('--test_step', default=20, help='path to val images')
parser.add_argument('--log_step', default=20, help='path to val images')
parser.add_argument('--num_GPU', default=1, help='number of GPU')
opt = parser.parse_args()

try:
    os.makedirs(opt.out_models)
except OSError:
    pass

if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
cudnn.benchmark = True

## 随机划分数据集
datasets = np.loadtxt(opt.image, dtype=str)
labelsets = np.loadtxt(opt.label, dtype=str)
X_train, X_test, y_train, y_test = train_test_split(datasets, labelsets, test_size=0.3, random_state=0)
train_dataset = ImageDataset(X_train, y_train)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
print("The length of train set is: {}\n".format(len(train_dataloader)*opt.batch_size))

## 添加验证集
val_dataset = ImageDataset(X_test, y_test)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)
print("The length of VAL set is: {}\n".format(len(val_dataloader)))

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

net = UNet(opt.input_nc, opt.output_nc)

if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init)
if opt.cuda:
    net.cuda()
if opt.num_GPU > 1:
    net = nn.DataParallel(net)

###########   LOSS & OPTIMIZER   ##########
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net = net.to(device)

if opt.cuda:
    criterion = criterion.cuda()

if __name__ == '__main__':

    results_file = "./checkpoint/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    log = open('./checkpoint/train_Unet_log.txt', 'w')
    start = time.time()
    net.train()


    for epoch in range(0, opt.niter):
        loader = iter(train_dataloader)
        loss_sum = 0.0

        for i, imgs in tqdm(enumerate(train_dataloader)):
            # Configure model input
            imgs_lr = Variable(imgs["data"].type(Tensor))
            im_geotrans = imgs["im_geotrans"]  # list
            im_proj = imgs["im_proj"]  # list
            imgs_label = imgs["label"].to(device)

            semantic_image_pred = net(imgs_lr)      # [B, output_nc, 128, 128]      [B, 1, 128, 128] #CHANGED

            loss = criterion(semantic_image_pred, imgs_label.long())
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ########### Logging ##########
            if i % opt.log_step == 0:
                print('epoch [%d/%d][%d/%d] Loss: %.4f' %
                      (epoch, opt.niter, i, len(train_dataloader), loss.item()))
                log.write('[%d/%d][%d/%d] Loss: %.4f\n' %
                          (epoch, opt.niter, i, len(train_dataloader), loss.item()))

        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), os.path.join(opt.out_models, "Unet_%d.pth" % epoch))

        ## 验证集
        if epoch % opt.test_step == 0:
            print('##############################val##################################')
            pa_sum = 0.0
            mpa_sum = 0.0
            mIoU_sum = 0.0
            FWIoU_sum = 0.0
            cpa_sum = np.zeros((1, opt.output_nc))
            IoU_sum = np.zeros((1, opt.output_nc))

            for i, imgs in tqdm(enumerate(val_dataloader)):
                # Configure model input
                imgs_lr = Variable(imgs["data"].type(Tensor))
                im_geotrans = imgs["im_geotrans"]  # list
                im_proj = imgs["im_proj"]  # list
                imgs_label = imgs["label"].to(device)

                semantic_image_pred = net(imgs_lr)       # [B, output_nc, 128, 128]      [B, 1, 128, 128] #CHANGED

                pre = semantic_image_pred.argmax(1)       ## [B, output_nc, 128, 128] → [B, 128, 128]

                ## 指标
                metric = SegmentationMetric(opt.output_nc)
                metric.addBatch(imgs_label.cpu(), pre.cpu())
                pa = metric.pixelAccuracy()
                cpa = metric.classPixelAccuracy()
                mpa = metric.meanPixelAccuracy()
                IoU = metric.intersectionOverUnion()
                mIoU = metric.meanIntersectionOverUnion()
                FWIoU = metric.frequencyWeightedIntersectionOverUnion()

                pa_sum += pa
                mpa_sum += mpa
                mIoU_sum += mIoU
                FWIoU_sum += FWIoU
                cpa_sum += cpa
                IoU_sum += IoU

            with open(results_file, 'a') as f:
                # 记录每个epoch对应的train_loss、lr以及验证集各指标
                #pa 像素准确率 cpa 类别像素准确率  mpa 类别平均像素准确率 mIou 平均交并比

                write_info = f"[epoch: {epoch}] " \
                             f"pa: {pa_sum/len(val_dataloader):.4f}\n " \
                             f"mpa: {mpa_sum/len(val_dataloader):.4f}\n" \
                             f"mIoU: {mIoU_sum/len(val_dataloader):.4f}\n " \
                             f"FWIoU: {FWIoU_sum/len(val_dataloader):.4f}\n " \
                             f"*********************************************\n"
                f.write(write_info)

            print('[%d/%d]  pa: %.4f  mIoU:% .4f' %
                  (epoch, opt.niter, pa_sum/len(val_dataloader), mIoU_sum/len(val_dataloader)))

    end = time.time()
    print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')
    log.close()
