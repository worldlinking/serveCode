from __future__ import print_function
import os
import torch
from torch.autograd import Variable
import numpy as np
from models.u_net import UNet
from datasets_tif import ImageDataset
from torch.utils.data import DataLoader
from osgeo import gdal
from sklearn.metrics import accuracy_score

image_path = r"image.txt"
label_path = r"label.txt"
datasets = np.loadtxt(image_path, dtype=str)
labelsets = np.loadtxt(label_path, dtype=str)
save_image = './checkpoint/test_images'
try:
    os.makedirs(save_image)
except OSError:
    pass

test_dataset = ImageDataset(datasets, labelsets)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)

def to_tif(arr, tran, projection, mapfile):
    """ 将数组保存成tif"""
    # 数组shape
    row = arr.shape[1]  # 行数
    columns = arr.shape[2]  # 列数
    dim = arr.shape[0]

    mat_mew = np.zeros((1, row, columns))

    mat = np.array(arr.cpu().detach())

    max_idx = np.argmax(mat, axis=0)
    mat_mew[0] = max_idx

    # 创建驱动
    driver = gdal.GetDriverByName('GTiff')
    # 创建文件
    dst_ds = driver.Create(mapfile, columns, row, 1, gdal.GDT_UInt16)
    # 设置几何信息
    dst_ds.SetGeoTransform(tran)
    dst_ds.SetProjection(projection)  # 设置投影

    for channel in np.arange(1):
        map = mat_mew[channel, :, :].astype('uint16')
        dst_ds.GetRasterBand(int(channel + 1)).WriteArray(map)

    # 写入硬盘
    #dst_ds.FlushCache()
    dst_ds = None

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## 输入4波段，输出7类
net = UNet(4, 7)
net = net.cuda()
model_path = './checkpoint/Unet_models/Unet_100.pth'
net.load_state_dict(torch.load(model_path))
# net.load_state_dict(torch.load(model_path, map_location='cpu'))

acc = 0
test_label = []
pred_test_y = []

for i, imgs in enumerate(test_dataloader):
    # Configure model input
    imgs_lr = Variable(imgs["data"].type(Tensor))
    im_geotrans = imgs["im_geotrans"]  # list
    im_proj = imgs["im_proj"]  # list
    imgs_label = imgs["label"].to(device)

    semantic_image_pred = net(imgs_lr)

    ### evaluate ###
    predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
    gts = imgs_label.data[:].squeeze_(0).cpu().numpy()  ##（h，w）
    acc_score = accuracy_score(gts.flatten(), predictions.flatten(), normalize=True)
    acc += acc_score
    print('[%d] acc_score: %.4f' % (i, acc_score))

    ##保存图像
    for j in range(len(semantic_image_pred)):
        image_name_ext_train = os.path.basename(imgs['path'][j])
        img_name_train, _ = os.path.splitext(image_name_ext_train)

        proj = "".join(im_proj[j])
        tran = []
        for a in im_geotrans:
            tran.append(a[j])

        save_img_path_trainH = os.path.join(save_image,
                                            'Unet_{:s}.tif'.format(img_name_train))
        to_tif(semantic_image_pred[j].cpu().detach(), tran, proj, save_img_path_trainH)

# print('acc:', acc)
print('acc_mean:', acc/len(test_dataloader))




