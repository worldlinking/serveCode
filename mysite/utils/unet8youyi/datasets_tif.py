from osgeo import gdal
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 读取图像
def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geotrans = dataset.GetGeoTransform()  # 仿射矩阵  tuple
    proj = dataset.GetProjection()  # 地图投影信息     str
    data = dataset.ReadAsArray(0, 0, width, height)
    if (len(data.shape) == 3):
        data = data.transpose((1, 2, 0))
    return data, geotrans, proj, width, height


# 构建dataset
class ImageDataset(Dataset):
    def __init__(self, image_A, label):
        self.image_A_paths = image_A  # 保存的是所有图片的路径及名字
        self.label_paths = label
        self.len = len(self.image_A_paths)
        assert len(self.image_A_paths) == len(self.label_paths), '两种影像数量不匹配'

        self.transform = A.Compose([
            A.Normalize(mean=[4.39, 6.04, 7.14, 6.69],  ##全局均值和标准差
                        std=[1.11, 1.29, 1.78, 2.46]),
            ToTensorV2()]
        )

    def __getitem__(self, index):
        path = self.image_A_paths[index % self.len]          ## 名字

        imageA, imageA_geotrans, imageA_proj, width, height = imgread(self.image_A_paths[index % self.len])
        label, label_geotrans, label_proj, width, height = imgread(self.label_paths[index % self.len])

        ## 256
        if imageA.shape[0] != 256 or imageA.shape[1] != 256:
            imageA = np.resize(imageA, (256, 256, imageA.shape[2]))
        if label.shape[1] != 256 or label.shape[0] != 256:
            label = np.resize(label, (256, 256))

        transformed_data = self.transform(image=imageA)
        imageA = transformed_data['image']

        return {"data": imageA, "im_geotrans": imageA_geotrans, "im_proj": imageA_proj,
                "label": label, "label_geotrans": label_geotrans, "im_proj_label": label_proj,
                "path": path, "width": width, "height": height}

    def __len__(self):
        return self.len

