from osgeo import gdal


def mask_crop(resourcee_file, shpFile, destination_filename):
    # shp = r"D:\Data\Landsat8\MaskCrop\Shanghai-city-2020\Shanghai-city-2020.shp"  # 圈选范围的路径
    # inputImage = r"D:\Data\Landsat8\lc81180382020053lgn00\LC08_L1TP_118038_20200222_20200225_01_T1_B1.TIF"  # 遥感影像的路径
    # dataset = gdal.Open(inputImage)  # 打开遥感影像
    # outputImage = r"D:\Data\Landsat8\MaskCrop\B1_image_mask.tif"  # 按照圈选范围提取出的影像所存放的路径

    shp = shpFile  # 圈选范围的路径
    inputImage = resourcee_file  # 遥感影像的路径
    dataset = gdal.Open(inputImage)  # 打开遥感影像
    outputImage = destination_filename  # 按照圈选范围提取出的影像所存放的路径
    print('图像裁剪...')
    gdal.Warp(outputImage, dataset, cutlineDSName=shp, cropToCutline=True)  # 按掩膜提取
    # del shp, inputImage, dataset, outputImage  # 释放内存
    # gc.collect()
