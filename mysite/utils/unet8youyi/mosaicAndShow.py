'''
修改inputDir和destFile即可
'''
from osgeo import gdal, gdalconst, osr
import matplotlib.pyplot as plt
import argparse
import os
import sys
os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='raster files mosaic', description='对输入目录中的栅格影像进行拼接，要求这些影像具有相同的空间参考')
    parser.add_argument('--inputDir', type=str,
                        default=r'D:\project\xianyu\cy\unet8youyi\checkpoints\test_images',
                        help='待拼接影像所在的栅格目录')
    parser.add_argument('--destFile', type=str,
                        default=r'D:\project\xianyu\cy\unet8youyi\results.tif',
                        help='拼接影像输出路径')
    parser.add_argument('--img_path', type=str,
                        default=r'D:\project\xianyu\cy\unet8youyi\results.png',
                        help='拼接影像输出路径')
    parser.add_argument('--resampleAlg', type=int, default=0,
                        help='重采样方法：\n{}\n{}\n{}'.format('0:NearestNeighbour', '1:Bilinear', '2:Cubic'))
    parser.add_argument('--cutlineSHP', type=str, default='',
                        required=False, help='一个SHP文件用于裁剪拼接后的栅格影像')
    args = parser.parse_args()
    inputDir = args.inputDir
    destFile = args.destFile

    resampleAlg = args.resampleAlg
    cutlineSHP = args.cutlineSHP
    tifs = os.listdir(inputDir)

    tifs = list(filter(lambda fileName: fileName.endswith('.tif'), tifs))
    if len(tifs) == 0:
        print('栅格目录为空！')
        exit(0)
    tifs = [os.path.join(inputDir, tif) for tif in tifs]

    gdal.AllRegister()
    osrs = []
    for tif in tifs:
        ds = gdal.Open(tif, gdalconst.GA_ReadOnly)
        # osr_ = gdal.Dataset.GetSpatialRef(ds)
        osr_ = ds.GetProjection()
        osrs.append(osr_)
    osr_ = osrs[0]
    print(osr_)
    # for osri in osrs:
    #     flag = osr.SpatialReference.IsSame(osr_, osri)
    #     if not(flag):
    #         print('待拼接的栅格影像必须有相同的空间参考！')
    #         exit(0)
    if resampleAlg == 0:
        resampleType = gdalconst.GRA_NearestNeighbour
    elif resampleAlg == 1:
        resampleType = gdalconst.GRA_Bilinear
    else:
        resampleType = gdalconst.GRA_Cubic
    if cutlineSHP:
        options = gdal.WarpOptions(
            srcSRS=osr_, dstSRS=osr_, format='GTiff', resampleAlg=resampleType, creationOptions=["COMPRESS=LZW"], cutlineDSName=cutlineSHP, cropToCutline=True)
    else:
        options = gdal.WarpOptions(
            srcSRS=osr_, dstSRS=osr_, format='GTiff', resampleAlg=resampleType, creationOptions=["COMPRESS=LZW"])
    gdal.Warp(destFile, tifs, options=options)
    print('Successfully ！')

    ## show
    img = args.destFile
    img = gdal.Open(img)
    img = img.ReadAsArray()
    print(img.shape)
    plt.axis('off')  # 去坐标轴
    plt.xticks([])  # 去刻度
    plt.yticks([])  # 去刻度
    plt.imshow(img)
    plt.savefig(args.img_path,bbox_inches='tight',pad_inches = -0.01)
    # plt.gray()        ## 显示灰度图
    exit(0)

