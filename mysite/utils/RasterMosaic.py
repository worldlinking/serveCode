from osgeo import gdal
from osgeo import gdalconst


def RasterMosaic(inputfilePath, referencefilefilePath, outputfilePath):
    print("图像匀色拼接...")

    inputrasfile1 = gdal.Open(inputfilePath, gdal.GA_ReadOnly)  # 第一幅影像
    inputProj1 = inputrasfile1.GetProjection()
    inputrasfile2 = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)  # 第二幅影像
    inputProj2 = inputrasfile2.GetProjection()

    options = gdal.WarpOptions(
        srcSRS=inputProj1, dstSRS=inputProj1,
        format='GTiff',
        resampleAlg=gdalconst.GRA_Bilinear)
    gdal.Warp(outputfilePath, [inputrasfile1, inputrasfile2], options=options, )  # gdal.Warp自动进行匀色处理
