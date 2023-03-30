# Name: ExtractByMask_Ex_02.py
# Description: Extracts the cells of a raster that correspond with the areas
#    defined by a mask.
# Requirements: Spatial Analyst Extension
# Import system modules
import arcpy
from arcpy import env
from arcpy.sa import *
from glob import glob
import os

files = glob(r'E:\unet\split_label\*')
savepath = r'E:\unet\split_data_9'

flag = 0
for file in files:
    flag += 1
    print('file:', file)
    path = file.split('\\')[3].split('_')[4].split('.')[0]
    print('path:', path)
    name = os.path.join(savepath, 'img224_' + path + '.tif')
    print('name:', name)

    ## Set environment settings
    env.workspace = r"E:\unet1"

    # Set local variables
    inRaster = r'9youyi-light.tif'

    # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # Execute ExtractByMask
    outExtractByMask = ExtractByMask(inRaster, file)

    # Save the output
    outExtractByMask.save(name)

print(flag)
