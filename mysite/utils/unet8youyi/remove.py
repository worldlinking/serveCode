'''
移除：裁剪后标签图均为1类(背景类)
'''
from osgeo import gdal
import os
from glob import glob

count = 0
files = glob(r'.\split_label\*')
remove_file = []
for file in files:
    data = gdal.Open(file)
    dataset = data.ReadAsArray()
    nums = set(dataset.flatten())
    if len(nums) == 1:
        count += 1
        print(file)
        # os.remove(file)
        remove_file.append(file)
print(count)

for i in range(count):
    os.remove(remove_file[i])


