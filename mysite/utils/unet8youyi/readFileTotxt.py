'''
保存文件名至txt
'''
from glob import glob

path1 = 'image.txt'
path2 = "label.txt"
image = r'split_data\*'
label = r'split_label\*'
with open(path1, "a") as f:
    for file in glob(image):
        f.write(file)
        f.write('\n')
with open(path2, "a") as f:
    for file in glob(label):
        f.write(file)
        f.write('\n')

