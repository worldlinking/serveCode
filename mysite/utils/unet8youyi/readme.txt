1. label重分类：arcgis实现
将背景设为0，共7类

2. label图像裁剪：arcgis实现
单幅256大小，共2070幅

3. 删除标签均为背景类的标签：remove.py     (我认为背景类太多会有影响)
2070→1233
保存在split_label

4. 图像数据裁剪：extract.py
1233幅
保存在split_data

5. txt文件保存文件名：
image.txt
label.txt

5. 运行
unet/train.py
test.py


