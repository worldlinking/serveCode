import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# from skimage.transform import resize as imresize
# import cv2

"""
ConfusionMetric
Mertric   P    N
P        TP    FN
N        FP    TN
"""


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, imgLabel, imgPredict):
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgLabel, imgPredict):
        assert imgLabel.shape == imgPredict.shape
        self.confusionMatrix += self.genConfusionMatrix(imgLabel, imgPredict)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def plot_confusion_matrix(self, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        cm = self.confusionMatrix
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if num > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    # 评价指标

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96）/ 3 =  0.89

    def intersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        IoU = self.intersectionOverUnion()
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def frequencyWeightedIntersectionOverUnion(self):
        # FWIOU = [(TP+FN)/(TP+FP+TN+FN)]*[TP/(TP+FP+FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = self.intersectionOverUnion()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

# label = r"C:\Users\User\Desktop\img\labels"
# label = np.transpose(label,(1, 2, 0))
# label = imresize(label, [512, 512])
# label = np.transpose(label,(2, 0, 1))


if __name__ == '__main__':
    label_true = np.array([0, 1, 1, 2, 1, 0, 2, 2, 1])  # 可直接换成标注图片
    label_pred = np.array([0, 1, 1, 2, 1, 0, 2, 2, 1])  # 可直接换成标注图片
    # label_pred = np.array(r"D:\python_project\pytorch\unet\test\002\Unet_images\test")  # 可直接换成预测图片
    metric = SegmentationMetric(4)  # 3表示有3个分类，有几个分类就填几
    metric.addBatch(label_true, label_pred)
    print(metric.confusionMatrix)
    pa = metric.pixelAccuracy()
    cpa = metric.classPixelAccuracy()
    mpa = metric.meanPixelAccuracy()
    IoU = metric.intersectionOverUnion()
    mIoU = metric.meanIntersectionOverUnion()
    FWIoU = metric.frequencyWeightedIntersectionOverUnion()
    print('pa is : %f' % pa)
    print('cpa is :', cpa)
    print('mpa is : %f' % mpa)
    print('IoU is :', IoU)
    print('mIoU is : %f' % mIoU)
    print('FWIoU is : %f' % FWIoU)
    # metric.plot_confusion_matrix(classes=['background', 'cat', 'dog'])

    # 对比sklearn
    metric = confusion_matrix(label_true, label_pred)
    print(metric)


