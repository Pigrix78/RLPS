import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, classification_report
import os
import random
import itertools
import matplotlib.pyplot as plt
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Indian', help='[Indian, PaviaU, KSC, Botswana, Salinas, Houston]')
parser.add_argument('--train_rate', type=float, default=0.05, help='Number of train samples.')
parser.add_argument('--tr_batch_size', type=int, default=256, help='train batch size')
parser.add_argument('--te_batch_size', type=int, default=256, help='test batch size')
parser.add_argument('--w', type=int, default=7, help='the width of the image cube')
parser.add_argument('--max_epoch', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--processData_path', type=str, default='./processData', help='process data path')
parser.add_argument('--model_path', type=str, default='./log', help='model path')
parser.add_argument('--seed', type=int, default=40, help='random seed')

opt = parser.parse_args()

print(opt)

def load_data(flag):
    data_path = 'E:/Datasets/' + flag + '/'
    if flag == 'Indian':
        data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')['indian_pines_corrected']
        label = sio.loadmat(data_path + 'Indian_pines_gt.mat')['indian_pines_gt']
    elif flag == 'PaviaU':
        data = sio.loadmat(data_path + 'PaviaU.mat')['paviaU']
        label = sio.loadmat(data_path + 'PaviaU_gt.mat')['paviaU_gt']
    elif flag == 'Botswana':
        data = sio.loadmat(data_path + 'Botswana.mat')['Botswana']
        label = sio.loadmat(data_path + 'Botswana_gt.mat')['Botswana_gt']
    elif flag == 'Salinas':
        data = sio.loadmat(data_path + 'Salinas_corrected.mat')['salinas_corrected']
        label = sio.loadmat(data_path + 'Salinas_gt.mat')['salinas_gt']
    elif flag == 'KSC':
        data = sio.loadmat(data_path + 'KSC.mat')['KSC']
        label = sio.loadmat(data_path + 'KSC_gt.mat')['KSC_gt']
    elif flag == 'Houston':
        data = sio.loadmat(data_path + 'Houston.mat')['houston']
        label = sio.loadmat(data_path + 'Houston_gt.mat')['houston_gt']

    (row, col, band) = data.shape
    label = label.flatten()
    classes = np.max(label)

    return classes, row, col, band, data, label

def padWithZeros(X, margin=2):
    x = np.zeros((X.shape[0] + 2*margin, X.shape[1] + 2*margin, X.shape[2]))
    x[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return x

def createImgaeCubes(X, y, w=5):
    index = np.where(y != 0)[0] #有效数据的索引

    margin = (w // 2)
    n = index.shape[0]
    x = padWithZeros(X, margin)
    patchData = np.zeros((n, w*w, X.shape[2]))
    patchLabel = y[index] - 1
    for i in range(n):
        center_i = index[i] // X.shape[1]
        center_j = index[i] % X.shape[1]
        t = x[center_i : center_i + 2*margin + 1, center_j : center_j + 2*margin + 1, :]
        t = np.reshape(t, (w*w, X.shape[2]))
        patchData[i, :, :] = t

    return patchData, patchLabel.astype("int"), index

def report(y_true, y_pred):
    result = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    s = np.sum(confusion, 1)
    correct = np.diag(confusion)
    each_acc = correct / s
    aa = np.mean(each_acc)

    return result, confusion, each_acc, oa, aa, kappa

# 绘制混淆矩阵
def plot_confusion_matrix(cm, dataset, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.savefig('./result/' + dataset + '/confusion_matrix.png')
    # plt.show()
    

def hex_to_rgb(value):
    n = len(value)
    colors = np.zeros((n, 3), dtype=np.int8)
    for i in range(n):
        v = value[i]
        v = v.lstrip('#')
        lv = len(v)
        colors[i] = tuple(int(v[j : j + lv // 3], 16) for j in range(0, lv, lv // 3))
    return colors

def getAdj(feature, y, classes):
    x = np.zeros((classes, feature.shape[1]))
    for i in range(classes):
        data = []
        for j in range(feature.shape[0]):
            if y[j] == i + 1:
                data.append(feature[j])
        data = np.asarray(data)
        x[i] = np.mean(data, axis=0)
    x = torch.from_numpy(x.astype(np.float32)).cuda()                                

    # 计算不同像素之间的l距离
    a = torch.norm(x[:, None] - x, dim=2, p=2)
    a = torch.exp(-a)
    # 计算度矩阵
    # d = 1. / torch.sum(a, dim=1)
    # adj = a * d + torch.eye(x.shape[0]).cuda()
    adj = a
    # print(adj)
    return adj

class myLoss(torch.nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
    
    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2))