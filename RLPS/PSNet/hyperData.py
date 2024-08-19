import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
import torch.nn.functional as F
import math
import pickle
import gc
from utils import *

class HyperData(Dataset):
    def __init__(self, data, label, idx, w):
        self.data = data
        self.label = label
        self.idx = idx
        self.w = w
        self.col = self.data.shape[1] - w + 1
        self.i = self.idx // self.col
        self.j = self.idx % self.col
    
    def __getitem__(self, index):
        i = self.i[index]
        j = self.j[index]
        img = self.data[i : i + self.w, j : j + self.w].reshape((self.w * self.w, -1))
        gt = self.label[self.idx[index]]
        return img, gt
    
    def __len__(self):
        return len(self.idx)

#按照比例划分
def train_test_split_1(y, percent, classes):
    train_idx = []
    test_idx = []
    sum_num = []
    train_num = []
    for i in range(classes):
        pos = np.where(y == i)[0]

        sum_num.append(pos.shape[0]) #每类样本的个数
        train_num.append(int(sum_num[i]*percent))

        np.random.shuffle(pos)
        pos = list(pos)
        train_idx += pos[:train_num[i]]
        test_idx += pos[train_num[i]:]

    return train_idx, test_idx

#每类训练样本最少为10
def train_test_split_2(y, percent, classes):
    train_idx = []
    test_idx = []
    for i in range(classes):
        pos = np.where(y == i)[0]

        sum_num = pos.shape[0] #每类样本的个数
        n = math.ceil(sum_num*percent)
        train_num = n if n >= 10 else 10
        print(train_num)

        np.random.shuffle(pos)
        pos = list(pos)
        train_idx += pos[:train_num]
        test_idx += pos[train_num:]

    return train_idx, test_idx

#每类训练样本个数相同
def train_test_split_3(y, train_num, classes):
    train_num = int(train_num)
    train_idx = []
    test_idx = []
    for i in range(classes):
        pos = np.where(y == i)[0]
        sum_num = pos.shape[0] #每类样本的个数
        # print(sum_num)

        np.random.shuffle(pos)
        pos = list(pos)
        if(sum_num <= train_num):
            train_idx += pos[:15]
            test_idx += pos[15:]
        else:
            train_idx += pos[:train_num]
            test_idx += pos[train_num:]

    return train_idx, test_idx

def load_hyperdata(classes, data, label, process_path, w, percent, tr_batch_size, te_batch_size):
    x = padWithZeros(data, w // 2)
    index = np.where(label != 0)[0]
    label = label - 1
    y = label[index]
    if(percent < 1):
        train_idx, test_idx = train_test_split_1(y, percent, classes)
    else:
        train_idx, test_idx = train_test_split_3(y, percent, classes)

    x = x.astype(np.float32)
    label = label.astype(np.int64)
    train_idx = index[train_idx]
    test_idx = index[test_idx]

    train_data = HyperData(x, label, train_idx, w)
    test_data = HyperData(x, label, test_idx, w)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=tr_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=te_batch_size, shuffle=False)

    return train_loader, test_loader, train_idx, test_idx

if __name__ == '__main__':
    classes, row, col, band, data, label = load_data(opt.dataset)
    feature = data.reshape(-1, band)

    process_path = opt.processData_path + '/' + opt.dataset
    feature_idx = sio.loadmat(process_path + '/feature_idx.mat')['feature_idx'][0]
    feature = feature[:, feature_idx]
    feature = StandardScaler().fit_transform(feature)
    feature = feature.reshape(row, col, -1)

    index = np.zeros((row, col, 2))
    for i in range(row):
        for j in range(col):
            index[i, j, 0] = i
            index[i, j, 1] = j
    index = index.reshape(row * col, 2)
    index = StandardScaler().fit_transform(index)
    index = index.reshape((row, col, 2))

    feature = np.concatenate((feature, index), 2)
    print(f"feature shape: {feature.shape}")
    print(f"First data:\n{feature[0]}")  # 打印第一个数据

    train_loader, test_loader, train_idx, test_idx = load_hyperdata(classes, feature, label, process_path, opt.w,
                                                                    opt.train_rate, opt.tr_batch_size,
                                                                    opt.te_batch_size)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")  # 打印批次信息