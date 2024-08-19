import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import math
import pickle
import gc
from utils import *

# class HyperData(Dataset):
#     def __init__(self, dataset):
#         self.data = dataset[0]
#         self.label = dataset[1]
    
#     def __getitem__(self, index):
#         img = self.data[index]
#         gt = self.label[index]
#         return img, gt
    
#     def __len__(self):
#         return len(self.label)

#     def __labels__(self):
#         return self.label

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
    
    # print("每类样本的个数:")
    # for i in range(classes):
    #     print(sum_num[i])

    # print("\n每类训练样本的个数:")
    # for i in range(classes):
    #     print(train_num[i])

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
    # x, y, index = createImgaeCubes(data, label, w)
    x = padWithZeros(data, w // 2)
    index = np.where(label != 0)[0]
    label = label - 1
    y = label[index]
    if(percent < 1):
        train_idx, test_idx = train_test_split_1(y, percent, classes)
    else:
        train_idx, test_idx = train_test_split_3(y, percent, classes)
    
    # x_train = x[train_idx].astype(np.float32)
    # x_test = x[test_idx].astype(np.float32)
    # y_train = y[train_idx].astype(np.int64)
    # y_test = y[test_idx].astype(np.int64)
    x = x.astype(np.float32)
    label = label.astype(np.int64)
    train_idx = index[train_idx]
    test_idx = index[test_idx]
    
    # train_data = HyperData((x_train, y_train))
    # test_data = HyperData((x_test, y_test))
    train_data = HyperData(x, label, train_idx, w)
    test_data = HyperData(x, label, test_idx, w)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=tr_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=te_batch_size, shuffle=False)

    # train = {
    #     'train_loader' : train_loader,
    #     'train_idx' : train_idx,
    # }

    # test = {
    #     'test_loader' : test_loader,
    #     'test_idx' : test_idx,
    # }
    
    # if not os.path.exists(process_path):
    #     os.makedirs(process_path)

    # with open(process_path + '/train.pickle', 'wb') as f:
    #     pickle.dump(train, f)

    # with open(process_path + '/test.pickle', 'wb') as f:
    #     pickle.dump(test, f)

    return train_loader, test_loader, train_idx, test_idx