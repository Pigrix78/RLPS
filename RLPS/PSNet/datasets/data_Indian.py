from __future__ import print_function
from hyperData import *
import os  # 导入操作系统模块
import glob  # 导入用于匹配文件路径模块
import h5py  # 导入处理HDF5文件格式的模块
import numpy as np  # 导入处理数组和矩阵的模块
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset  # 导入PyTorch中的数据集基类
from utils import *
from collections import defaultdict  # 导入默认字典模块
import random  # 导入随机数生成模块

from model import PSNet
from point_nn import Point_NN
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import time
from torchsummary import summary
import pandas as pd

from utils import opt

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # 配置HDF5文件锁定设置为FALSE


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]  # 读取文件列表并返回


def load_data_Indian(partition):
    #download()  # 下载数据集
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录路径
    # DATA_DIR = BASE_DIR + '/../processData'  # 数据存储目录
    # all_data = []  # 存储所有数据
    # all_label = []  # 存储所有标签
    # for h5_name in glob.glob(os.path.join(DATA_DIR, 'Indian', 'ply_data_%s*.h5' % partition)):
    #     f = h5py.File(h5_name, 'r')  # 打开HDF5文件
    #     data = f['data'][:].astype('float32')  # 读取数据并转为浮点型
    #     label = f['label'][:].astype('int64')  # 读取标签并转为整型
    #     f.close()
    #     all_data.append(data)
    #     all_label.append(label)
    #
    # print(f"all_data shape: {all_data}")
    # print(f"First data:\n{all_data[0]}")  # 打印第一个数据
    # all_data = np.concatenate(all_data, axis=0)  # 拼接所有数据
    # print(f"all_data shape: {all_data.shape}")
    # print(f"First data:\n{all_data[0]}")  # 打印第一个数据
    # all_label = np.concatenate(all_label, axis=0)  # 拼接所有标签
    # print(f"all_label shape: {all_label.shape}")
    # print(f"First data:\n{all_label[0]}")  # 打印第一个数据
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
    print(f"label shape: {label.shape}")
    print(f"First data:\n{label[0]}")  # 打印第一个数据
    return feature, label

# 随机丢失部分点云
def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random() * max_dropout_ratio  # 随机生成丢失点比例
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]  # 随机选择丢失点的索引
    if len(drop_idx) > 0:
        pc[drop_idx, :] = pc[0, :]  # 将选择的点设为第一个点
    return pc

# 点云平移
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])  # 随机生成平移参数
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])  # 随机生成平移参数
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')  # 进行平移操作
    return translated_pointcloud

# 点云抖动
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)  # 添加抖动操作
    return pointcloud


class Indian(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_Indian(partition)  # 加载数据
        self.num_points = num_points  # 设置点云数量
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]  # 获取指定数量的点云数据
        label = self.label[item]  # 获取标签
        if self.partition == 'train':
            np.random.shuffle(pointcloud)  # 在训练集上进行点云打乱
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]  # 返回数据集大小
"""
这段代码定义了一个名为ModelNet40的数据集类，继承自PyTorch中的Dataset基类。该类用于加载和处理ModelNet40数据集。

__init__(self, num_points, partition='train')：初始化函数。在创建ModelNet40对象时调用。它接受两个参数，num_points表示每个点云数据的采样点数，
partition表示数据集的划分（训练集或测试集）。在初始化过程中，使用load_data(partition)函数加载相应的数据，并将数据存储在self.data和self.label中。

__getitem__(self, item)：获取数据和标签的函数。当使用train[index]方式获取数据集中的样本时，会调用该函数。它接收一个索引item作为输入，
并返回对应索引处的点云数据和标签。在该函数中，根据指定的采样点数num_points从self.data中获取对应数量的点云数据，
然后根据partition判断是否在训练集上进行点云打乱操作（通过随机打乱点的顺序），最后返回点云数据和对应的标签。

__len__(self)：返回数据集大小的函数。当使用len(train)获取数据集的大小时，会调用该函数。它返回self.data的第一个维度大小，即数据集的样本数量。

通过执行train = ModelNet40(1024)语句，创建了一个ModelNet40对象，并传入num_points参数为1024。
train变量会成为一个ModelNet40类的实例对象，可以使用其索引方式train[index]获取具体的样本数据和标签。

在创建train对象时，会执行ModelNet40类的初始化函数__init__来加载数据，并根据传入的num_points参数和数据集的划分类型进行配置。
其他两个函数__getitem__和__len__会在需要获取数据或获得数据集大小时被调用。
"""

if __name__ == '__main__':
    #设置num_points参数为1024，可以将每个点云数据统一采样为包含1024个点的点云。
    train = Indian(74)  # 创建训练集
    test = Indian(74, 'test')  # 创建测试集

    from torch.utils.data import DataLoader
    train_loader = DataLoader(Indian(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)  # 创建训练数据加载器
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")  # 打印批次信息

    train_set = Indian(partition='train', num_points=1024)  # 获取训练集
    test_set = Indian(partition='test', num_points=1024)  # 获取测试集
    print(f"train_set size {train_set.__len__()}")  # 打印训练集大小
    print(f"test_set size {test_set.__len__()}")  # 打印测试集大小