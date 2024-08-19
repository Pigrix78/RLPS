import numpy as np
import scipy.io as sio
import os
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import torch
from scipy.spatial.distance import euclidean, hamming, sqeuclidean, chebyshev

flag = 'Houston'

memory_size = 50000

learning_rate = 0.0001
gamma = 0.99
target_replace_iter = 100
batch_size = 100

train_rate= 0.05

def load_data():
    data_path = 'E:/Datasets/' + flag + '/'
    if flag == 'Indian':
        data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')['indian_pines_corrected']
        gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')['indian_pines_gt']
    elif flag == 'PaviaU':
        data = sio.loadmat(data_path + 'PaviaU.mat')['paviaU']
        gt = sio.loadmat(data_path + 'PaviaU_gt.mat')['paviaU_gt']
    elif flag == 'Botswana':
        data = sio.loadmat(data_path + 'Botswana.mat')['Botswana']
        gt = sio.loadmat(data_path + 'Botswana_gt.mat')['Botswana_gt']
    elif flag == 'Salinas':
        data = sio.loadmat(data_path + 'Salinas_corrected.mat')['salinas_corrected']
        gt = sio.loadmat(data_path + 'Salinas_gt.mat')['salinas_gt']
    elif flag == 'KSC':
        data = sio.loadmat(data_path + 'KSC.mat')['KSC']
        gt = sio.loadmat(data_path + 'KSC_gt.mat')['KSC_gt']
    elif flag == 'Houston':
        data = sio.loadmat(data_path + 'Houston.mat')['houston']
        gt = sio.loadmat(data_path + 'Houston_gt.mat')['houston_gt']

    row, col, band = data.shape

    row, col = data.shape[0], data.shape[1]

    gt = gt.flatten()

    idx = np.where(gt != 0)[0]
    gt = gt[idx] - 1
    data = data.reshape(row * col, band)

    data = data[idx]

    return data, gt, band

def report(y_true, y_pred):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)
    s = np.sum(confusion, 1)
    correct = np.diag(confusion)
    each_acc = correct / s
    aa = np.mean(each_acc)

    return confusion, each_acc, oa, aa, kappa

def get_cosine(x, y):
    d = np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))
    return d

def my_func(x, y):
    sid = entropy(x, y) + entropy(y, x)
    s = sid
    return s 

def get_f(data, classes):
    n_cl = classes
    initial_centers = kmeans_plusplus_initializer(data, amount_centers=n_cl).initialize()
    # kmeans_instance = kmeans(data, initial_centers)
    kmeans_instance = kmeans(data, initial_centers, metric=distance_metric(type_metric.USER_DEFINED, func=my_func))
    kmeans_instance.process()
    clusters = kmeans_instance.get_clusters()
    final_centers = kmeans_instance.get_centers()
    # x = PCA(2).fit_transform(data)
    # kmeans_visualizer().show_clusters(x, clusters, final_centers)
    final_centers = np.array(final_centers)
    print(final_centers.shape)
    return final_centers

    # y_pred = KMeans(n_clusters=classes, n_init=30, max_iter=500).fit_predict(data)
    # f = np.zeros((classes, band))
    # x = [[]] * classes
    # for i in range(data.shape[0]):
    #     x[y_pred[i]].append(data[i])
    # for i in range(classes):
    #     f[i] = np.mean(x[i])
    # return f

def get_dis(f):
    f = f.T 
    d = cosine_distances(f)
    return d

data, gt, band = load_data()

data = MinMaxScaler().fit_transform(data)

classes = gt.max() + 1
print(classes)

if not os.path.exists(flag):
    os.makedirs(flag)

if not os.path.exists(flag + '/D.mat'):
    f = get_f(data, classes)
    D = get_dis(f)
    sio.savemat(flag + '/D.mat', {'D' : D})
else:
    D = sio.loadmat(flag + '/D.mat')['D']

MIE = entropy(data)
MIE = MIE / MIE.sum()

# MIE[np.where(MIE < MIE.mean())] = 0

# plt.hist(MIE, 50)
# plt.show()