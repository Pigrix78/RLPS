import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

dataset = 'Indian'

if dataset == 'PaviaU':
    img = sio.loadmat('E:/Datasets/' + dataset + '/PaviaU.mat')['paviaU']
    gt = sio.loadmat('E:/Datasets/' + dataset + '/PaviaU_gt.mat')['paviaU_gt']
elif dataset == 'Indian':
    img = sio.loadmat('E:/Datasets/' + dataset + '/Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = sio.loadmat('E:/Datasets/' + dataset + '/Indian_pines_gt.mat')['indian_pines_gt']
elif dataset == 'Houston':
    img = sio.loadmat('E:/Datasets/' + dataset + '/Houston.mat')['houston']
    gt = sio.loadmat('E:/Datasets/' + dataset + '/Houston_gt.mat')['houston_gt']

row, col, band = img.shape
img = img.reshape(row * col, band)
img = MinMaxScaler().fit_transform(img)
e = entropy(img)
print(e.shape)
img = img.reshape(row, col, band)

method = ['MVPCA', 'OPBS', 'BS-Net-Conv', 'TRC-OC-FDPC', 'ASPS', 'DRL', 'DRLBS']
color = ['green', 'pink', 'blue', 'coral', 'gold', 'red', 'darkorchid', 'black']
marker = ['o', 'v', 'D', '+', 's', 'p', 'h']

band = img.shape[-1]

def export_legend(legend):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

plt.subplot(2, 1, 1)
x = np.arange(band)
for i, n in enumerate(method):
    y = np.zeros((band, 1))
    feature_idx = sio.loadmat(dataset + '/' + n + '.mat')['feature_idx'][0].astype(int) - 1
    y[feature_idx] = i + 1
    if i != 3:
        plt.scatter(x, y, c='', marker=marker[i], edgecolors=color[i], label=n)
    else:
        plt.scatter(x, y, c=[color[i]], marker=marker[i], label=n)
    plt.axis('off')
legend=plt.legend(bbox_to_anchor=(-0.15, -1.5), loc=2, ncol=7, prop={'size': 8})
export_legend(legend)

plt.subplot(2, 1, 2)
plt.plot(x, e, c='black')

plt.xlabel('光谱波段')

plt.savefig('ip_compare.png', dpi=300.0, bbox_inches='tight')