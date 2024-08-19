import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity

dataset = 'Indian'

if dataset == 'PaviaU':
    img = sio.loadmat('D:/data/' + dataset + '/PaviaU.mat')['paviaU']
    gt = sio.loadmat('D:/data/' + dataset + '/PaviaU_gt.mat')['paviaU_gt']
elif dataset == 'Indian':
    img = sio.loadmat('D:/data/' + dataset + '/Indian_pines_corrected.mat')['indian_pines_corrected']
    gt = sio.loadmat('D:/data/' + dataset + '/Indian_pines_gt.mat')['indian_pines_gt']
elif dataset == 'Houston':
    img = sio.loadmat('D:/data/' + dataset + '/Houston.mat')['houston']
    gt = sio.loadmat('D:/data/' + dataset + '/Houston_gt.mat')['houston_gt']

def get_dis(f):
    f = f.T 
    d = cosine_distances(f)
    return d

row, col, band = img.shape
img = img.reshape(row * col, band)
img = MinMaxScaler().fit_transform(img)
e = entropy(img)
print(e.shape)
D = get_dis(img)
img = img.reshape(row, col, band)

method = ['MVPCA', 'OPBS', 'BS-Net-Conv', 'TRC-OC-FDPC', 'ASPS', 'DRL', 'DRLBS']
color = ['green', 'pink', 'blue', 'coral', 'gold', 'red', 'darkorchid', 'black']
if dataset == 'Indian':
    OA = [49.37, 61.39, 57.35, 70.13, 69.14, 63.69, 71.85]
elif dataset == 'PaviaU':
    OA = [84.09, 88.69, 84.27, 87.22, 85.78, 86.07, 89.10]
elif dataset == 'Houston':
    OA = [68.53, 78.47, 50.65, 80.39, 79.78, 79.77, 81.41]

marker = ['o', 'v', 'D', '+', 's', 'p', 'h']

band = img.shape[-1]

plt.figure(figsize=(10, 6))

width = 0.2

x = np.arange(len(method))
y = []
y1 = []
for i, n in enumerate(method):
    feature_idx = sio.loadmat(dataset + '/' + n + '.mat')['feature_idx'][0].astype(int)
    if max(feature_idx == band):
        feature_idx = feature_idx - 1
    y.append(np.mean(e[feature_idx]))
    d = D[feature_idx, :][:, feature_idx]
    y1.append(np.mean(d))

y = y / max(y)
y1 = y1 / max(y1)
OA = np.array(OA)
OA = OA / max(OA)

# plt.plot(x, y, label='Mean Entropy')
# plt.plot(x, y1, label='Mean Cosine Distance')
# plt.plot(x, y2, label='OA')

plt.bar(x, y, width=width, tick_label=method, label='Mean Entropy')
plt.bar(x + width, y1, width=width, tick_label=method, label='Mean Cosine Distance')
plt.bar(x + 2 * width, OA, width=width, tick_label=method, label='OA')

# plt.legend(loc='best')

# plt.xlabel('band selection methods')

plt.savefig('ip_dis.png', dpi=300.0, bbox_inches='tight')
plt.show()