from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, tree
from sklearn.linear_model import LogisticRegression
from util import *
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from train import rlTrain

train_rate = 0.05

model_name = 'svm'
mode = 'rl'
n = 10

# np.random.seed(42)  
#按照比例划分
def train_test_split(y, percent, classes):
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
                                                                                                                                                                                                                                     
def get_model(model_name):
    if model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == 'svm':
        model = svm.SVC(kernel='rbf', C=10)
    elif model_name == 'dt':
        model = tree.DecisionTreeClassifier()
    elif model_name == 'lr':
        model = LogisticRegression()
    return model

def get_select_data(path):
    if mode == 'rl':
        feature_idx = sio.loadmat(path + '/feature_idx.mat')['feature_idx'][0].astype(int)
        # print(feature_idx.shape)
        selectData = data[:, feature_idx]
    elif mode == 'all':
        selectData = data
    selectData = StandardScaler().fit_transform(selectData)
    return selectData

data, gt, band = load_data()
classes = gt.max() + 1

result = []
for j in range(1, 9):
    path = './' + flag + '/' + str(5 * j)
    # rlTrain(n_select_feature=j * 5)
    selectData = get_select_data(path)
    # print(selectData.shape)

    each_accs = []
    oas = []
    aas = []
    kappas = []
    result_path_csv = path + '/result.csv'
    result_path_txt = path + '/result.txt'
    for _ in range(n):
        model = get_model(model_name)
        
        train_idx, test_idx = train_test_split(gt, train_rate, classes)

        x_train = selectData[train_idx]
        y_train = gt[train_idx]

        x_test = selectData[test_idx]
        
        y_test = gt[test_idx]

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        confusion, each_acc, oa, aa, kappa = report(y_test, pred)
        # result = classification_report(y_test, pred, output_dict=True)
        # df = pd.DataFrame(result).transpose().to_csv(result_path_csv, mode='a')

        each_accs.append(each_acc)
        oas.append(oa)
        aas.append(aa)
        kappas.append(kappa)

        del model

    f = open(result_path_txt, 'w')
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    each_accs = np.asarray(each_accs)
    each_acc_mean = np.mean(each_accs, 0)
    each_acc_std = np.std(each_accs, 0)

    oas = np.asarray(oas)
    oa_mean = np.mean(oas)
    oa_std = np.std(oas)

    aas = np.asarray(aas)
    aa_mean = np.mean(aas)
    aa_std = np.std(aas)

    kappas = np.asarray(kappas)
    kappa_mean = np.mean(kappas)
    kappa_std = np.std(kappas)

    for i in range(classes):
        f.write("%.2f\u00B1%.2f\n" % (each_acc_mean[i] * 100, each_acc_std[i] * 100))
    f.write("%.2f\u00B1%.2f\n" % (oa_mean * 100, oa_std * 100))
    f.write("%.2f\u00B1%.2f\n" % (aa_mean * 100, aa_std * 100))
    f.write("%.2f\u00B1%.2f\n" % (kappa_mean * 100, kappa_std * 100))
    

    result.append([oa_mean, oa_std])
    print(oa_mean)

    f.close()

for i in range(8):
    print("%.2f\u00B1%.2f" % (result[i][0] * 100, result[i][1] * 100))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                