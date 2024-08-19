from __future__ import print_function
from hyperData import *
from model import PSNet
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
import time
from torchsummary import summary
import pandas as pd
import gc

# np.random.seed(opt.seed)
# torch.manual_seed(opt.seed)
# torch.cuda.manual_seed(opt.seed)

model_path = opt.model_path + '/' + opt.dataset


def train():
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # ctiterion = myLoss()
    min_loss = 100000
    max_acc = 0.0
    for epoch in tqdm(range(opt.max_epoch)):
        loss_ = 0
        y_true, y_pred = [], []
        for i, data in enumerate(train_loader):
            points, target = data
            y_true += list(target.numpy())
            points = points.cuda()
            target = target.cuda()

            classifier.train()
            pred = classifier(points)
            one_hot_target = torch.nn.functional.one_hot(target, int(classes)).float()
            # loss = ctiterion(pred, one_hot_target)
            loss = F.mse_loss(pred, one_hot_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            y_pred += list(pred_choice.cpu().numpy())
            loss_ += loss

        acc = accuracy_score(y_true, y_pred)
        if epoch % 15 == 0:
            print('epoch %d: train loss: %f accuracy: %f' % (epoch, loss_.item(), acc))

        scheduler.step()
        if loss_ <= min_loss and max_acc <= acc:
            torch.save(classifier.state_dict(), '%s/model.pth' % model_path)
            min_loss = loss_
            max_acc = acc
        loss_ = 0


def test(num):
    pred = []
    y = []
    classifier.load_state_dict(torch.load('%s/model.pth' % model_path))
    for i, data in enumerate(test_loader):
        points, target = data
        y += list(target.numpy())
        points, target = points.cuda(), target.cuda()
        classifier.eval()
        pred_choice = classifier(points)
        pred_choice = pred_choice.data.max(1)[1]
        pred += list(pred_choice.cpu().numpy())

    result, confusion, each_acc, oa, aa, kappa = report(y, pred)

    colors = ['#8B008B', '#008B8B', '#00008B', '#A9A9A9', '#FF00FF', '#FFAEB9', '#FF0000', '#EE9A00',
              '#66CD00', '#2E8B57', '#00E5EE', '#FF7256', '#7B68EE', '#EEE9BF', '#8B795E', '#FF69B4']
    colors = hex_to_rgb(colors)

    img = np.zeros((row, col, 3), dtype=np.uint8)

    for i in range(len(y)):
        img[test_idx[i] // col, test_idx[i] % col] = colors[pred[i]]
    for i in range(train_idx.shape[0]):
        img[train_idx[i] // col, train_idx[i] % col] = colors[label[train_idx[i]] - 1]
    img = Image.fromarray(img)
    map_path = './result/' + opt.dataset
    if not os.path.exists(map_path):
        os.makedirs(map_path)
    img.save(map_path + '/map_' + str(opt.train_rate) + '_' + str(num) + '.png', transparent=True, dpi=(300.0, 300.0))

    # attack_types = np.arange(classes) + 1
    # plot_confusion_matrix(confusion, opt.dataset, classes=attack_types, normalize=True, title='confusion matrix')

    return result, each_acc, oa, aa, kappa


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

    print(feature.shape)

    result_txt = './result/' + opt.dataset + '/result_' + str(opt.train_rate) + '.txt'
    result_csv = './result/' + opt.dataset + '/result_' + str(opt.train_rate) + '.csv'

    n = 10
    each_accs = []
    oas = []
    aas = []
    kappas = []
    train_times = []
    test_times = []

    for i in range(n):
        train_loader, test_loader, train_idx, test_idx = load_hyperdata(classes, feature, label, process_path, opt.w,
                                                                        opt.train_rate, opt.tr_batch_size,
                                                                        opt.te_batch_size)

        # with open(process_path + '/train.pickle', 'rb') as f:
        #     tr = pickle.load(f)
        #     train_loader = tr['train_loader']
        #     train_idx = tr['train_idx']
        # with open(process_path + '/test.pickle', 'rb') as f:
        #     te = pickle.load(f)
        #     test_loader = te['test_loader']
        #     test_idx = te['test_idx']

        print(len(train_idx))
        print(len(test_idx))

        classifier = PSNet(feature.shape[-1], classes).cuda()

        # summary(classifier, (opt.w * opt.w, feature.shape[-1]))

        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        start = time.time()
        train()
        end = time.time()
        train_times.append(end - start)

        start = time.time()
        result, each_acc, oa, aa, kappa = test(i)
        end = time.time()
        test_times.append(end - start)

        each_accs.append(each_acc)
        oas.append(oa)
        aas.append(aa)
        kappas.append(kappa)

        pd.DataFrame(result).transpose().to_csv(result_csv, mode='a')

        torch.cuda.empty_cache()

        print(oa)

    f = open(result_txt, 'w')

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

    train_times = np.asarray(train_times)
    train_time_mean = np.mean(train_times)
    train_time_std = np.std(train_times)

    test_times = np.asarray(test_times)
    test_time_mean = np.mean(test_times)
    test_time_std = np.std(test_times)

    for i in range(classes):
        f.write("%.2f\u00B1%.2f\n" % (each_acc_mean[i] * 100, each_acc_std[i] * 100))
    f.write("oa_mean:%.2f\u00B1%.2f\n" % (oa_mean * 100, oa_std * 100))
    f.write("aa_mean:%.2f\u00B1%.2f\n" % (aa_mean * 100, aa_std * 100))
    f.write("kappa_mean:%.2f\u00B1%.2f\n" % (kappa_mean * 100, kappa_std * 100))
    f.write("%.2f\u00B1%.2f\n" % (train_time_mean * 100, train_time_std * 100))
    f.write("%.2f\u00B1%.2f\n" % (test_time_mean * 100, test_time_std * 100))

    f.close()