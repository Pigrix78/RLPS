import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

color = ['green', 'pink', 'blue', 'coral', 'gold', 'red', 'darkorchid', 'black']
marker = ['o', 'v', 'D', '+', 's', 'p', 'h']
cname = ['IP', 'PU', 'HS']
cn = ['ip_bs', 'pu_bs', 'hs_bs']

for t in range(3):
    plt.figure(t)
    dt = pd.read_excel('result.xls', sheet_name='Sheet' + str(t + 1))
    x = dt[cname[t]].values
    label = list(dt.columns)
    label.pop(0)
    for i in range(len(label) - 1):
        l = label[i]
        y = list(dt[l].values)
        for j in range(len(y)):
            y[j] = y[j][:5]
            y[j] = float(y[j])
        plt.plot(x, y, color=color[i], marker=marker[i], label=l)
    y = list(dt[label[-1]].values)
    print(label[-1])
    for j in range(len(y)):
        y[j] = y[j][:5]
        y[j] = float(y[j])
    plt.plot(x, y, color=color[-1], linestyle=':', label=label[-1])
    plt.xlabel('波段数目')
    plt.ylabel('OA')
    legend = plt.legend()
    plt.savefig(cn[t] + '.png', dpi=300.0)