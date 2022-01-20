import torch
from torch.utils.data import DataLoader, Subset
from MyData import MyDataSet
from utils import get_index
from Models import Net, NetWithBN, NetWithAttention
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score


def run_test(cv_idx, run_test_idx, trainDataSet, testDataSet, epochs, lr, n_class, drop_rate=0.2):
    train_loader = DataLoader(trainDataSet, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataSet, shuffle=False, batch_size=1)
    ins_len = len(trainDataSet[0][0][0])
    model = NetWithAttention(ins_len=ins_len, n_class=n_class, drop_rate=drop_rate)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # 与tensorflow不同，此处交叉熵损失先会将输入softmax
                                                            # 并且真实标签只能为单数字，不能为one-hot
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练一轮，测试一次
    acc_list = []
    f1_list = []
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            target = target.long()

            inputs = inputs.squeeze(0)
            outputs = model(inputs)

            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('loss of %3d -th opoch in %2d -th Run Test of %d -th CV: %.3f :'
              % (epoch + 1, run_test_idx + 1, cv_idx + 1, running_loss / len(trainDataSet)), end=' # ')
        # testing phase
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []
            for data in test_loader:
                inputs, labels = data
                y_true.append(labels)
                inputs = inputs.squeeze(0)
                outputs = model(inputs)

                _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引
                y_pred.append(pred)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                acc = correct / total
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print('Test: acc: %.2f | f1: %.2f' % (acc, f1))
        acc_list.append(acc)
        f1_list.append(f1)
        if acc == 1:
            break
    return np.max(acc_list), np.max(f1_list)


def one_cv(path, cv_idx, para_k, epochs, lr, drop_rate=0.2):
    AllDataSet = MyDataSet(path=path)
    n_class = np.max(AllDataSet[:][1]) + 1
    train_idx_list, test_idx_list = get_index(len(AllDataSet), para_k=para_k)
    acc_list, f1_list = [], []
    for i in range(para_k):
        trainDataSet = Subset(AllDataSet, train_idx_list[i])
        testDataSet = Subset(AllDataSet, test_idx_list[i])
        acc, f1 = run_test(cv_idx=cv_idx, run_test_idx=i, trainDataSet=trainDataSet, testDataSet=testDataSet,
                           epochs=epochs, lr=lr, n_class=n_class, drop_rate=drop_rate)
        acc_list.append(acc)
        f1_list.append(f1)
    print('-' * 50 + 'One CV Done' + '-' * 6 + 'acc: ' + str(np.mean(acc_list)) + ' f1: ' + str(np.mean(f1_list)))
    return float(np.mean(acc_list)), float(np.mean(f1_list))


def n_cv(path, num_cv, para_k, epochs, lr, drop_rate=0.2):
    acc_list, f1_list = [], []
    for i in range(num_cv):
        acc, f1 = one_cv(path=path, cv_idx=i, para_k=para_k, epochs=epochs, lr=lr, drop_rate=drop_rate)
        acc_list.append(acc)
        f1_list.append(f1)
    print('*' * 10 + path.split('/')[-1].split('.')[0] + '*' * 10)
    print('lr: ', lr)
    print('epochs: ', epochs)
    print('drop: ', drop_rate)
    return float(np.mean(acc_list)), float(np.std(acc_list)), float(np.mean(f1_list)), float(np.std(f1_list))


if __name__ == '__main__':
    # Text: lr=0.00001, Benchmark: 0.00005 | epochs: 100, K: 3
    acc, acc_std, f1, f1_std = n_cv(path='../Data/Benchmark/fox+.mat', num_cv=5, para_k=10,
                                    epochs=120, lr=0.002, drop_rate=0.2)
    print('Mean Result of 5 CV : acc: $%.2f_{\\pm%.2f}$ | f1: $%.2f$_{\\pm%.2f}'
          % (acc * 100, acc_std * 100, f1 * 100, f1_std * 100))
    # Without BN
    # musk1: lr=0.0001, epochs=120, acc: $92.40_{\pm1.02}$ | f1: $84.13$_{\pm2.88}
    # musk2: lr=0.001, epochs=120, drop=0.2, acc: $88.24_{\pm2.18}$ | f1: $81.95$_{\pm4.73}
    # elephant: lr=0.001, epochs=100, acc: $86.00_{\pm6.44}$ | f1: $87.09$_{\pm4.98}
    # News.aa: lr=0.0001, epochs=100, drop=0.2, acc: $86.80_{\pm6.01}$ | f1: $86.58$_{\pm7.19}

    # WithBN
    # News.aa: lr=0.0001, epochs=100, drop=0.2, acc: $86.80_{\pm6.01}$ | f1: $86.58$_{\pm7.19}
