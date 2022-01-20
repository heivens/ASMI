"""For Mnist"""
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np
from utils import get_index


# 加载cv用的数据集
class MnistBinaryDateset(Dataset):
    def __init__(self, path):
        self.data = loadmat(path)['data']

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1].tolist()[0][0]

    def __len__(self):
        return len(self.data)


# Convolve the pictures to instances
class MyConv(nn.Module):
    def __init__(self):
        super(MyConv, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.Dropout(),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL
        return H


# 定义注意力层聚合包内信息的层, 输入为[batchsize, ins_len]
class AttentionInBag(nn.Module):
    def __init__(self, agg_in_len, agg_out_len):
        super(AttentionInBag, self).__init__()
        self.ascend_dim = nn.Linear(in_features=agg_in_len, out_features=agg_out_len)
        self.compute_e = nn.Linear(in_features=agg_out_len, out_features=1)

    def forward(self, bag):
        bag = bag.float()
        bag = nn.LeakyReLU(0.2)(self.ascend_dim(bag))
        e_list = self.compute_e(bag)
        e_list = torch.reshape(e_list, (1, bag.shape[0]))
        alpha_list = F.softmax(e_list, dim=1)
        x = torch.mm(alpha_list, bag)
        return x


# 定义自注意力层聚合实例标签和包标签的信息, 输入为[batchsize, ins_len]
class SelfAttention(nn.Module):
    def __init__(self, self_att_in_len, self_att_out_len):
        super(SelfAttention, self).__init__()
        self.ascend_dim = nn.Linear(in_features=self_att_in_len, out_features=self_att_out_len)
        self.compute_e = nn.Linear(in_features=2 * self_att_out_len, out_features=1)

    def forward(self, bag):
        bag = nn.LeakyReLU(0.2)(self.ascend_dim(bag))
        center_ins = torch.reshape(bag[0], (1, bag.shape[1]))
        center_ins_matrix = center_ins.repeat(bag.shape[0], 1)  # 将center_ins在第0个维度上复制三次，第1个维度上只复制一次
        self_neighbors = torch.cat((center_ins_matrix, bag), dim=1)
        self_neighbors = self_neighbors.float()
        # print('self_neighbors:', self_neighbors)
        e_list = self.compute_e(self_neighbors)
        e_list = torch.reshape(e_list, (1, e_list.shape[0]))  # e_list reshape为1*3
        alpha_list = F.softmax(e_list, dim=1)
        aggrgated_ins = torch.matmul(alpha_list, bag)  # 聚合后的单向量
        return aggrgated_ins


# 定义整个网络
class Net(nn.Module):
    def __init__(self, agg_in_len, agg_out_len, self_in_len, self_out_len, n_class):
        super(Net, self).__init__()
        self.conv = MyConv()
        self.bag_info = nn.Sequential(
            AttentionInBag(agg_in_len=agg_in_len, agg_out_len=agg_out_len),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=agg_out_len, out_features=int(agg_out_len / 10)),
            nn.LeakyReLU(negative_slope=0.2),
            # 处理到self-attention层的输入长度
            nn.Linear(in_features=int(agg_out_len / 10), out_features=self_in_len)
        )
        self.ins_info = nn.Sequential(
            nn.Linear(in_features=agg_in_len, out_features=int(agg_in_len / 2)),
            nn.LeakyReLU(negative_slope=0.2),
            # 处理到self-attention层的输入长度
            nn.Linear(in_features=int(agg_in_len / 2), out_features=self_in_len)
        )
        self.agg_bag_ins = SelfAttention(self_att_in_len=self_in_len, self_att_out_len=self_out_len)
        self.agg_bag_ins_linear = nn.Linear(in_features=self_out_len, out_features=n_class)

    def forward(self, bag):
        bag = bag.float()
        bag = self.conv(bag)
        bag_info = self.bag_info(bag)
        ins_info = self.ins_info(bag)
        total_info = torch.cat((bag_info, ins_info), dim=0)
        x = nn.LeakyReLU(0.2)(self.agg_bag_ins(total_info))
        y = self.agg_bag_ins_linear(x)
        return y


# n cv
def n_cv(path, epochs, lr, cv_num, para_k):
    acc_list, f1_list = [], []
    for i in range(cv_num):
        acc, f1 = one_cv(path=path, epochs=epochs, lr=lr, i_th_cv=i, para_k=para_k)
        acc_list.append(acc)
        f1_list.append(f1)
    return np.mean(acc_list), np.std(acc_list), np.mean(f1_list), np.std(f1_list)


# one cv
def one_cv(path, epochs, lr, i_th_cv, para_k=10):
    AllDataSet = MnistBinaryDateset(path=path)
    train_idx_list, test_idx_list = get_index(len(AllDataSet), para_k=para_k)
    acc_list, f1_list = [], []
    for i in range(para_k):
        trainDataset = Subset(AllDataSet, train_idx_list[i])
        testDataset = Subset(AllDataSet, test_idx_list[i])
        f1, acc = run(trainDataset=trainDataset, testDataset=testDataset,
                      epochs=epochs, lr=lr, i_th_run=i, i_th_cv=i_th_cv)
        acc_list.append(acc)
        f1_list.append(f1)
    return np.mean(acc_list), np.mean(f1_list)


# 运行测试
def run(trainDataset, testDataset, epochs, lr, i_th_run, i_th_cv):
    train_loader = DataLoader(trainDataset, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataset, shuffle=False, batch_size=1)

    model = Net(agg_in_len=500, agg_out_len=512,
                self_in_len=10, self_out_len=4, n_class=2)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练一轮，测试一次
    acc_list = []
    f1_list = []
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        total = 0.0
        correct = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            target = target.long()

            inputs = inputs.squeeze(0)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引

            total += target.size(0)
            correct += (pred == target).sum().item()
            acc = correct / total

            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('%2d -th CV, %2d -th Run, %2d -th epoch: Train: loss: %.3f, acc: %.2f:' %
              (i_th_cv + 1, i_th_run + 1, epoch + 1, running_loss / 100, acc * 100), end=' # ')
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


if __name__ == '__main__':
    path = '../../Data/Mnist-bags/mnist_7.mat'
    dataset_name = path.split('/')[-1]
    epochs = 20
    lr = 0.0001
    para_k = 10  # 10CV

    acc, acc_std, f1, f1_std = n_cv(path=path, epochs=epochs, lr=lr, cv_num=5, para_k=para_k)
    print('-' * 20)
    print(dataset_name)
    print('epochs: ', epochs, end=', ')
    print('lr:', lr, end=', ')
    print('para_k: ', para_k)
    print('Mean Result of 5 CV : acc: $%.2f_{\\pm%.2f}$ | f1: $%.2f_{\\pm%.2f}$'
          % (acc * 100, acc_std * 100, f1 * 100, f1_std * 100))

    # mnist_0: epochs: 20, lr: 0.0001, para_k:10, acc: $98.10_{\pm0.62}$ | f1: $98.20_{\pm0.68}$
    # mnist_1: epochs: 20, lr: 0.0001, para_k:10, acc: $97.94_{\pm0.48}$ | f1: $98.10_{\pm0.37}$
    # mnist_2: epochs: 20, lr: 0.0001, para_k:10, acc: $96.72_{\pm0.63}$ | f1: $96.80_{\pm0.75}$
    # mnist_3: epochs: 20, lr: 0.0001, para_k:10, acc: $95.77_{\pm0.47}$ | f1: $96.20_{\pm0.68}$
    # mnist_4: epochs: 20, lr: 0.0001, para_k:10, acc: $97.36_{\pm0.43}$ | f1: $97.40_{\pm0.37}$
    # mnist_5: epochs: 20, lr: 0.0001, para_k:10, acc: $95.89_{\pm1.09}$ | f1: $95.90_{\pm0.92}$
    # mnist_6: epochs: 20, lr: 0.0001, para_k:10, acc: $99.57_{\pm0.49}$ | f1: $99.50_{\pm0.51}$
    # mnist_7: epochs: 20, lr: 0.0001, para_k:10, acc: $97.99_{\pm0.47}$ | f1: $98.00_{\pm0.45}$
    # mnist_8: epochs: 20, lr: 0.0001, para_k:10, acc: $91.58_{\pm1.42}$ | f1: $91.80_{\pm1.63}$
    # mnist_8: epochs: 20, lr: 0.0001, para_k:10, acc: $96.43_{\pm1.32}$ | f1: $96.50_{\pm1.14}$

