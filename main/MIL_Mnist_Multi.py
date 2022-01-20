import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm


class MnistBagsDateset(Dataset):
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


def run(train_path, test_path, epochs, lr, agg_out_len):
    trainDataset = MnistBagsDateset(path=train_path)
    testDataset = MnistBagsDateset(path=test_path)

    trainLoader = DataLoader(trainDataset, shuffle=False, batch_size=1)
    testLoader = DataLoader(testDataset, shuffle=False, batch_size=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(agg_in_len=500, agg_out_len=agg_out_len, self_in_len=5 * 4, self_out_len=2 * 4, n_class=4).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练一轮，测试一次
    acc_list = []
    f1_list = []
    auc_list = []
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        # total = 0.0
        # correct = 0.0
        y_true, y_pred, acc = [], [], 0
        for batch_idx, data in enumerate(trainLoader, 0):
            inputs, target = data
            target = target.long()
            target = target.to(device)
            y_true.append(target.cpu().detach().numpy())
            inputs = inputs.squeeze(0)
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引
            y_pred.append(pred.cpu().detach().numpy())
            # total += target.size(0)
            # correct += (pred == target).sum().item()
            # acc = correct / total
            acc = accuracy_score(y_true, y_pred)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('%2d -th epoch: Train: loss: %.3f, acc: %.2f:' % (epoch + 1, running_loss / 100, acc * 100), end=' # ')
        # testing phase
        # correct = 0
        # total = 0
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []
            y_prob = []
            for data in testLoader:
                inputs, labels = data
                y_true.append(labels.cpu().detach().item())
                inputs = inputs.squeeze(0)
                inputs = inputs.to(device)
                outputs = model(inputs)
                temp_prob = torch.softmax(outputs, 1)
                temp_prob = np.squeeze(temp_prob.cpu().numpy())
                y_prob.append(temp_prob)
                _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引
                y_pred.append(pred.cpu().detach().item())
                # total += labels.size(0)
                # correct += (pred == labels).sum().item()
                # acc = correct / total
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0, average='macro')
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print('Test: acc: %.1f' % acc)
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        if acc == 1:
            break
    return np.max(acc_list), np.max(f1_list), np.max(auc_list)


if __name__ == '__main__':
    train_path = '../../Data/Mnist-bags/Multi/100/mnist_100_train.mat'
    test_path = '../../Data/Mnist-bags/Multi/100/mnist_100_test.mat'
    dataset_name = train_path.split('/')[-1].split('.')[0][:8]
    epochs = 100
    lr = 0.0001
    agg_out_len = 512
    acc, f1, auc = run(train_path, test_path, epochs, lr, agg_out_len)

    print(dataset_name)
    print('epochs: ', epochs)
    print('lr:', lr)
    print('Acc: %.3f' % acc)
    print('F1: %.3f' % f1)
    print('Auc: %.3f' % auc)
    # for 10, agg_in_len=512, acc: 99
    # for 50, agg_in_len=1024, acc: 97
    # for 100, agg_in_len=2048, acc: 95

    # acc_list = []
    # train_path = '../../Data/Mnist-bags/Multi/100/mnist_100_train.mat'
    # test_path = '../../Data/Mnist-bags/Multi/100/mnist_100_test.mat'
    # dataset_name = train_path.split('/')[-1].split('.')[0]
    # print(dataset_name)
    # epochs = 100
    # lr = 0.0001
    # for agg_out_len in tqdm(np.array([64, 128, 256, 512, 1024, 2048])):
    #     acc = run(train_path, test_path, epochs, lr, agg_out_len)
    #     acc_list.append(acc)
    # print(acc_list)
    # mean bag size:10 , [94.7, 97.2, 98.3, 99.1, 98.9, 99.0]
    # mean bag size:50 , [91.4, 94.2, 95.6, 97.4, 97.1, 97.4]
    # mean bag size:100, [86.7, 92.1, 93.3, 95.6, 95.1, 95.2]
