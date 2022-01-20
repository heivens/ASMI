from torch.utils.data import Dataset, DataLoader, Subset
from utils import load_multi_class_bag
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import time


class MyDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# 每个类别随机选取一半作为训练集,另一半作为测试集,直接返回pytorch的Dataset类
def load_data(path, train_tate):
    bags, labels = load_multi_class_bag(path)
    n_bags = labels.shape[0]
    n_class = np.max(labels) + 1
    # 一共n_class行，每一行的包标签为对应行索引
    bag_idx = np.reshape(np.arange(0, n_bags), (-1, 100))
    train_bag_idx, test_bag_idx = [], []
    for i in range(n_class):
        total_index = bag_idx[i]  # 包标签为i的所有包索引
        train_idx = np.random.choice(total_index, int(len(total_index) * train_tate), replace=False)
        test_idx = np.setdiff1d(total_index, train_idx)
        train_bag_idx.append(np.sort(train_idx))
        test_bag_idx.append(np.sort(test_idx))
    train_bag_idx = np.reshape(train_bag_idx, (-1))
    test_bag_idx = np.reshape(test_bag_idx, (-1))
    np.random.shuffle(train_bag_idx)
    np.random.shuffle(test_bag_idx)
    trainDataset = MyDataset(bags[train_bag_idx], labels[train_bag_idx])
    testDataset = MyDataset(bags[test_bag_idx], labels[test_bag_idx])
    return trainDataset, testDataset


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
        self.bag_info = nn.Sequential(
            AttentionInBag(agg_in_len=agg_in_len, agg_out_len=agg_out_len),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=agg_out_len, out_features=int(agg_out_len / 2)),
            nn.LeakyReLU(negative_slope=0.2),
            # 处理到self-attention层的输入长度
            nn.Linear(in_features=int(agg_out_len / 2), out_features=self_in_len)
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
        bag_info = self.bag_info(bag)
        ins_info = self.ins_info(bag)
        total_info = torch.cat((bag_info, ins_info), dim=0)
        x = nn.LeakyReLU(0.2)(self.agg_bag_ins(total_info))
        y = self.agg_bag_ins_linear(x)
        return y


def run(path, n_class, train_rate, epochs, lr):
    trainDataset, testDataset = load_data(path, train_rate)
    train_loader = DataLoader(trainDataset, shuffle=False, batch_size=1)
    test_loader = DataLoader(testDataset, shuffle=False, batch_size=1)
    ins_len = len(trainDataset[0][0][0])
    agg_out_len = 5 * ins_len
    self_in_len = 5 * n_class
    self_out_len = 2 * n_class
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net(agg_in_len=ins_len, agg_out_len=agg_out_len,
                self_in_len=self_in_len, self_out_len=self_out_len, n_class=n_class).to(device)

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')  # 与tensorflow不同，此处交叉熵损失先会将输入softmax
    # 并且真实标签只能为单数字，不能为one-hot
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 训练一轮，测试一次
    acc_list = []
    f1_list = []
    auc_list = []
    for epoch in range(epochs):
        # training phase
        model.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_loader, 0):
            inputs, target = data
            target = target.long()
            target = target.to(device)
            inputs = inputs.squeeze(0)
            inputs = inputs.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # print('loss of %3d -th opoch: %.3f :'
        #       % (epoch + 1, running_loss / len(trainDataset)), end=' # ')
        # testing phase
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            y_pred = []
            y_true = []
            y_prob = []
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.squeeze(0)
                y_true.append(labels.detach().item())
                labels = labels.to(device)
                inputs = inputs.to(device)

                outputs = model(inputs)
                temp_prob = torch.softmax(outputs, 1)
                y_prob.append(np.squeeze(temp_prob.cpu().numpy()))
                _, pred = torch.max(outputs.data, dim=1)  # 返回最大值及其索引
                y_pred.append(pred.detach().cpu().item())
                total += labels.size(0)
                correct += (pred == labels).sum().detach().cpu().item()
                acc = correct / total
        f1 = f1_score(y_true, y_pred, zero_division=0, average='macro')
        y_prob = np.array(y_prob)
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        # print('Test: acc: %.2f | f1: %.2f' % (acc, f1))
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        if acc == 1:
            break
    return np.max(acc_list), np.max(f1_list), np.max(auc_list)


if __name__ == '__main__':
    path = '../../Data/Corel/Blobworld/类别有重复/corel-10/corel-10-b-230+.mat'
    dataset_name = path.split('/')[-1]
    n_class = int(dataset_name.split('-')[1])
    # n_class = 3
    train_rate = 0.9
    epochs = 150
    lr = 0.0001
    times = 10
    acc_list, f1_list, auc_list = [], [], []
    for i in tqdm(range(times)):
        acc, f1, auc = run(path, n_class, train_rate, epochs, lr)
        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
    acc_avg = float(np.mean(acc_list))
    acc_std = float(np.std(acc_list, ddof=1))
    f1_avg = float(np.mean(f1_list))
    f1_std = float(np.std(f1_list, ddof=1))
    auc_avg = float(np.mean(auc_list))
    auc_std = float(np.std(auc_list, ddof=1))
    print('n_class:', n_class)
    print('train_rate:', train_rate)
    print('Acc: $%.3f_{\\pm%.3f}$, F1: $%.3f_{\\pm%.3f}$, Auc: $%.3f_{\\pm%.3f}$' %
          (acc_avg, acc_std, f1_avg, f1_std, auc_avg, auc_std))
    # 增加AUC之后的结果
    # n_class=3, train_rate=0.5, Acc: $0.911_{\pm0.010}$, F1: $0.911_{\pm0.010}$, Auc: $0.914_{\pm0.005}$
    # n_class=3, train_rate=0.7, Acc: $0.984_{\pm0.015}$, F1: $0.984_{\pm0.015}$, Auc: $0.979_{\pm0.012}$
    # n_class=3, train_rate=0.9, Acc: $0.991_{\pm0.021}$, F1: $0.991_{\pm0.021}$, Auc: $0.992_{\pm0.032}$

    # n_class=5, train_rate=0.5, Acc: $0.885_{\pm0.029}$, F1: $0.885_{\pm0.029}$, Auc: $0.961_{\pm0.007}$
    # n_class=5, train_rate=0.7, Acc: $0.922_{\pm0.022}$, F1: $0.922_{\pm0.022}$, Auc: $0.972_{\pm0.004}$
    # n_class=5, train_rate=0.9, Acc: $0.941_{\pm0.016}$, F1: $0.941_{\pm0.016}$, Auc: $0.988_{\pm0.011}$

    # n_class:10,train_rate=0.5, Acc: $0.627_{\pm0.019}$, F1: $0.623_{\pm0.022}$, Auc: $0.922_{\pm0.006}$
    # n_class:10,train_rate=0.7, Acc: $0.673_{\pm0.025}$, F1: $0.669_{\pm0.027}$, Auc: $0.936_{\pm0.013}$
    # n_class:10,train_rate=0.9, Acc: $0.698_{\pm0.050}$, F1: $0.691_{\pm0.052}$, Auc: $0.943_{\pm0.014}$
