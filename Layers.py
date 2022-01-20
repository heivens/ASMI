import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import Module, Linear, Dropout, Sequential, LeakyReLU, BatchNorm1d, ReLU, Tanh, Sigmoid


# 定义传统注意力网络层，聚合包内实例
class AttentionLayerBag(Module):
    def __init__(self, ins_len, out_len):
        super(AttentionLayerBag, self).__init__()
        self.ascend_dim = Linear(in_features=ins_len, out_features=512)
        self.reduct_dim = Linear(in_features=512, out_features=out_len)
        self.compute_e = Linear(in_features=out_len, out_features=1)

    def forward(self, bag):
        bag = bag.float()
        bag = F.relu(self.ascend_dim(bag))
        bag = torch.tanh(self.reduct_dim(bag))
        # 计算初始注意力系数e
        e_list = self.compute_e(bag)
        e_list = torch.reshape(e_list, (1, e_list.shape[0]))
        e_list = torch.nn.LeakyReLU(0.2)(e_list)
        alpha_list = F.softmax(e_list, dim=1)
        vector = torch.mm(alpha_list, bag)
        return vector


# 定义自注意力网络层，聚合包和实例的信息
class SelfAttentionLayer(Module):
    def __init__(self, ins_len=166, out_len=166):
        super(SelfAttentionLayer, self).__init__()
        self.W = nn.Parameter(nn.Parameter(xavier_normal_(torch.zeros(ins_len, out_len))))
        self.compute_e = nn.Parameter(nn.Parameter(xavier_normal_(torch.zeros(2 * out_len, 1))))

    def forward(self, bag):
        bag = bag.float()
        bag = torch.nn.LeakyReLU(0.2)(torch.mm(bag, self.W))
        center_ins = torch.reshape(bag[0], (1, bag[0].shape[0]))
        center_ins_matrix = center_ins.repeat(bag.shape[0], 1)  # 将center_ins在第0个维度上复制三次，第1个维度上只复制一次
        self_neighbors = torch.cat((center_ins_matrix, bag), dim=1)
        self_neighbors = self_neighbors.float()
        e_list = torch.mm(self_neighbors, self.compute_e)
        e_list = torch.reshape(e_list, (1, e_list.shape[0]))  # e_list reshape为1*3
        e_list = torch.nn.LeakyReLU(0.2)(e_list)
        alpha_list = F.softmax(e_list, dim=1)
        aggrgated_ins = torch.matmul(alpha_list, bag)  # 聚合后的单向量
        return aggrgated_ins


# attention-net的网络，聚合包内信息
class Attention(Module):
    def __init__(self, ins_len, n_class):
        super(Attention, self).__init__()
        self.num_att = ins_len
        self.L = 500
        self.D = ins_len
        self.K = 1

        self.feature_extractor_part = Sequential(
            Linear(self.num_att, self.L),
            ReLU(),
        )

        self.attention = Sequential(
            Linear(self.L, self.D),
            Tanh(),
            Linear(self.D, self.K)
        )

        self.classifier = Sequential(
            Linear(self.L*self.K, n_class),
        )

    def forward(self, x):

        H = self.feature_extractor_part(x)  # 输出为500

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)
        x = self.classifier(M)
        return x


if __name__ == '__main__':
    input_ = torch.randn(3, 166)
    layer = AttentionLayerBag(ins_len=166, out_len=166 * 2)
    # layer = SelfAttentionLayer(ins_len=2, out_len=2 * 2)
    out = layer(input_)
    print(out)
    print(out.shape)
