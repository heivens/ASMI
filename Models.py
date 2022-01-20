from Layers import AttentionLayerBag, SelfAttentionLayer, Attention
from torch.nn import Module, Linear, Dropout, Sequential, LeakyReLU, BatchNorm1d, ReLU, Tanh, Sigmoid
import torch
import torch.nn.functional as F


# 定义聚合包信息再聚合实例信息的网络
class Net(Module):
    def __init__(self, ins_len, out_len, n_class, drop_rate=0.2):
        super(Net, self).__init__()
        # 包信息传播网络
        self.bag_info = Sequential(
            AttentionLayerBag(ins_len=ins_len, out_len=out_len),
            LeakyReLU(negative_slope=0.2),

            Linear(in_features=out_len, out_features=int(out_len / 2)),
            LeakyReLU(negative_slope=0.2),

            Dropout(p=drop_rate),

            Linear(in_features=int(out_len / 2), out_features=n_class),
            LeakyReLU(negative_slope=0.2),
        )

        # 实例信息传播
        self.ins_info = Sequential(
            Linear(in_features=ins_len, out_features=int(ins_len / 2)),
            LeakyReLU(negative_slope=0.2),

            Dropout(p=drop_rate),

            Linear(in_features=int(ins_len / 2), out_features=int(ins_len / 4)),
            LeakyReLU(negative_slope=0.2),

            Linear(in_features=int(ins_len / 4), out_features=n_class),
            LeakyReLU(negative_slope=0.2),
        )

        # 包信息与实例信息聚合
        self.agg_info = Sequential(
            SelfAttentionLayer(ins_len=n_class, out_len=2 * n_class),
            LeakyReLU(negative_slope=0.2),
            # 最后一层不用softmax，因为pytorch交叉熵损失函数会自动将输入softmax
            Linear(in_features=2 * n_class, out_features=n_class)
        )

    def forward(self, bag):
        bag = bag.float()
        bag_info = self.bag_info(bag)
        ins_info = self.ins_info(bag)
        # 信息拼接
        info = torch.cat((bag_info, ins_info), dim=0)
        y = self.agg_info(info)
        return y


# 定义功能同上，但包先过一层bn1的网络
class NetWithBN(Module):
    def __init__(self, ins_len, out_len, n_class, drop_rate=0.2):
        super(NetWithBN, self).__init__()
        # 包信息传播网络
        self.bag_info = Sequential(
            BatchNorm1d(num_features=ins_len),
            AttentionLayerBag(ins_len=ins_len, out_len=out_len),
            LeakyReLU(negative_slope=0.2),

            Linear(in_features=out_len, out_features=int(out_len / 2)),
            LeakyReLU(negative_slope=0.2),

            Dropout(p=drop_rate),

            Linear(in_features=int(out_len / 2), out_features=n_class),
            LeakyReLU(negative_slope=0.2),
        )

        # 实例信息传播
        self.ins_info = Sequential(
            BatchNorm1d(num_features=ins_len),
            Linear(in_features=ins_len, out_features=int(ins_len / 2)),
            LeakyReLU(negative_slope=0.2),

            Dropout(p=drop_rate),

            Linear(in_features=int(ins_len / 2), out_features=int(ins_len / 4)),
            LeakyReLU(negative_slope=0.2),

            Linear(in_features=int(ins_len / 4), out_features=n_class),
            LeakyReLU(negative_slope=0.2)
        )

        # 包信息与实例信息聚合
        self.agg_info = Sequential(
            SelfAttentionLayer(ins_len=n_class, out_len=2 * n_class),
            LeakyReLU(negative_slope=0.2),
            # 最后一层不用softmax，因为pytorch交叉熵损失函数会自动将输入softmax
            Linear(in_features=2 * n_class, out_features=n_class)
        )

    def forward(self, bag):
        bag = bag.float()
        bag_info = self.bag_info(bag)
        ins_info = self.ins_info(bag)
        # 信息拼接
        info = torch.cat((bag_info, ins_info), dim=0)
        y = self.agg_info(info)
        return y


# 定义用attention-net的网络聚合包内信息的总网络
class NetWithAttention(Module):
    def __init__(self, ins_len, n_class, drop_rate=0.2):
        super(NetWithAttention, self).__init__()
        # 包信息传播网络
        self.bag_info = Attention(ins_len=ins_len, n_class=n_class)

        # 实例信息传播
        self.ins_info = Sequential(
            Linear(in_features=ins_len, out_features=int(ins_len / 2)),
            LeakyReLU(negative_slope=0.2),

            Dropout(p=drop_rate),

            Linear(in_features=int(ins_len / 2), out_features=int(ins_len / 4)),
            LeakyReLU(negative_slope=0.2),

            Linear(in_features=int(ins_len / 4), out_features=n_class),
            LeakyReLU(negative_slope=0.2),
        )

        # 包信息与实例信息聚合
        self.agg_info = Sequential(
            SelfAttentionLayer(ins_len=n_class, out_len=2 * n_class),
            LeakyReLU(negative_slope=0.2),
            # 最后一层不用softmax，因为pytorch交叉熵损失函数会自动将输入softmax
            Linear(in_features=2 * n_class, out_features=n_class)
        )

    def forward(self, bag):
        bag = bag.float()
        bag_info = self.bag_info(bag)
        ins_info = self.ins_info(bag)
        # 信息拼接
        info = torch.cat((bag_info, ins_info), dim=0)
        y = self.agg_info(info)
        return y


if __name__ == '__main__':
    input_ = torch.randn(3, 166)
    net = NetWithAttention(ins_len=166, n_class=2, drop_rate=0.2)
    out = net(input_)
    print(out)
    print(out.shape)
