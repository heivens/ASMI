import numpy as np
from MILFrame.MIL import MIL
from tqdm import tqdm
from sklearn.metrics import euclidean_distances as eucl


def get_index(num_bags=92, para_k=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    temp_rand_idx = np.random.permutation(num_bags)

    temp_fold = int(np.ceil(num_bags / para_k))
    ret_tr_idx = {}
    ret_te_idx = {}
    for i in range(para_k):
        temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
        temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
        ret_tr_idx[i] = temp_tr_idx
        ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
    return ret_tr_idx, ret_te_idx


# 加载包含包标签和实例标签的数据集
def load_data(path='../Data/Benchmark/musk1+.mat'):
    mil = MIL(para_path=path)
    labels = mil.bags_label
    bag_labels = np.where(labels < 0, 0, labels)
    bags = []
    ins_labels = []
    for i in range(mil.num_bags):
        bags.append(mil.bags[i, 0][:, :-1])
        ins_labels.append(mil.bags[i, 0][:, -1])
    return bags, bag_labels


# 返回多分类的包和标签
def load_multi_class_bag(path):
    mil = MIL(para_path=path)
    bags = []
    for i in range(mil.num_bags):
        bags.append(mil.bags[i, 0][:, :-1])
    lables = mil.bags_label
    return np.array(bags, dtype=object), np.array(lables)


def compute_discer(vectors, labels):
    positive_vectors, negative_vectors = [], []
    for i in range(len(vectors)):
        if labels[i] == 1:
            positive_vectors.append(vectors[i])
        elif labels[i] == 0:
            negative_vectors.append(vectors[i])
    positive_vectors = np.array(positive_vectors)
    negative_vectors = np.array(negative_vectors)
    # 均值向量
    positive_mean = np.mean(positive_vectors, axis=0)
    negative_mean = np.mean(negative_vectors, axis=0)
    # 平均距离
    positive_dis = np.mean(eucl(positive_vectors), axis=None)
    negative_dis = np.mean(eucl(negative_vectors), axis=None)
    fenmu = positive_dis + negative_dis
    return eucl([positive_mean], [negative_mean])[0][0] / fenmu  # if fenmu > 1e-3 else 1e-3


if __name__ == '__main__':
    # bags, bag_label = load_data(path='../Data/Benchmark/musk1+.mat')
    # print(bags[0])
    # print(bag_label[0])
    vectors_1 = np.random.randn(2, 3) + 1
    vectors_2 = np.random.randn(2, 3) - 1
    labels = np.array([0, 1, 1, 0])
    # for i in range(len(vectors)):
    #     print(vectors[i], end='  ')
    #     print(labels[i])
    print(compute_discer(np.concatenate((vectors_1, vectors_2), axis=0), labels))

