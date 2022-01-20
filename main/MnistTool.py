import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.io import savemat
import os


# 生成二分类数据集，用于CV
class MnistBagsBinaryCV:
    def __init__(self, path='../../Data/', po_num=0, mean_bag_length=10, var_bag_length=2, po_ins_range=(1, 4), seed=1,
                 total_bag_num=100, save_path='../../Data/Mnist-bags'):
        super(MnistBagsBinaryCV, self).__init__()
        print('Creating Binary Classification Dataset.')
        self.po_num = po_num
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.po_ins_range = np.arange(po_ins_range[0], po_ins_range[1])
        self.total_bag_num = total_bag_num
        self.seed = seed
        self.r = np.random.RandomState(seed)

        mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))]))
        mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))]))

        bags, labels = self.generate_bags(train_data=mnist_train, test_data=mnist_test, bag_num=self.total_bag_num)
        bags_labels = []
        for i in range(self.total_bag_num):
            temp = [bags[i], labels[0, i]]
            bags_labels.append(temp)
        print(bags_labels[0])
        print(len(bags_labels))
        bags_labels = np.array(bags_labels, dtype=object)
        # save as mat
        save_path = save_path + '/' + 'mnist_' + str(self.po_num) + '.mat'
        savemat(save_path, {'data': bags_labels})

    def generate_bags(self, train_data, test_data, bag_num):
        # Convert tensor to numpy array for storage and use
        mnist_photos, mnist_labels = [], []
        for i in tqdm(range(len(train_data)), desc='Loading mnist train set'):
            photo = train_data[i][0][0].detach().numpy()
            label = train_data[i][1]
            mnist_photos.append(photo)
            mnist_labels.append(label)

        for i in tqdm(range(len(test_data)), desc='Loading mnist test set'):
            photo = test_data[i][0][0].detach().numpy()
            label = test_data[i][1]
            mnist_photos.append(photo)
            mnist_labels.append(label)

        mnist_photos = np.array(mnist_photos)
        mnist_labels = np.array(mnist_labels)

        # Generate bags
        po_photo_index = np.argwhere(mnist_labels == self.po_num).flatten()
        ne_photo_index = np.argwhere(mnist_labels != self.po_num).flatten()
        # Generate positive bags
        po_bags = []
        po_labels = []
        for i in range(int(bag_num / 2)):
            # 此包包含的正图片数量
            temp_po_ins_num = self.r.choice(self.po_ins_range, replace=False)
            # 此包包含的正图片索引
            temp_po_ins_idx = self.r.choice(po_photo_index, size=temp_po_ins_num, replace=False)
            # 从训练的正图片索引中删除本次选出来的索引
            po_photo_index = np.setdiff1d(po_photo_index, temp_po_ins_idx)
            # 此包包含的正图片数据
            temp_po_ins = mnist_photos[temp_po_ins_idx]
            # 此包大小
            temp_bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # 此包包含的负图片数量
            temp_ne_ins_num = temp_bag_length - temp_po_ins_num
            # 此包的负图片索引
            temp_ne_ins_idx = self.r.choice(ne_photo_index, size=temp_ne_ins_num, replace=False)
            # 从训练的负图片中删除本次选出的索引
            ne_photo_index = np.setdiff1d(ne_photo_index, temp_ne_ins_idx)
            # 此包的负图片数据
            temp_ne_ins = mnist_photos[temp_ne_ins_idx]
            # 此包的总实例数据
            temp_total_ins = np.concatenate((temp_po_ins, temp_ne_ins), axis=0)
            # 此包的标签
            temp_label = 1
            # 打乱包内实例排列
            self.r.shuffle(temp_total_ins)
            po_bags.append(temp_total_ins)
            po_labels.append(temp_label)

        # Generate negative bags
        ne_bags = []
        ne_labels = []
        for i in range(bag_num - int(bag_num / 2)):
            # 此包内总实例数
            temp_bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # 此包包含的负实例索引
            temp_ne_ins_idx = self.r.choice(ne_photo_index, size=temp_bag_length, replace=False)
            # 去除选出来的索引
            ne_photo_index = np.setdiff1d(ne_photo_index, temp_ne_ins_idx)
            # 包内负实例数据
            temp_ne_ins = mnist_photos[temp_ne_ins_idx]
            temp_label = 0
            self.r.shuffle(temp_ne_ins)
            ne_bags.append(temp_ne_ins)
            ne_labels.append(temp_label)
        po_bags = np.array(po_bags, dtype=object)  # avoid warning
        ne_bags = np.array(ne_bags, dtype=object)

        # concate po bags and ne bags, then shuffle
        bags = np.concatenate((po_bags, ne_bags), axis=0)
        labels = np.concatenate((po_labels, ne_labels), axis=0)

        # shuffle and return
        # self.r.shuffle(bags)
        # self.r.shuffle(labels)
        labels = np.reshape(labels, (1, bag_num))
        return bags, labels


# 生成多分类数据集，与attention-net相同, 3, 5, 9为正, 其他为负, 也用于CV
class MnistBagsMulti:
    def __init__(self, path='../../Data/', po_nums=(3, 5, 9), mean_bag_length=10, var_bag_length=2, po_ins_range=(1, 4),
                 seed=1, total_bag_num=400, save_path='../../Data/Mnist-bags/Multi'):
        super(MnistBagsMulti, self).__init__()
        print('Creating Multiple Classification Dataset.')
        self.po_nums = po_nums
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.po_ins_range = np.arange(po_ins_range[0], po_ins_range[1])
        self.total_bag_num = total_bag_num
        # 分类数，包括负类
        self.class_num = len(self.po_nums) + 1
        # 每一类的包数
        self.each_class_bag_num = []
        for i in range(self.class_num):
            if i != self.class_num - 1:
                self.each_class_bag_num.append(int(self.total_bag_num / self.class_num))
            else:
                self.each_class_bag_num.append(self.total_bag_num - (self.class_num - 1)
                                               * (int(self.total_bag_num / self.class_num)))
        self.seed = seed
        self.r = np.random.RandomState(seed)

        mnist_train = datasets.MNIST(path, train=True, download=True, transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))]))
        mnist_test = datasets.MNIST(path, train=False, download=True, transform=transforms.Compose([
                                                                          transforms.ToTensor(),
                                                                          transforms.Normalize((0.1307,), (0.3081,))]))

        # Convert tensor to numpy array for storage and use
        self.mnist_photos, self.mnist_labels = [], []
        for i in tqdm(range(len(mnist_train)), desc='Loading mnist train set'):
            photo = mnist_train[i][0][0].detach().numpy()
            label = mnist_train[i][1]
            self.mnist_photos.append(photo)
            self.mnist_labels.append(label)

        for i in tqdm(range(len(mnist_test)), desc='Loading mnist test set'):
            photo = mnist_test[i][0][0].detach().numpy()
            label = mnist_test[i][1]
            self.mnist_photos.append(photo)
            self.mnist_labels.append(label)

        self.mnist_photos = np.array(self.mnist_photos)
        self.mnist_labels = np.array(self.mnist_labels)

        bags, labels = self.generate_bags()  # 生成包
        # print(labels)
        bags_labels = []
        for i in range(self.total_bag_num):
            temp = [bags[i], labels[i]]
            bags_labels.append(temp)
        # print(bags_labels[0])
        # print(len(bags_labels))
        bags_labels = np.array(bags_labels, dtype=object)
        trian_idx_list, test_idx_list = self.get_index(self.total_bag_num, para_k=5)
        # save as mat
        save_path = save_path + '/' + str(self.mean_bag_length) + '/'
        train_save_path = save_path + 'mnist_' + str(self.mean_bag_length) + '_train.mat'
        test_save_path = save_path + 'mnist_' + str(self.mean_bag_length) + '_test.mat'
        savemat(train_save_path, {'data': bags_labels[trian_idx_list[0]]})
        savemat(test_save_path, {'data': bags_labels[test_idx_list[0]]})

    def generate_bags(self):
        # 计算所有负图片的索引
        ne_idx = np.arange(len(self.mnist_labels))
        for i in self.po_nums:
            ne_idx = np.setdiff1d(ne_idx, np.argwhere(self.mnist_labels == i).flatten())
        # 生成所有正类的包
        bags, labels = [], []
        for i in range(self.class_num - 1):  # 类别数包含负类，所以减1
            # 当前作为正类的数字
            po_num = self.po_nums[i]
            # 正数字的索引
            po_idx = np.argwhere(self.mnist_labels == po_num).flatten()
            # 循环此类的包的数量，生成以temp_po_num为正实例的正包
            for j in range(self.each_class_bag_num[i]):
                # 此包内正图片数量
                temp_po_photo_num = self.r.choice(self.po_ins_range, 1, replace=False)
                # 此包正图片索引
                temp_po_photo_idx = self.r.choice(po_idx, temp_po_photo_num, replace=False)
                # 从总的正图片索引中删除上一步选出来的索引
                po_idx = np.setdiff1d(po_idx, temp_po_photo_idx)
                # 此包内正图片数据
                temp_po_photo = self.mnist_photos[temp_po_photo_idx]
                # 此包大小
                temp_bag_size = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
                # 此包负图片数量
                temp_ne_photo_num = temp_bag_size - temp_po_photo_num
                # 此包负图片索引
                temp_ne_photo_idx = self.r.choice(ne_idx, temp_ne_photo_num, replace=False)
                # 从总的负图片索引中删除上一步选出来的
                ne_idx = np.setdiff1d(ne_idx, temp_ne_photo_idx)
                # 此包内负图片数据
                temp_ne_photo = self.mnist_photos[temp_ne_photo_idx]
                # 总图片数据
                total_photo = np.concatenate((temp_po_photo, temp_ne_photo), axis=0)
                self.r.shuffle(total_photo)
                bags.append(total_photo)
                labels.append(i + 1)  # 因为i从0开始，所以加1

        # 生成所有负类包, 即label为0
        for i in range(self.each_class_bag_num[-1]):
            # 此负包大小
            temp_bag_size = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # 包内负图片索引
            temp_ne_photo_idx = self.r.choice(ne_idx, temp_bag_size, replace=False)
            # 包内负图片数据
            temp_ne_photo = self.mnist_photos[temp_ne_photo_idx]
            bags.append(temp_ne_photo)
            labels.append(0)
        return bags, labels

    def get_index(self, num_bags, para_k=10):
        temp_rand_idx = self.r.permutation(num_bags)
        temp_fold = int(np.ceil(num_bags / para_k))
        ret_tr_idx = {}
        ret_te_idx = {}
        for i in range(para_k):
            temp_tr_idx = temp_rand_idx[0: i * temp_fold].tolist()
            temp_tr_idx.extend(temp_rand_idx[(i + 1) * temp_fold:])
            ret_tr_idx[i] = temp_tr_idx
            ret_te_idx[i] = temp_rand_idx[i * temp_fold: (i + 1) * temp_fold].tolist()
        return ret_tr_idx, ret_te_idx


if __name__ == '__main__':
    MnistBagsMulti(po_nums=(3, 5, 9), total_bag_num=500, seed=66, mean_bag_length=100)