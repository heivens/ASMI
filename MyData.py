from torch.utils.data import Dataset, DataLoader, Subset
from utils import load_data, get_index


# 返回包与其标签的DataSet类
class MyDataSet(Dataset):
    def __init__(self, path='../MILFrame/data/benchmark/musk1+.mat'):
        self.bags, self.labels = load_data(path=path)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    All = MyDataSet(path='../Data/Benchmark/musk1+.mat')
    print(All[:][1])