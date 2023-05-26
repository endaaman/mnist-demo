import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTDataset(Dataset):
    def __init__(self, limit, train):
        ds = MNIST(root='data/',
                   train=train,
                   download=True)
        self.data = ds.data[:limit]
        self.labels = ds.targets[:limit]

    def __len__(self):
        # return len(self.data)
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        label = self.labels[idx]
        x = x / 255
        x = x[None, :, :]
        return x, label


if __name__ == '__main__':
    ds = MNISTDataset(train=True)
