import torch
import torchvision.datasets as vdsets


class Dataset(object):
    def __init__(self, loc, transform=None):
        self.dataset = torch.load(loc).float().div(255)
        self.transform = transform

    def __len__(self):
        return self.dataset.size(0)

    @property
    def ndim(self):
        return self.dataset.size(1)

    def __getitem__(self, index):
        x = self.dataset[index]
        x = self.transform(x) if self.transform is not None else x
        return x


class MNIST(object):
    def __init__(self, dataroot, train=True, transform=None):
        self.mnist = vdsets.MNIST(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.mnist)

    @property
    def ndim(self):
        return 1

    def __getitem__(self, index):
        return self.mnist[index][0]


class CIFAR10(object):
    def __init__(self, dataroot, train=True, transform=None):
        self.cifar10 = vdsets.CIFAR10(dataroot, train=train, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar10)

    @property
    def ndim(self):
        return 3

    def __getitem__(self, index):
        return self.cifar10[index][0]


class CelebA(Dataset):
    TRAIN_LOC = 'data/celeba/celeba_train.pth'
    VAL_LOC = 'data/celeba/celeba_val.pth'

    def __init__(self, train=True, transform=None):
        return super(CelebA, self).__init__(self.TRAIN_LOC if train else self.VAL_LOC, transform)
