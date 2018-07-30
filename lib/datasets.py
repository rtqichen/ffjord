import torch


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
        return x, 0


class CelebA(Dataset):
    TRAIN_LOC = 'data/celeba/celeba_train.pth'
    VAL_LOC = 'data/celeba/celeba_val.pth'

    def __init__(self, train=True, transform=None):
        return super(CelebA, self).__init__(self.TRAIN_LOC if train else self.VAL_LOC, transform)
