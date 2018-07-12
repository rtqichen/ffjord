import torch.nn as nn

from . import basic
from . import container


class ConcatResNet(container.SequentialDiffEq):
    def __init__(self, dim, intermediate_dim, n_resblocks):
        super(ConcatResNet, self).__init__()

        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks

        layers = []
        layers.append(basic.ConcatConv2d(dim, intermediate_dim, ksize=1, bias=False))
        layers.append(nn.GroupNorm(2, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_resblocks):
            layers.append(ConcatBasicBlock(intermediate_dim))
        layers.append(basic.ConcatConv2d(intermediate_dim, dim, ksize=1, bias=False))

        super(ConcatResNet, self).__init__(*layers)

    def extra_repr(self):
        return 'dim={}, idim={}, n_resblocks={}'.format(self.dim, self.intermediate_dim, self.n_resblocks)


class ConcatBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super(ConcatBasicBlock, self).__init__()
        self.conv1 = basic.ConcatConv2d(dim, dim, ksize=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(2, dim, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = basic.ConcatConv2d(dim, dim, ksize=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(2, dim, eps=1e-4)

    def forward(self, t, x):
        residual = x

        out = self.conv1(t, x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(t, out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out
