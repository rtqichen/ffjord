import torch.nn as nn

from . import basic
from . import container


class ResNet(container.SequentialDiffEq):
    def __init__(self, dim, intermediate_dim, n_resblocks, conv_block=None):
        super(ResNet, self).__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.dim = dim
        self.intermediate_dim = intermediate_dim
        self.n_resblocks = n_resblocks

        layers = []
        layers.append(conv_block(dim, intermediate_dim, ksize=1, bias=False))
        layers.append(nn.GroupNorm(2, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_resblocks):
            layers.append(BasicBlock(intermediate_dim, conv_block))
        layers.append(conv_block(intermediate_dim, dim, ksize=1, bias=False))

        super(ResNet, self).__init__(*layers)

    def __repr__(self):
        return (
            '{name}({dim}, intermediate_dim={intermediate_dim}, n_resblocks={n_resblocks})'.format(
                name=self.__class__.__name__, **self.__dict__
            )
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, dim, conv_block=None):
        super(BasicBlock, self).__init__()

        if conv_block is None:
            conv_block = basic.ConcatCoordConv2d

        self.conv1 = conv_block(dim, dim, ksize=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(2, dim, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_block(dim, dim, ksize=3, padding=1, bias=False)
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
