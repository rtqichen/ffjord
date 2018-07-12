import torch
import torch.nn as nn
import torch.nn.functional as F

from . import resnet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.constant_(m.weight, 0)
        nn.init.normal_(m.bias, 0, 0.01)


class HyperLinear(nn.Module):
    def __init__(self, input_shape, dim_out, hypernet_dim=8, n_hidden=1, activation=nn.Tanh, **unused_kwargs):
        super(HyperLinear, self).__init__()
        self.dim_in = input_shape[0]
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out

        layers = []
        dims = [1] + [hypernet_dim] * n_hidden + [self.params_dim]
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if i < len(dims) - 1:
                layers.append(activation())
        self._hypernet = nn.Sequential(*layers)
        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(self.dim_out)
        w = params[self.dim_out:].view(self.dim_out, self.dim_in)
        return F.linear(x, w, b)


class IgnoreLinear(nn.Module):
    def __init__(self, input_shape, dim_out, **unused_kwargs):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(input_shape[0], dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, input_shape, dim_out, **unused_kwargs):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(input_shape[0] + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class BlendLinear(nn.Module):
    def __init__(self, input_shape, dim_out, **kwargs):
        super(BlendLinear, self).__init__()
        self._layer0 = nn.Linear(input_shape[0], dim_out)
        self._layer1 = nn.Linear(input_shape[0], dim_out)

    def forward(self, t, x):
        y0 = self._layer0(x)
        y1 = self._layer1(x)
        return y0 + (y1 - y0) * t


class HyperConv2d(nn.Module):
    def __init__(
        self, input_shape, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(HyperConv2d, self).__init__()
        assert len(input_shape) == 3, "input_shape expected to be of (C, H, W)."
        dim_in = input_shape[0]
        assert dim_in % groups == 0 and dim_out % groups == 0, "dim_in and dim_out must both be divisible by groups."
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.transpose = transpose

        self.params_dim = int(dim_in * dim_out * ksize * ksize / groups)
        if self.bias:
            self.params_dim += dim_out
        self._hypernet = nn.Linear(1, self.params_dim)
        self.conv_fn = F.conv_transpose2d if transpose else F.conv2d
        self.input_shape = input_shape

        self._hypernet.apply(weights_init)

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        weight_size = int(self.dim_in * self.dim_out * self.ksize * self.ksize / self.groups)
        if self.transpose:
            weight = params[:weight_size].view(self.dim_in, self.dim_out // self.groups, self.ksize, self.ksize)
        else:
            weight = params[:weight_size].view(self.dim_out, self.dim_in // self.groups, self.ksize, self.ksize)
        bias = params[:self.dim_out].view(self.dim_out) if self.bias else None
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        x = self.conv_fn(
            x, weight=weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups,
            dilation=self.dilation
        )
        return x.view(batchsize, -1)


class IgnoreConv2d(nn.Module):
    def __init__(
        self, input_shape, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(IgnoreConv2d, self).__init__()
        assert len(input_shape) == 3, "input_shape expected to be of (C, H, W)."
        dim_in = input_shape[0]
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self.input_shape = input_shape

    def forward(self, t, x):
        batchsize = x.shape[0]
        return self._layer(x.view(batchsize, *self.input_shape)).view(batchsize, -1)


class ConcatConv2d(nn.Module):
    def __init__(
        self, input_shape, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(ConcatConv2d, self).__init__()
        assert len(input_shape) == 3, "input_shape expected to be of (C, H, W)."
        dim_in = input_shape[0]
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self.input_shape = input_shape

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx).view(batchsize, -1)


class IgnoreResNet(nn.Module):
    def __init__(self, dim, intermediate_dim, n_resblocks):
        super(IgnoreResNet, self).__init__()

        layers = []
        layers.append(nn.Conv2d(dim, intermediate_dim, kernel_size=1, bias=False))
        layers.append(nn.GroupNorm(2, intermediate_dim, eps=1e-4))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(n_resblocks):
            layers.append(resnet.BasicBlock(intermediate_dim))
        layers.append(nn.Conv2d(intermediate_dim, dim, kernel_size=1, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        del t
        return self.net(x)
