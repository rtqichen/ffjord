import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CNF"]


class HyperLinear(nn.Module):
    def __init__(self, input_shape, dim_out, hypernet_dim=64, activation=nn.Tanh, **unused_kwargs):
        super(HyperLinear, self).__init__()
        self.dim_in = input_shape[0]
        self.dim_out = dim_out
        self.params_dim = self.dim_in * self.dim_out + self.dim_out
        self._hypernet = nn.Sequential(
            nn.Linear(1, hypernet_dim), activation(), nn.Linear(hypernet_dim, self.params_dim)
        )

    def forward(self, t, x):
        params = self._hypernet(t.view(1, 1)).view(-1)
        b = params[:self.dim_out].view(1, self.dim_out)
        w = params[self.dim_out:].view(self.dim_in, self.dim_out)
        x = torch.matmul(x, w) + b
        return x


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
    def __init__(self, input_shape, dim_out, **unused_kwargs):
        super(BlendLinear, self).__init__()
        dim_in = input_shape[0]
        self._layer0 = nn.Linear(dim_in, dim_out)
        self._layer1 = nn.Linear(dim_in, dim_out)

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

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

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


class BlendConv2d(nn.Module):
    def __init__(
        self, input_shape, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False,
        **unused_kwargs
    ):
        super(BlendConv2d, self).__init__()
        assert len(input_shape) == 3, "input_shape expected to be of (C, H, W)."
        dim_in = input_shape[0]
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer0 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self._layer1 = module(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )
        self.input_shape = input_shape

    def forward(self, t, x):
        batchsize = x.shape[0]
        y0 = self._layer0(x.view(batchsize, *self.input_shape)).view(batchsize, -1)
        y1 = self._layer1(x.view(batchsize, *self.input_shape)).view(batchsize, -1)
        return y0 + (y1 - y0) * t


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(z, x, e=None):
    if e is None:
        e = torch.randn(z.shape)
    e_dzdx = torch.autograd.grad(z, x, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.sum(dim=1)
    return approx_tr_dzdx


class ODEfunc(nn.Module):
    def __init__(
        self, hidden_dims, input_shape, strides, conv, layer_type="concat", nonlinearity="softplus",
        divergence_fn="approximate"
    ):
        super(ODEfunc, self).__init__()
        assert layer_type in ("ignore", "hyper", "concat", "blend")
        assert divergence_fn in ("brute_force", "approximate")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")

        self.nonlinearity = {"tanh": F.tanh, "relu": F.relu, "softplus": F.softplus, "elu": F.elu}[nonlinearity]

        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {"ignore": IgnoreConv2d, "hyper": HyperConv2d,
                          "concat": ConcatConv2d, "blend": BlendConv2d}[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {"ignore": IgnoreLinear, "hyper": HyperLinear,
                          "concat": ConcatLinear, "blend": BlendLinear}[layer_type]

        # build layers and add them
        layers = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):

            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 3, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 3, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layers.append(base_layer(hidden_shape, dim_out, **layer_kwargs))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self._num_evals = 0

    def forward(self, t, y):
        # increment num evals
        self._num_evals += 1

        # to tensor
        t = torch.tensor(t).type_as(y)
        y = y[:, :-1]  # remove logp
        batchsize = y.shape[0]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dx = y
            for l, layer in enumerate(self.layers):
                dx = layer(t, dx)
                # if not last layer, use nonlinearity
                if l < len(self.layers) - 1:
                    dx = self.nonlinearity(dx)

            divergence = self.divergence_fn(dx, y, e=self._e).view(batchsize, 1)
        return torch.cat([dx, -divergence], 1)


class CNF(nn.Module):
    def __init__(
        self, hidden_dims, T, odeint, input_shape, strides=None, conv=False, layer_type="concat",
        divergence_fn="approximate", nonlinearity="tanh"
    ):
        super(CNF, self).__init__()
        self.odefunc = ODEfunc(
            hidden_dims=hidden_dims, input_shape=input_shape, strides=strides, conv=conv, layer_type=layer_type,
            divergence_fn=divergence_fn, nonlinearity=nonlinearity
        )
        self.input_shape = input_shape
        self.time_range = torch.tensor([0., float(T)])
        self.odeint = odeint
        self._num_evals = 0

    def forward(self, z, logpz=None, integration_times=None, reverse=False, full_output=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        assert z.shape[1] == np.prod(self.input_shape), "Input data shape {} different from input_shape {}.".format(
            z.shape[1], self.input_shape
        )
        inputs = torch.cat([z.view(z.shape[0], -1), _logpz.view(-1, 1)], 1)

        if integration_times is None:
            integration_times = self.time_range
        if reverse:
            integration_times = _flip(integration_times, 0)

        # fix noise throughout integration and reset counter to 0
        self.odefunc._e = torch.randn(z.shape).to(z.device)
        self.odefunc._num_evals = 0
        outputs = self.odeint(self.odefunc, inputs, integration_times.to(inputs), atol=1e-6, rtol=1e-5)
        z_t, logpz_t = outputs[:, :, :-1], outputs[:, :, -1:]

        if len(integration_times) == 2:
            z_t, logpz_t = z_t[1], logpz_t[1]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

    def num_evals(self):
        return self.odefunc._num_evals


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]
