import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ["CNF"]


class HyperLinear(nn.Module):
    def __init__(self, dim_in, dim_out, hypernet_dim=64, activation=nn.Tanh):
        super(HyperLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.params_dim = dim_in * dim_out + dim_out
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
    def __init__(self, dim_in, dim_out, **kwargs):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)

    def forward(self, t, x):
        return self._layer(x)


class ConcatLinear(nn.Module):
    def __init__(self, dim_in, dim_out, **kwargs):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Linear(dim_in + 1, dim_out)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class IgnoreConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(IgnoreLinear, self).__init__()
        self._layer = nn.Conv2d(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        return self._layer(x)


class ConcatConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs):
        super(ConcatLinear, self).__init__()
        self._layer = nn.Conv2d(
            dim_in, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


def divergence_bf(dx, y, **kwargs):
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
    def __init__(self, dims, layer_type="concat", layer_kwargs={}, nonlinearity="tanh", divergence_fn="approximate"):
        super(ODEfunc, self).__init__()
        assert layer_type in ("ignore", "hyper", "concat")
        assert divergence_fn in ("brute_force", "approximate")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")
        assert len(dims) >= 2
        self.nonlinearity = {"tanh": F.tanh, "relu": F.relu, "softplus": F.softplus, "elu": F.elu}[nonlinearity]
        base_layer = {"ignore": IgnoreLinear, "hyper": HyperLinear, "concat": ConcatLinear}[layer_type]
        # build layers and add them
        self.layers = []
        for i in range(1, len(dims)):
            dim_out = dims[i]
            dim_in = dims[i - 1]
            self.layers.append(base_layer(dim_in, dim_out, **layer_kwargs))
        self.layers.append(base_layer(dims[-1], dims[0], **layer_kwargs))
        for idx, layer in enumerate(self.layers):
            self.add_module(str(idx), layer)

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self._num_evals = 0

    def forward(self, t, y):
        # increiment num evals
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
    def __init__(self, dims, T, odeint, layer_type="concat", divergence_fn="approximate", nonlinearity="tanh"):
        super(CNF, self).__init__()
        self.odefunc = ODEfunc(dims, layer_type=layer_type, divergence_fn=divergence_fn, nonlinearity=nonlinearity)
        self.time_range = torch.tensor([0., float(T)])
        self.odeint = odeint
        self._num_evals = 0

    def forward(self, z, logpz=None, integration_times=None, reverse=False, full_output=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        inputs = torch.cat([z, _logpz.view(-1, 1)], 1)

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


if __name__ == "__main__":

    dim_in = 10
    dim_h = 10
    batch_size = 16

    net = nn.Sequential(
        nn.Linear(dim_in, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_in)
    )

    x = torch.randn(batch_size, dim_in, requires_grad=True)
    z = net(x)
    div_bf = divergence_bf(z, x)
    print(div_bf.mean())
    n_samples = 100
    da = 0.
    ss = []
    for i in range(n_samples):
        div = divergence_approx(z, x)
        da += div
        ss.append(div.mean().detach().numpy())

    da = da / n_samples
    print(da.mean())
    print(np.var(ss, axis=0))
