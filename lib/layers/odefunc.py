import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import diffeq_layers
from .diffeq_layers import diffeq_wrapper

__all__ = ["ODEnet", "ODEfunc", "AutoencoderODEfunc"]


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    if e is None:
        e = torch.randn(f.shape)
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx


class ODEnet(nn.Module):
    """
    Helper class to make neural nets for use in continuous normalizing flows
    """

    def __init__(self, hidden_dims, input_shape, strides, conv, layer_type="concat", nonlinearity="softplus"):
        super(ODEnet, self).__init__()
        assert layer_type in ("ignore", "hyper", "concat", "blend")
        assert nonlinearity in ("tanh", "relu", "softplus", "elu")

        self.nonlinearity = {"tanh": F.tanh, "relu": F.relu, "softplus": F.softplus, "elu": F.elu}[nonlinearity]
        if conv:
            assert len(strides) == len(hidden_dims) + 1
            base_layer = {
                "ignore": diffeq_layers.IgnoreConv2d,
                "hyper": diffeq_layers.HyperConv2d,
                "concat": diffeq_layers.ConcatConv2d,
                "blend": diffeq_layers.BlendConv2d,
            }[layer_type]
        else:
            strides = [None] * (len(hidden_dims) + 1)
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "concat": diffeq_layers.ConcatLinear,
                "blend": diffeq_layers.BlendLinear,
            }[layer_type]

        # build layers and add them
        layers = []
        hidden_shape = input_shape

        for dim_out, stride in zip(hidden_dims + (input_shape[0],), strides):
            if stride is None:
                layer_kwargs = {}
            elif stride == 1:
                layer_kwargs = {"ksize": 3, "stride": 1, "padding": 1, "transpose": False}
            elif stride == 2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": False}
            elif stride == -2:
                layer_kwargs = {"ksize": 4, "stride": 2, "padding": 1, "transpose": True}
            else:
                raise ValueError('Unsupported stride: {}'.format(stride))

            layers.append(base_layer(hidden_shape[0], dim_out, **layer_kwargs))

            hidden_shape = list(copy.copy(hidden_shape))
            hidden_shape[0] = dim_out
            if stride == 2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] // 2, hidden_shape[2] // 2
            elif stride == -2:
                hidden_shape[1], hidden_shape[2] = hidden_shape[1] * 2, hidden_shape[2] * 2

        self.layers = nn.ModuleList(layers)

    def forward(self, t, y):
        dx = y
        for l, layer in enumerate(self.layers):
            dx = layer(t, dx)
            # if not last layer, use nonlinearity
            if l < len(self.layers) - 1:
                dx = self.nonlinearity(dx)
        return dx


class ODEfunc(nn.Module):
    def __init__(self, input_shape, diffeq, divergence_fn="approximate"):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        self.input_shape = input_shape
        self.diffeq = diffeq_wrapper(diffeq)

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals = 0

    def forward(self, t, y_and_logpy):
        y, _ = y_and_logpy  # remove logpy

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        t = torch.tensor(t).type_as(y)
        batchsize = y.shape[0]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)
            divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)

        return dy, -divergence


class AutoencoderODEfunc(nn.Module):
    def __init__(self, input_shape, diffeq_encoder, diffeq_decoder, divergence_fn="approximate"):
        assert divergence_fn in ("brute_force", "approximate")
        self.input_shape = input_shape
        self.diffeq_encoder = diffeq_wrapper(diffeq_encoder)
        self.diffeq_decoder = diffeq_wrapper(diffeq_decoder)

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals = 0

    def forward(self, t, y):
        self._num_evals += 1
        t = torch.tensor(t).type_as(y)
        y = y[:, :-1]  # remove logp
        batchsize = y.shape[0]
        y = y.view(batchsize, *self.input_shape)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            h = self.diffeq_encoder(t, y)
            dy = self.diffeq_decoder(t, h)

            # Estimate divergence.
            if self._e is None:
                self._e = torch.randn(h.shape).to(h)

            e_vjp_dhdy = torch.autograd.grad(h, y, self._e, retain_graph=True)
            e_vjp_dfdy = torch.autograd.grad(dy, h, e_vjp_dhdy, retain_graph=True)
            divergence = torch.sum((e_vjp_dfdy * self._e).view(batchsize, -1), 1, keepdim=True)

            return torch.cat([dy.view(batchsize, -1), -divergence], 1)
