from inspect import signature
import torch
import torch.nn as nn

__all__ = ["diffeq_wrapper", "reshape_wrapper", "fourier_wrapper"]


class DiffEqWrapper(nn.Module):
    def __init__(self, module):
        super(DiffEqWrapper, self).__init__()
        self.module = module
        if len(signature(self.module.forward).parameters) == 1:
            self.diffeq = lambda t, y: self.module(y)
        elif len(signature(self.module.forward).parameters) == 2:
            self.diffeq = self.module
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")

    def forward(self, t, y):
        return self.diffeq(t, y)

    def __repr__(self):
        return self.diffeq.__repr__()


def diffeq_wrapper(layer):
    return DiffEqWrapper(layer)


class ReshapeDiffEq(nn.Module):
    def __init__(self, input_shape, net):
        super(ReshapeDiffEq, self).__init__()
        assert len(signature(net.forward).parameters) == 2, "use diffeq_wrapper before reshape_wrapper."
        self.input_shape = input_shape
        self.net = net

    def forward(self, t, x):
        batchsize = x.shape[0]
        x = x.view(batchsize, *self.input_shape)
        return self.net(t, x).view(batchsize, -1)

    def __repr__(self):
        return self.diffeq.__repr__()


def reshape_wrapper(input_shape, layer):
    return ReshapeDiffEq(input_shape, layer)


class FourierWrapper(nn.Module):
    def __init__(self, net):
        super(FourierWrapper, self).__init__()
        assert len(signature(net.forward).parameters) == 2, "use diffeq_wrapper before reshape_wrapper."
        self.net = net

    def forward(self, t, x):
        hidden = torch.rfft(x, 2, onesided=True)
        b, c, f_h, f_w, _ = hidden.shape
        hidden = hidden.permute(0, 1, 4, 2, 3).view(b, c * 2, f_h, f_w)
        hidden = self.net(t, hidden).view(b, c, 2, f_h, f_w).permute(0, 1, 3, 4, 2)
        hidden = torch.irfft(hidden, 2, onesided=True, signal_sizes=x.shape[-2:])
        return hidden


def fourier_wrapper(layer):
    return FourierWrapper(layer)
