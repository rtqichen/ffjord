from inspect import signature

import torch
import torch.nn as nn

__all__ = ["ODEfunc", "wrap_diffeq"]


class _WrapDiffEq(nn.Module):
    def __init__(self, diffeq):
        super(_WrapDiffEq, self).__init__()
        self.diffeq = diffeq

    def forward(self, t, y):
        if len(signature(self.diffeq.forward).parameters) == 1:
            return self.diffeq(y)
        elif len(signature(self.diffeq.forward).parameters) == 2:
            return self.diffeq(t, y)
        else:
            raise ValueError("Differential equation needs to either take (t, y) or (y,) as input.")


def wrap_diffeq(diffeq):
    return _WrapDiffEq(diffeq)


class MixtureODELayer(nn.Module):
    def __init__(self, experts):
        super(MixtureODELayer, self).__init__()
        assert len(experts) > 1
        wrapped_experts = [wrap_diffeq(ex) for ex in experts]
        self.experts = nn.ModuleList(wrapped_experts)
        self.mixture_weights = nn.Linear(1, len(self.experts))

    def forward(self, t, y):
        dys = []
        for f in self.experts:
            dys.append(f(t, y))
        dys = torch.stack(dys, 0)
        weights = self.mixture_weights(t).view(-1, *([1] * (dys.ndimension() - 1)))

        dy = torch.sum(dys * weights, dim=0, keepdim=False)
        return dy


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


class ODEfunc(nn.Module):
    def __init__(self, input_shape, diffeq, divergence_fn="approximate"):
        super(ODEfunc, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        self.input_shape = input_shape
        self.diffeq = wrap_diffeq(diffeq)

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

        self._e = None
        self._num_evals = 0

    def forward(self, t, y):
        # increment num evals
        self._num_evals += 1

        # to tensor
        t = torch.tensor(t).type_as(y)
        y = y[:, :-1]  # remove logp
        batchsize = y.shape[0]
        y = y.view(batchsize, *self.input_shape)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t, y)
            divergence = self.divergence_fn(dy, y, e=self._e).view(batchsize, 1)

        return torch.cat([dy.view(batchsize, -1), -divergence], 1)
