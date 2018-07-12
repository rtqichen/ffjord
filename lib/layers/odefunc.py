import torch
import torch.nn as nn

from .diffeq_layers import diffeq_wrapper

__all__ = ["ODEfunc", "AutoencoderODEfunc"]


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
        self.diffeq = diffeq_wrapper(diffeq)

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals = 0

    def forward(self, t, y):
        # increment num evals
        self._num_evals += 1

        # convert to tensor
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
