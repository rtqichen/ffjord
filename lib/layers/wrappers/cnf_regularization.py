import torch
import torch.nn as nn


class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc, regularization_fn):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fn = regularization_fn

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        with torch.enable_grad():
            x, logp = state[:2]
            x.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.odefunc(t, (x, logp))
            dx, dlogp = dstate[:2]
            reg_state = self.regularization_fn(x, logp, dx, dlogp)
            return dstate + (reg_state,)

    @property
    def _num_evals(self):
        return self.odefunc._num_evals


def l2_regularzation_fn(x, logp, dx, dlogp):
    return torch.mean(dx**2)


def directional_l2_regularization_fn(x, logp, dx, dlogp):
    directional_dy = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    return torch.mean(directional_dy**2)
