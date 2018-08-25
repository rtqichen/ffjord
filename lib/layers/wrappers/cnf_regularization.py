import torch
import torch.nn as nn


class RegularizedODEfunc(nn.Module):
    def __init__(self, odefunc, regularization_fns):
        super(RegularizedODEfunc, self).__init__()
        self.odefunc = odefunc
        self.regularization_fns = regularization_fns

    def before_odeint(self, *args, **kwargs):
        self.odefunc.before_odeint(*args, **kwargs)

    def forward(self, t, state):
        class SharedContext(object):
            pass

        with torch.enable_grad():
            x, logp = state[:2]
            x.requires_grad_(True)
            logp.requires_grad_(True)
            dstate = self.odefunc(t, (x, logp))
            if len(state) > 2:
                dx, dlogp = dstate[:2]
                reg_states = tuple(reg_fn(x, logp, dx, dlogp, SharedContext) for reg_fn in self.regularization_fns)
                return dstate + reg_states
            else:
                return dstate

    @property
    def _num_evals(self):
        return self.odefunc._num_evals


def _batch_root_mean_squared(tensor):
    tensor = tensor.view(tensor.shape[0], -1)
    return torch.mean(torch.norm(tensor, p=2, dim=1) / tensor.shape[1]**0.5)


def l1_regularzation_fn(x, logp, dx, dlogp, unused_context):
    del x, logp, dlogp
    return torch.mean(torch.abs(dx))


def l2_regularzation_fn(x, logp, dx, dlogp, unused_context):
    del x, logp, dlogp
    return _batch_root_mean_squared(dx)


def directional_l2_regularization_fn(x, logp, dx, dlogp, unused_context):
    del logp, dlogp
    directional_dx = torch.autograd.grad(dx, x, dx, create_graph=True)[0]
    return _batch_root_mean_squared(directional_dx)


def jacobian_frobenius_regularization_fn(x, logp, dx, dlogp, context):
    del logp, dlogp
    if hasattr(context, "jac"):
        jac = context.jac
    else:
        jac = _get_minibatch_jacobian(dx, x)
        context.jac = jac
    return _batch_root_mean_squared(jac)


def jacobian_diag_frobenius_regularization_fn(x, logp, dx, dlogp, context):
    del logp, dlogp
    if hasattr(context, "jac"):
        jac = context.jac
    else:
        jac = _get_minibatch_jacobian(dx, x)
        context.jac = jac
    diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]  # assumes jac is minibatch square, ie. (N, M, M).
    return _batch_root_mean_squared(diagonal)


def jacobian_offdiag_frobenius_regularization_fn(x, logp, dx, dlogp, context):
    del logp, dlogp
    if hasattr(context, "jac"):
        jac = context.jac
    else:
        jac = _get_minibatch_jacobian(dx, x)
        context.jac = jac
    diagonal = jac.view(jac.shape[0], -1)[:, ::jac.shape[1]]  # assumes jac is minibatch square, ie. (N, M, M).
    ss_offdiag = torch.sum(jac.view(jac.shape[0], -1)**2, dim=1) - torch.sum(diagonal**2, dim=1)
    ms_offdiag = ss_offdiag / (diagonal.shape[1] * (diagonal.shape[1] - 1))
    return torch.mean(ms_offdiag)


def _get_minibatch_jacobian(y, x, create_graph=False):
    """Computes the Jacobian of y wrt x assuming minibatch-mode.

    Args:
      y: (N, ...) with a total of D_y elements in ...
      x: (N, ...) with a total of D_x elements in ...
    Returns:
      The minibatch Jacobian matrix of shape (N, D_y, D_x)
    """
    assert y.shape[0] == x.shape[0]
    y = y.view(y.shape[0], -1)

    # Compute Jacobian row by row.
    jac = []
    for j in range(y.shape[1]):
        dy_j_dx = torch.autograd.grad(y[:, j], x, torch.ones_like(y[:, j]), retain_graph=True,
                                      create_graph=True)[0].view(x.shape[0], -1)
        jac.append(torch.unsqueeze(dy_j_dx, 1))
    jac = torch.cat(jac, 1)
    return jac
