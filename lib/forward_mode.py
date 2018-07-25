"""
Utilities for forward mode autodiff in pytorch
"""
import torch
import torch.nn as nn

def forward_gradients(f, t, u=None):
    if u is None:
        u = torch.ones_like(t)
    v = torch.ones_like(f, requires_grad=True)
    g = torch.autograd.grad(f, t, v, create_graph=True)[0]
    return torch.autograd.grad(g, v, u, create_graph=True)[0]

if __name__ == "__main__":
    # finite difference params
    eps = 1e-6
    atol = 1e-5
    rtol = 1e-3

    # model params
    dim_in = 10
    dim_out = 24
    net = nn.Sequential(nn.Linear(dim_in, dim_out), nn.Tanh(), nn.Linear(dim_out, dim_out), nn.Tanh()).double()
    batch_size = 13
    for i in range(10):
        t = torch.randn((batch_size, 1), requires_grad=True).double()
        x = torch.randn((batch_size, dim_in - 1), requires_grad=True).double()
        inp = torch.cat([t, x], 1)
        # get output and analytic grads
        f = net(inp)
        dzdt_analytic = forward_gradients(f, t)
        # get numeric gradients
        t_plus = t + (eps / 2)
        t_minus = t - (eps / 2)
        inp_plus = torch.cat([t_plus, x], 1)
        inp_minus = torch.cat([t_minus, x], 1)
        f_plus = net(inp_plus)
        f_minus = net(inp_minus)
        dzdt_numeric = (f_plus - f_minus) / eps

        diff = (dzdt_analytic - dzdt_numeric).abs()
        if (diff <= (atol + rtol * dzdt_numeric.abs())).all():
            print("Pass")
        else:
            print("Fail")
