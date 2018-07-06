import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['CNF']


class CNFBottleneck(nn.Module):
    def __init__(self, prev_dim, hidden_dim, activation_fn=F.tanh):
        super(CNFBottleneck, self).__init__()
        self.prev_dim = prev_dim
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn

        self.fc_layer = nn.Linear(prev_dim, hidden_dim)

    def forward(self, elem_x):
        """
        Returns:
            outputs (batch_size, next_dim)
            elem_grad_inputs (hidden_dim, batch_size, prev_dim)
            elem_outputs (hidden_dim, batch_size, 1)
        """
        prev_dim, hidden_dim = self.prev_dim, self.hidden_dim
        w, b = self.fc_layer.weight, self.fc_layer.bias

        # (hidden_dim, batch_size, 1)
        elem_outputs = self.activation_fn(
            torch.matmul(elem_x, w.view(hidden_dim, prev_dim, 1)) + b.view(hidden_dim, 1, 1))

        # (hidden_dim, batch_size, prev_dim)
        # each elem_grad_input[j] is dh_j/dx
        elem_grad_inputs = torch.autograd.grad(elem_outputs.sum(), elem_x, create_graph=True)[0]
        # (batch_size, hidden_dim)
        outputs = elem_outputs.permute([2, 1, 0]).view(elem_x.shape[1], self.hidden_dim)

        return outputs, elem_grad_inputs, elem_outputs


class Identity(nn.Module):
    def forward(self, t, x):
        return x


# standard mlp
def MLP(dims, activation_fn=nn.Tanh):
    args = []
    for i in range(1, len(dims)):
        din, dout = dims[i - 1], dims[i]
        args.append(nn.Linear(din, dout))
        args.append(activation_fn())
    return nn.Sequential(*args)


class HypernetMLP(nn.Module):
    def __init__(self, dims, hypernet_dim=64, activation_fn=F.tanh):
        super(HypernetMLP, self).__init__()
        assert len(dims) >= 2
        self.dims = dims
        layer_weights = []
        weights_idx = 0
        for i in range(len(dims) - 1):
            layer_weights.append((i, weights_idx))
            weights_idx += (dims[i] + 1) * dims[i + 1]
        self.layer_weights = layer_weights
        self.num_weights = weights_idx
        self._hypernet = nn.Sequential(
            nn.Linear(1, hypernet_dim),
            nn.Tanh(),
            nn.Linear(hypernet_dim, self.num_weights), )
        self.activation_fn = activation_fn

    def forward(self, t, x):
        weights = self._hypernet(torch.tensor(t).type_as(x).view(1, 1)).view(-1)
        for i, weights_idx in self.layer_weights:
            in_dim, out_dim = self.dims[i], self.dims[i + 1]
            w_offset = in_dim * out_dim
            b_offset = out_dim
            w = weights[weights_idx:weights_idx + w_offset].view(in_dim, out_dim)
            b = weights[weights_idx + w_offset:weights_idx + w_offset + b_offset].view(1, out_dim)

            x = torch.matmul(x, w) + b
            if i < len(self.layer_weights) - 1:
                x = self.activation_fn(x)
        return x


class ODEfunc(nn.Module):

    def __init__(self, dims):
        super(ODEfunc, self).__init__()
        assert len(dims) >= 2
        self.net1 = Identity() if len(dims) == 2 else HypernetMLP(dims=dims[:-1])
        self.bottleneck = CNFBottleneck(dims[-2], dims[-1])
        # TODO: use forward-mode autodiff to allow arbitrary output networks.
        self.net2 = nn.Linear(dims[-1], dims[0])

        self.input_dim = dims[0]
        self.bottleneck_dim = dims[-1]

    def forward(self, t, y):
        y = y[:, :-1]  # remove logp
        batchsize = y.shape[0]

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            Y = y[None].expand((self.bottleneck_dim, ) + y.shape)
            H = self.net1(t, Y)
            prehidden = H
            h, elem_grad_prehidden, _ = self.bottleneck(H)
            dx = self.net2(h)

            # TODO: replace & generalize this with forward-mode autodiff.
            grad_outputs = self.net2.weight.t().contiguous().view(self.bottleneck_dim, self.input_dim, 1)
            grad_inputs = torch.autograd.grad(prehidden, Y, elem_grad_prehidden, create_graph=True)[0]
            divergence = torch.matmul(grad_inputs, grad_outputs).sum(0).view(batchsize, 1)

        # sum_diag = 0.
        # for i in range(y.shape[1]):
        #     sum_diag += torch.autograd.grad(dx[:, i].sum(), y, retain_graph=True)[0][:, i]
        # print(torch.mean(torch.abs(sum_diag - divergence.view(-1))))

        return torch.cat([dx, -divergence], 1)


class CNF(nn.Module):

    def __init__(self, dims, T, odeint):
        super(CNF, self).__init__()
        self.odefunc = ODEfunc(dims)
        self.time_range = torch.tensor([0., float(T)])
        self.odeint = odeint

    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        inputs = torch.cat([z, _logpz.view(-1, 1)], 1)

        if integration_times is None:
            integration_times = self.time_range
        if reverse:
            integration_times = _flip(integration_times, 0)

        outputs = self.odeint(self.odefunc, inputs, integration_times.to(inputs), atol=1e-6, rtol=1e-5)
        z_t, logpz_t = outputs[:, :, :-1], outputs[:, :, -1:]

        if len(integration_times) == 2:
            z_t, logpz_t = z_t[1], logpz_t[1]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


if __name__ == "__main__":
    def divergence_bf(z, x):
        nin = x.shape[1]
        zs = z.sum(dim=0)
        div = 0.
        for i in range(nin):
            dzidxi = torch.autograd.grad(zs[i], x, create_graph=True)[0][:, i]
            div += dzidxi
        return div

    def divergence_approx(z, x):
        e = torch.randn(z.shape)
        e_dzdx = torch.autograd.grad(z, x, e, create_graph=True)[0]
        #print(e.shape)
        #print(e_dzdx.shape)
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.sum(dim=1)
        #print(approx_tr_dzdx)
        return approx_tr_dzdx

    dim_in = 10
    dim_h = 10
    batch_size = 16

    net = nn.Sequential(nn.Linear(dim_in, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_in))

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
