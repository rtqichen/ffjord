import torch
import torch.nn as nn


class CouplingLayer(nn.Module):

    def __init__(self, d, intermediate_dim=32, swap=False):
        nn.Module.__init__(self)
        self.d = d // 2
        self.swap = swap
        self.net_s_t = nn.Sequential(
            nn.Linear(self.d, intermediate_dim), nn.ReLU(inplace=True), nn.Linear(intermediate_dim, self.d * 2)
        )

    def forward(self, x, logpx=None, reverse=False):

        if self.swap:
            x = torch.cat([x[:, self.d:], x[:, :self.d]], 1)

        s_t = self.net_s_t(x[:, :self.d])
        scale = torch.sigmoid(s_t[:, :self.d] + 2.)
        shift = s_t[:, self.d:]

        logdetjac = torch.mean(torch.log(scale).view(scale.shape[0], -1), 1, keepdim=True)

        if not reverse:
            y1 = x[:, self.d:] * scale + shift
            delta_logp = -logdetjac
        else:
            y1 = (x[:, self.d:] - shift) / scale
            delta_logp = logdetjac

        y = torch.cat([x[:, :self.d], y1], 1) if not self.swap else torch.cat([y1, x[:, :self.d]], 1)

        if logpx is None:
            return y
        else:
            return y, logpx + delta_logp
