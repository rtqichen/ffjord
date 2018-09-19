import torch
import torch.nn as nn
import torch.nn.functional as F


class BruteForceLayer(nn.Module):

    def __init__(self, dim):
        super(BruteForceLayer, self).__init__()
        self.weight = nn.Parameter(torch.eye(dim))

    def forward(self, x, logpx=None, reverse=False):

        if not reverse:
            y = F.linear(x, self.weight)
            if logpx is None:
                return y
            else:
                return y, logpx - self._logdetgrad.expand_as(logpx)

        else:
            y = F.linear(x, self.weight.double().inverse().float())
            if logpx is None:
                return y
            else:
                return y, logpx + self._logdetgrad.expand_as(logpx)

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.weight.double()))).float()
