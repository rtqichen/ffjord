import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitTransform(nn.Module):
    """
    The proprocessing step used in Real NVP:
    y = sigmoid(x) - a / (1 - 2a)
    x = logit(a + (1 - 2a)*y)
    """

    def __init__(self, alpha=0.05):
        nn.Module.__init__(self)
        self.alpha = alpha

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._logit(x, logpx)
        else:
            return self._sigmoid(x, logpx)

    def _logit(self, x, logpx=None):
        s = self.alpha + (1 - 2 * self.alpha) * x
        y = torch.log(s) - torch.log(1 - s)
        if logpx is None:
            return y
        return y, logpx - self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True)

    def _sigmoid(self, y, logpy=None):
        x = (F.sigmoid(y) - self.alpha) / (1 - 2 * self.alpha)
        if logpy is None:
            return x
        return x, logpy + self._logdetgrad(x).view(x.size(0), -1).sum(1, keepdim=True)

    def _logdetgrad(self, x):
        s = self.alpha + (1 - 2 * self.alpha) * x
        logdetgrad = -torch.log(s - s * s) + math.log(1 - 2 * self.alpha)
        return logdetgrad
