from __future__ import print_function
import torch
import torch.utils.data

import math

MIN_EPSILON = 1e-5
MAX_EPSILON = 1. - 1e-5

PI = torch.FloatTensor([math.pi])
if torch.cuda.is_available():
    PI = PI.cuda()

# N(x | mu, var) = 1/sqrt{2pi var} exp[-1/(2 var) (x-mean)(x-mean)]
# log N(x| mu, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -(x - mean) * (x - mean)
    log_norm *= torch.reciprocal(2. * log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * torch.log(2. * PI)

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_bernoulli(x, mean, average=False, reduce=True, dim=None):
    probs = torch.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if reduce:
        if average:
            return torch.mean(log_bern, dim)
        else:
            return torch.sum(log_bern, dim)
    else:
        return log_bern
