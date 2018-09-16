from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from vae_lib.utils.distributions import log_normal_diag, log_normal_standard, log_bernoulli
import torch.nn.functional as F


def binary_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    reconstruction_function = nn.BCELoss(size_average=False)

    batch_size = x.size(0)

    # - N E_q0 [ ln p(x|z_k) ]
    bce = reconstruction_function(recon_x, x)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = torch.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = torch.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = bce + beta * kl

    loss /= float(batch_size)
    bce /= float(batch_size)
    kl /= float(batch_size)

    return loss, bce, kl


def multinomial_loss_function(x_logit, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    num_classes = 256
    batch_size = x.size(0)

    x_logit = x_logit.view(batch_size, num_classes, args.input_size[0], args.input_size[1], args.input_size[2])

    # make integer class labels
    target = (x * (num_classes - 1)).long()

    # - N E_q0 [ ln p(x|z_k) ]
    # sums over batch dimension (and feature dimension)
    ce = cross_entropy(x_logit, target, size_average=False)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = torch.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = torch.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = ce + beta * kl

    loss /= float(batch_size)
    ce /= float(batch_size)
    kl /= float(batch_size)

    return loss, ce, kl


def binary_loss_array(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss without averaging or summing over the batch dimension.
    """

    batch_size = x.size(0)

    # if not summed over batch_dimension
    if len(ldj.size()) > 1:
        ldj = ldj.view(ldj.size(0), -1).sum(-1)

    # TODO: upgrade to newest pytorch version on master branch, there the nn.BCELoss comes with the option
    # reduce, which when set to False, does no sum over batch dimension.
    bce = -log_bernoulli(x.view(batch_size, -1), recon_x.view(batch_size, -1), dim=1)
    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    #  ln q(z_0) - ln p(z_k) ]
    logs = log_q_z0 - log_p_zk

    loss = bce + beta * (logs - ldj)

    return loss


def multinomial_loss_array(x_logit, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Computes the discritezed logistic loss without averaging or summing over the batch dimension.
    """

    num_classes = 256
    batch_size = x.size(0)

    x_logit = x_logit.view(batch_size, num_classes, args.input_size[0], args.input_size[1], args.input_size[2])

    # make integer class labels
    target = (x * (num_classes - 1)).long()

    # - N E_q0 [ ln p(x|z_k) ]
    # computes cross entropy over all dimensions separately:
    ce = cross_entropy(x_logit, target, size_average=False, reduce=False)
    # sum over feature dimension
    ce = ce.view(batch_size, -1).sum(dim=1)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k.view(batch_size, -1), dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(
        z_0.view(batch_size, -1), mean=z_mu.view(batch_size, -1), log_var=z_var.log().view(batch_size, -1), dim=1
    )

    #  ln q(z_0) - ln p(z_k) ]
    logs = log_q_z0 - log_p_zk

    loss = ce + beta * (logs - ldj)

    return loss


def cross_entropy(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    r"""
    Taken from the master branch of pytorch, accepts (N, C, d_1, d_2, ..., d_K) input shapes
    instead of only (N, C, d_1, d_2) or (N, C).
    This criterion combines `log_softmax` and `nll_loss` in a single
    function.
    See :class:`~torch.nn.CrossEntropyLoss` for details.
    Args:
        input: Variable :math:`(N, C)` where `C = number of classes`
        target: Variable :math:`(N)` where each value is
            `0 <= targets[i] <= C-1`
        weight (Tensor, optional): a manual rescaling weight given to each
                class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
                over observations for each minibatch. However, if the field
                sizeAverage is set to False, the losses are instead summed
                for each minibatch. Ignored if reduce is False. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
                and does not contribute to the input gradient. When size_average is
                True, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): By default, the losses are averaged or summed over
                observations for each minibatch depending on size_average. When reduce
                is False, returns a loss per batch element instead and ignores
                size_average. Default: ``True``
    """
    return nll_loss(F.log_softmax(input, 1), target, weight, size_average, ignore_index, reduce)


def nll_loss(input, target, weight=None, size_average=True, ignore_index=-100, reduce=True):
    r"""
    Taken from the master branch of pytorch, accepts (N, C, d_1, d_2, ..., d_K) input shapes
    instead of only (N, C, d_1, d_2) or (N, C).
    The negative log likelihood loss.
    See :class:`~torch.nn.NLLLoss` for details.
    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K > 1`
            in the case of K-dimensional loss.
        target: :math:`(N)` where each value is `0 <= targets[i] <= C-1`,
            or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K >= 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): By default, the losses are averaged
            over observations for each minibatch. If size_average
            is False, the losses are summed for each minibatch. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            True, the loss is averaged over non-ignored targets. Default: -100
    """
    dim = input.dim()
    if dim == 2:
        return F.nll_loss(
            input, target, weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce
        )
    elif dim == 4:
        return F.nll_loss(
            input, target, weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce
        )
    elif dim == 3 or dim > 4:
        n = input.size(0)
        c = input.size(1)
        out_size = (n,) + input.size()[2:]
        if target.size()[1:] != input.size()[2:]:
            raise ValueError('Expected target size {}, got {}'.format(out_size, input.size()))
        input = input.contiguous().view(n, c, 1, -1)
        target = target.contiguous().view(n, 1, -1)
        if reduce:
            _loss = nn.NLLLoss2d(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce)
            return _loss(input, target)
        out = F.nll_loss(
            input, target, weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce
        )
        return out.view(out_size)
    else:
        raise ValueError('Expected 2 or more dimensions (got {})'.format(dim))


def calculate_loss(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Picks the correct loss depending on the input type.
    """

    if args.input_type == 'binary':
        loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)
        bpd = 0.

    elif args.input_type == 'multinomial':
        loss, rec, kl = multinomial_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args, beta=beta)
        bpd = loss.data[0] / (np.prod(args.input_size) * np.log(2.))

    else:
        raise ValueError('Invalid input type for calculate loss: %s.' % args.input_type)

    return loss, rec, kl, bpd


def calculate_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args):
    """
    Picks the correct loss depending on the input type.
    """

    if args.input_type == 'binary':
        loss = binary_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj)

    elif args.input_type == 'multinomial':
        loss = multinomial_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args)

    else:
        raise ValueError('Invalid input type for calculate loss: %s.' % args.input_type)

    return loss
