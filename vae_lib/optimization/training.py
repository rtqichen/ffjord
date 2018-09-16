from __future__ import print_function
import time
import torch

from vae_lib.optimization.loss import calculate_loss
from vae_lib.utils.visual_evaluation import plot_reconstructions
from vae_lib.utils.log_likelihood import calculate_likelihood

import numpy as np
from train_misc import count_nfe, override_divergence_fn


def train(epoch, train_loader, model, opt, args, logger):

    model.train()
    train_loss = np.zeros(len(train_loader))
    train_bpd = np.zeros(len(train_loader))

    num_data = 0

    # set warmup coefficient
    beta = min([(epoch * 1.) / max([args.warmup, 1.]), args.max_beta])
    logger.info('beta = {:5.4f}'.format(beta))
    end = time.time()
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()

        if args.dynamic_binarization:
            data = torch.bernoulli(data)

        data = data.view(-1, *args.input_size)

        opt.zero_grad()
        x_mean, z_mu, z_var, ldj, z0, zk = model(data)

        if 'cnf' in args.flow:
            f_nfe = count_nfe(model)

        loss, rec, kl, bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args, beta=beta)

        loss.backward()

        if 'cnf' in args.flow:
            t_nfe = count_nfe(model)
            b_nfe = t_nfe - f_nfe

        train_loss[batch_idx] = loss.item()
        train_bpd[batch_idx] = bpd

        opt.step()

        rec = rec.item()
        kl = kl.item()

        num_data += len(data)

        batch_time = time.time() - end
        end = time.time()

        if batch_idx % args.log_interval == 0:
            if args.input_type == 'binary':
                perc = 100. * batch_idx / len(train_loader)
                log_msg = (
                    'Epoch {:3d} [{:5d}/{:5d} ({:2.0f}%)] | Time {:.3f} | Loss {:11.6f} | '
                    'Rec {:11.6f} | KL {:11.6f}'.format(
                        epoch, num_data, len(train_loader.sampler), perc, batch_time, loss.item(), rec, kl
                    )
                )
            else:
                perc = 100. * batch_idx / len(train_loader)
                tmp = 'Epoch {:3d} [{:5d}/{:5d} ({:2.0f}%)] | Time {:.3f} | Loss {:11.6f} | Bits/dim {:8.6f}'
                log_msg = tmp.format(epoch, num_data, len(train_loader.sampler), perc, batch_time, loss.item(),
                                     bpd), '\trec: {:11.3f}\tkl: {:11.6f}'.format(rec, kl)
                log_msg = "".join(log_msg)
            if 'cnf' in args.flow:
                log_msg += ' | NFE Forward {} | NFE Backward {}'.format(f_nfe, b_nfe)
            logger.info(log_msg)

    if args.input_type == 'binary':
        logger.info('====> Epoch: {:3d} Average train loss: {:.4f}'.format(epoch, train_loss.sum() / len(train_loader)))
    else:
        logger.info(
            '====> Epoch: {:3d} Average train loss: {:.4f}, average bpd: {:.4f}'.
            format(epoch, train_loss.sum() / len(train_loader), train_bpd.sum() / len(train_loader))
        )

    return train_loss


def evaluate(data_loader, model, args, logger, testing=False, epoch=0):
    model.eval()
    loss = 0.
    batch_idx = 0
    bpd = 0.

    if args.input_type == 'binary':
        loss_type = 'elbo'
    else:
        loss_type = 'bpd'

    if testing and 'cnf' in args.flow:
        override_divergence_fn(model, "brute_force")

    for data, _ in data_loader:
        batch_idx += 1

        if args.cuda:
            data = data.cuda()

        with torch.no_grad():
            data = data.view(-1, *args.input_size)

            x_mean, z_mu, z_var, ldj, z0, zk = model(data)

            batch_loss, rec, kl, batch_bpd = calculate_loss(x_mean, data, z_mu, z_var, z0, zk, ldj, args)

            bpd += batch_bpd
            loss += batch_loss.item()

            # PRINT RECONSTRUCTIONS
            if batch_idx == 1 and testing is False:
                plot_reconstructions(data, x_mean, batch_loss, loss_type, epoch, args)

    loss /= len(data_loader)
    bpd /= len(data_loader)

    if testing:
        logger.info('====> Test set loss: {:.4f}'.format(loss))

    # Compute log-likelihood
    if testing and not ("cnf" in args.flow):  # don't compute log-likelihood for cnf models

        with torch.no_grad():
            test_data = data_loader.dataset.tensors[0]

            if args.cuda:
                test_data = test_data.cuda()

            logger.info('Computing log-likelihood on test set')

            model.eval()

            if args.dataset == 'caltech':
                log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, logger, S=2000, MB=500)
            else:
                log_likelihood, nll_bpd = calculate_likelihood(test_data, model, args, logger, S=5000, MB=500)

        if 'cnf' in args.flow:
            override_divergence_fn(model, args.divergence_fn)
    else:
        log_likelihood = None
        nll_bpd = None

    if args.input_type in ['multinomial']:
        bpd = loss / (np.prod(args.input_size) * np.log(2.))

    if testing and not ("cnf" in args.flow):
        logger.info('====> Test set log-likelihood: {:.4f}'.format(log_likelihood))

        if args.input_type != 'binary':
            logger.info('====> Test set bpd (elbo): {:.4f}'.format(bpd))
            logger.info(
                '====> Test set bpd (log-likelihood): {:.4f}'.
                format(log_likelihood / (np.prod(args.input_size) * np.log(2.)))
            )

    if not testing:
        return loss, bpd
    else:
        return log_likelihood, nll_bpd
