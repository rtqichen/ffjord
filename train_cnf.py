import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim

import integrate

import lib.models as models
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons'], type=str,
                    default='moons')
parser.add_argument('--dims', type=str, default='2,64,64,10')
parser.add_argument('--time_length', type=float, default=1.0)

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--batch-size', type=int, default=200)
parser.add_argument('--lr-max', type=float, default=5e-4)
parser.add_argument('--lr-min', type=float, default=1e-4)
parser.add_argument('--lr-interval', type=float, default=2000)
parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def update_lr(optimizer, itr):
    lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1 + np.cos(itr / args.niters * np.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    _odeint = integrate.odeint_adjoint if args.adjoint else integrate.odeint
    dims = tuple(map(int, args.dims.split(',')))
    cnf = models.CNF(dims=dims, T=args.time_length, odeint=_odeint)
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        cnf.load_state_dict(checkpt['state_dict'])
    cnf.to(device)

    optimizer = optim.Adam(cnf.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    end = time.time()
    best_loss = float('inf')
    for itr in range(1, args.niters + 1):
        update_lr(optimizer, itr)
        optimizer.zero_grad()

        # load data
        x = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
        x = torch.from_numpy(x).type(torch.float32).to(device)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # backward to get z (standard normal)
        z, delta_logp = cnf(x, zero, reverse=True)

        # compute log q(z)
        logpz = standard_normal_logprob(z).sum(1, keepdim=True)

        # forward to get logpx
        # x_rec, logpx = cnf(z, logpz)
        # _logpx = logpz - delta_logp
        # print(torch.mean(torch.abs(_logpx - logpx)).item())
        # print(torch.mean(torch.abs(x - x_rec)).item())

        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)
        loss.backward()

        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        print('Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(itr, time_meter.val, time_meter.avg,
                                                                               loss_meter.val, loss_meter.avg))

        if loss.item() < best_loss:
            best_loss = loss.item()
            utils.makedirs(args.save)
            torch.save({
                'args': args,
                'state_dict': cnf.state_dict(),
            }, os.path.join(args.save, 'checkpt.pth'))

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                p_samples = toy_data.inf_train_gen(args.data, batch_size=10000)

                plt.figure(figsize=(9, 3))
                visualize_transform(p_samples, torch.randn, standard_normal_logprob, cnf, samples=True, device=device)
                fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(itr))
                utils.makedirs(os.path.dirname(fig_filename))
                plt.savefig(fig_filename)

        end = time.time()
