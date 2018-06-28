import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import math
import numpy as np

import torch
import integrate

import lib.models as models
import lib.toy_data as toy_data
import lib.utils as utils

assert (__name__ == '__main__')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpt', type=str, required=True)
parser.add_argument('--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons'], type=str,
                    default='moons')
parser.add_argument('--dims', type=str, default='2,64,64,10')
parser.add_argument('--time_length', type=float, default=1.0)

parser.add_argument('--ntimes', type=int, default=101)
parser.add_argument('--save', type=str, default='trajectory')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


if __name__ == '__main__':
    checkpt = torch.load(args.checkpt)
    ckpt_args = checkpt['args']
    args.dims = ckpt_args.dims
    args.time_length = ckpt_args.time_length
    args.data = ckpt_args.data

    dims = tuple(map(int, args.dims.split(',')))
    cnf = models.CNF(dims=dims, T=args.time_length, odeint=integrate.odeint).to(device)
    cnf.load_state_dict(checkpt['state_dict'])
    cnf.to(device)

    integration_times = torch.linspace(0, args.time_length, args.ntimes)

    with torch.no_grad():

        # sample from the prior
        z = torch.randn(40000, 2).to(device)
        logpz = torch.sum(standard_normal_logprob(z), 1)
        z_traj, logpz_traj = cnf(z, logpz, integration_times=integration_times)

        # sample from a grid
        npts = 800
        side = np.linspace(-4, 4, npts)
        xx, yy = np.meshgrid(side, side)
        xx = torch.from_numpy(xx).type(torch.float32).to(device)
        yy = torch.from_numpy(yy).type(torch.float32).to(device)
        grid_z = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)
        logpz = torch.sum(standard_normal_logprob(grid_z), 1)
        grid_z_traj, grid_logpz_traj = cnf(grid_z, logpz, integration_times=integration_times)

        data_samples = toy_data.inf_train_gen(args.data, batch_size=10000)
        z_traj, unused_logpz_traj = z_traj.cpu().numpy(), logpz_traj.cpu().numpy()
        grid_z_traj, grid_logpz_traj = grid_z_traj.cpu().numpy(), grid_logpz_traj.cpu().numpy()

        plt.figure(figsize=(16, 4))
        for t in range(z_traj.shape[0]):

            plt.clf()

            # plot target potential function
            ax = plt.subplot(1, 4, 1, aspect='equal')

            ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title('Target', fontsize=32)

            # plot the density
            ax = plt.subplot(1, 4, 2, aspect='equal')

            z, logqz = grid_z_traj[t], grid_logpz_traj[t]

            xx = z[:, 0].reshape(npts, npts)
            yy = z[:, 1].reshape(npts, npts)
            qz = np.exp(logqz).reshape(npts, npts)

            plt.pcolormesh(xx, yy, qz)
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            cmap = matplotlib.cm.get_cmap(None)
            ax.set_axis_bgcolor(cmap(0.))
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title('Density', fontsize=32)

            # plot the samples
            ax = plt.subplot(1, 4, 3, aspect='equal')

            zk = z_traj[t]
            ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title('Samples', fontsize=32)

            # plot vector field
            ax = plt.subplot(1, 4, 4, aspect='equal')

            K = 13j
            y, x = np.mgrid[-4:4:K, -4:4:K]
            K = int(K.imag)
            combined_inputs = np.concatenate([np.stack([x, y], -1).reshape(K * K, 2), np.zeros((K * K, 1))], 1)
            combined_inputs = torch.from_numpy(combined_inputs).type(torch.float32).to(device)
            dydt = cnf.odefunc(integration_times[t], combined_inputs)
            dydt = dydt.cpu().numpy().reshape(-1, 3)[:, :2]
            dydt = dydt.reshape(K, K, 2)

            logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
            ax.quiver(x, y, dydt[:, :, 0], dydt[:, :, 1],
                      np.exp(logmag), cmap='coolwarm', scale=10., width=0.015, pivot="mid")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.axis('off')
            ax.set_title('Vector Field', fontsize=32)

            utils.makedirs(args.save)
            plt.savefig(os.path.join(args.save, f'viz-{t:05d}.jpg'))
