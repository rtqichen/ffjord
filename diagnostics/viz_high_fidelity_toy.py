import os
import math
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_density_traj(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu'):
    model.eval()

    # sample from a grid
    npts = 800
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logpz_grid = torch.sum(standard_normal_logprob(z_grid), 1, keepdim=True)
        for cnf in model.chain:
            end_time = cnf.sqrt_end_time * cnf.sqrt_end_time
            viz_times = torch.linspace(0., end_time, ntimes)

            logpz_grid = [standard_normal_logprob(z_grid).sum(1, keepdim=True)]
            for t in tqdm(viz_times[1:]):
                inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
                logpz_t = []
                for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                    z0, delta_logp = cnf(
                        z_grid[ii],
                        torch.zeros(z_grid[ii].shape[0], 1).to(z_grid), integration_times=torch.tensor([0.,
                                                                                                        t.item()])
                    )
                    logpz_t.append(standard_normal_logprob(z0).sum(1, keepdim=True) - delta_logp)
                logpz_grid.append(torch.cat(logpz_t, 0))
            logpz_grid = torch.stack(logpz_grid, 0).cpu().detach().numpy()
            z_grid = z_grid.cpu().detach().numpy()

            plt.figure(figsize=(8, 8))
            for t in range(logpz_grid.shape[0]):

                plt.clf()
                ax = plt.gca()

                # plot the density
                z, logqz = z_grid, logpz_grid[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                plt.pcolormesh(xx, yy, qz, cmap='binary')
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap('binary')
                ax.set_axis_bgcolor(cmap(0.))
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                plt.tight_layout()

                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))


def trajectory_to_video(savedir):
    import subprocess
    bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


if __name__ == '__main__':
    import argparse
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

    import lib.toy_data as toy_data
    from train_misc import count_parameters
    from train_misc import set_cnf_options, add_spectral_norm, create_regularization_fns
    from train_misc import build_model_tabular

    def get_ckpt_model_and_data(args):
        # Load checkpoint.
        checkpt = torch.load(args.checkpt, map_location=lambda storage, loc: storage)
        ckpt_args = checkpt['args']
        state_dict = checkpt['state_dict']

        # Construct model and restore checkpoint.
        regularization_fns, regularization_coeffs = create_regularization_fns(ckpt_args)
        model = build_model_tabular(ckpt_args, 2, regularization_fns).to(device)
        if ckpt_args.spectral_norm: add_spectral_norm(model)
        set_cnf_options(ckpt_args, model)

        model.load_state_dict(state_dict)
        model.to(device)

        print(model)
        print("Number of trainable parameters: {}".format(count_parameters(model)))

        # Load samples from dataset
        data_samples = toy_data.inf_train_gen(ckpt_args.data, batch_size=2000)

        return model, data_samples

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', type=str, required=True)
    parser.add_argument('--ntimes', type=int, default=101)
    parser.add_argument('--memory', type=float, default=0.01, help='Higher this number, the more memory is consumed.')
    parser.add_argument('--save', type=str, default='trajectory')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, data_samples = get_ckpt_model_and_data(args)
    save_density_traj(model, data_samples, args.save, ntimes=args.ntimes, memory=args.memory, device=device)
    trajectory_to_video(args.save)
