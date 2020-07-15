import matplotlib
matplotlib.use('agg') # for linux server with no tkinter
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim

import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

from diagnostics.viz_toy import save_trajectory, trajectory_to_video

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams', 'do']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS) # default='dopri5'
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=0.05, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--resume', type=str, default='experiments/cnf/toy/8gaussians/OD/checkpt.pth')
parser.add_argument('--save', type=str, default='experiments/cnf/toy')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn


def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log q(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


if __name__ == '__main__':

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 2, regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    logger.info(model)
    nWeights = count_parameters(model)
    logger.info("Number of trainable parameters: {}".format(nWeights))


    end = time.time()

    if args.resume is None:
        logger.info('must provide a checkpoint to resume')
        exit(1)
    checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])

    with torch.no_grad():
        model.eval()





        test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
        test_nfe = count_nfe(model)
        log_message = '[TEST]  | Test Loss {:.6f} | NFE {:.0f}'.format(test_loss, test_nfe)
        logger.info(log_message)

        nSamples = 1000 # 00

        seed = np.random.RandomState(1)
        p_samples = toy_data.inf_train_gen(args.data, batch_size=nSamples, rng=seed)
        p_samples = torch.Tensor(p_samples).to(device)
        # print(p_samples[0:3,:])






        sample_fn, density_fn = get_transforms(model) # reverse of model, then forward model
        logger.info('check inverse')
        invErr = torch.norm(density_fn(sample_fn(p_samples)) - p_samples).item() / p_samples.shape[0]
        logger.info('inverse: {:.3e}'.format(invErr))



        # intermediates / trajectories -------------------------------

        # visualize_transform(
        #     p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
        #     samples=True, npts=800, device=device
        # )


        # def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title="$x ~ q(x)$", device="cpu"):
        #
        # memory = 100
        # npts   = 100
        # device = "cpu"
        # transform = density_fn
        #
        # z = torch.randn(npts * npts, 2).type(torch.float32).to(device)
        # zk = []
        # inds = torch.arange(0, z.shape[0]).to(torch.int64)
        # for ii in torch.split(inds, int(memory ** 2)):
        #     zk.append(transform(z[ii]))
        # zk = torch.cat(zk, 0).cpu().numpy()

        # -------------------------------------------------------------

        # def save_trajectory(model, data_samples, savedir, ntimes=101, memory=0.01, device='cpu'):
        device = "cpu"
        ntimes = 101
        memory = 0.01
        model.eval()

        #  Sample from prior
        z_samples = torch.randn(10, 2).to(device)

        # sample from a grid
        npts = 5 # 800
        side = np.linspace(-4, 4, npts)
        xx, yy = np.meshgrid(side, side)
        xx = torch.from_numpy(xx).type(torch.float32).to(device)
        yy = torch.from_numpy(yy).type(torch.float32).to(device)
        z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

        with torch.no_grad():
            # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
            logp_samples = torch.sum(standard_normal_logprob(z_samples), 1, keepdim=True)
            logp_grid = torch.sum(standard_normal_logprob(z_grid), 1, keepdim=True)
            t = 0
            for cnf in model.chain:
                end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
                integration_times = torch.linspace(0, end_time, ntimes)

                z_traj, _ = cnf(z_samples, logp_samples, integration_times=integration_times, reverse=True)
                z_traj = z_traj.cpu().numpy()

                grid_z_traj, grid_logpz_traj = [], []
                inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
                for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        z_grid[ii], logp_grid[ii], integration_times=integration_times, reverse=True
                    )
                    _grid_z_traj, _grid_logpz_traj = _grid_z_traj.cpu().numpy(), _grid_logpz_traj.cpu().numpy()
                    grid_z_traj.append(_grid_z_traj)
                    grid_logpz_traj.append(_grid_logpz_traj)
                grid_z_traj = np.concatenate(grid_z_traj, axis=1)
                grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

        # -------------------------------------------------------------






        #
        # LOW  = -5 # axes limits
        # HIGH = 5
        # plt.figure(figsize=(9, 9))
        # plt.clf()
        # ax = plt.subplot(2, 2, 1, aspect="equal")
        # ax.hist2d(p_samples.numpy()[:, 0], p_samples.numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        # ax.invert_yaxis()
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        # ax.set_title(r'$\mathbf{x} \sim \rho_0(\mathbf{x}) $')

        forw = density_fn(p_samples) # map to rho_1
        # ax = plt.subplot(2, 2, 2, aspect="equal")
        # ax.hist2d(forw.numpy()[:, 0], forw.numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        # ax.invert_yaxis()
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        # ax.set_title(r'$f(\mathbf{x})$')

        invForw = sample_fn(forw) # back to original space....hopefully the function is actually invertible
        # ax = plt.subplot(2, 2, 3, aspect="equal")
        # ax.hist2d(invForw.numpy()[:, 0], invForw.numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        # ax.invert_yaxis()
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        # ax.set_title(r'$f^{-1}(f(\mathbf{x}))$')

        # sample from rho_1 (the standard normal) and generate
        norm_samples = torch.randn(nSamples,2)
        norm_samples = norm_samples.to(device)
        # print(norm_samples[0:3, :])

        genSamples = sample_fn(norm_samples) # back to original space....hopefully the function is actually invertible
        # ax = plt.subplot(2, 2, 4, aspect="equal")
        # ax.hist2d(genSamples.numpy()[:, 0], genSamples.numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        # ax.invert_yaxis()
        # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        # ax.set_title(r'$f^{-1}(\mathbf{y} \sim \rho_1(\mathbf{y}) )$')


        # save to h5 file
        import h5py
        with h5py.File('8gaussTestFFJORD.h5', 'w') as f:
            f.create_dataset('x',           data=p_samples.cpu().detach().numpy())
            f.create_dataset('fx',          data=forw.cpu().detach().numpy())
            f.create_dataset('finvfx',      data=invForw.cpu().detach().numpy())
            f.create_dataset('normSamples', data=norm_samples.cpu().detach().numpy())
            f.create_dataset('genSamples',  data=genSamples.cpu().detach().numpy())
            f.create_dataset('nWeights',    data=nWeights)
            # f.create_dataset('testTime',    data=timeMeter.sum)
            f.create_dataset('testBatchSize', data=nSamples)
            f.create_dataset('invErr',   data=invErr)

        # plt.savefig("../image/eval8gaussiansRk4Good.pdf")
        # plt.show()


