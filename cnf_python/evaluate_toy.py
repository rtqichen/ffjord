import matplotlib
matplotlib.use('agg') # for linux server with no tkinter
import matplotlib.pyplot as plt
from matplotlib import rc

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

        test_loss = compute_loss(args, model, batch_size=args.batch_size)
        test_nfe = count_nfe(model)
        log_message = '[TEST]  | Test Loss {:.6f} | NFE {:.0f}'.format(test_loss, test_nfe)
        logger.info(log_message)

        nSamples = args.batch_size

        seed = np.random.RandomState(1)
        p_samples = toy_data.inf_train_gen(args.data, batch_size=nSamples, rng=seed)
        p_samples = torch.Tensor(p_samples).to(device)

        sample_fn, density_fn = get_transforms(model) # reverse of model, then forward model
        logger.info('check inverse')
        logger.info(torch.norm(density_fn(sample_fn(p_samples)) - p_samples).item() / p_samples.shape[0])

        LOW  = -5 # axes limits
        HIGH = 5
        plt.figure(figsize=(9, 9))
        plt.clf()
        ax = plt.subplot(2, 2, 1, aspect="equal")
        ax.hist2d(p_samples.detach().cpu().numpy()[:, 0], p_samples.detach().cpu().numpy()[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(r'$\mathbf{x} \sim \rho_0(\mathbf{x}) $')

        forw = density_fn(p_samples) # map to rho_1
        ax = plt.subplot(2, 2, 2, aspect="equal")
        ax.hist2d(forw.detach().cpu().numpy()[:, 0], forw.detach().cpu().numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(r'$f(\mathbf{x})$')

        invForw = sample_fn(forw) # back to original space....hopefully the function is actually invertible
        ax = plt.subplot(2, 2, 3, aspect="equal")
        ax.hist2d(invForw.detach().cpu().numpy()[:, 0], invForw.detach().cpu().numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(r'$f^{-1}(f(\mathbf{x}))$')

        # sample from rho_1 (the standard normal) and generate
        norm_samples = torch.randn(6000,2)
        norm_samples = norm_samples.to(device)

        genSamples = sample_fn(norm_samples) # back to original space....hopefully the function is actually invertible
        ax = plt.subplot(2, 2, 4, aspect="equal")
        ax.hist2d(genSamples.detach().cpu().numpy()[:, 0], genSamples.detach().cpu().numpy()[:, 1],range=[[LOW, HIGH], [LOW, HIGH]], bins=100)
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(r'$f^{-1}(\mathbf{y} \sim \rho_1(\mathbf{y}) )$')


        # save to h5 file
        import h5py
        with h5py.File('8gaussGenFFJORD.h5', 'w') as f:
            f.create_dataset('x',       data=p_samples.detach().cpu().numpy())
            f.create_dataset('fx',      data=forw.detach().cpu().numpy())
            f.create_dataset('finvfx',  data=invForw.detach().cpu().numpy())
            f.create_dataset('y',       data=norm_samples.detach().cpu().numpy())
            f.create_dataset('finvy',   data=genSamples.detach().cpu().numpy())


        sPath = "image/eval8gaussians.pdf"
        utils.makedirs(os.path.dirname(sPath))
        plt.savefig(sPath)
        logger.info('save image to ' + sPath)



