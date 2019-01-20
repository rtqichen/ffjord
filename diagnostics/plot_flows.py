from inspect import getsourcefile
import sys
import os

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os

import torch

import lib.toy_data as toy_data
import lib.utils as utils
import lib.visualize_flow as viz_flow
import lib.layers.odefunc as odefunc
import lib.layers as layers

from train_misc import standard_normal_logprob
from train_misc import build_model_tabular, count_parameters

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)

parser.add_argument('--discrete', action='store_true')

parser.add_argument('--depth', help='number of coupling layers', type=int, default=10)
parser.add_argument('--glow', type=eval, choices=[True, False], default=False)

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

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--niters', type=int, default=2500)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

parser.add_argument('--checkpt', type=str, required=True)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def construct_discrete_model():

    chain = []
    for i in range(args.depth):
        if args.glow: chain.append(layers.BruteForceLayer(2))
        chain.append(layers.CouplingLayer(2, swap=i % 2 == 0))
    return layers.SequentialFlow(chain)


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


if __name__ == '__main__':

    if args.discrete:
        model = construct_discrete_model().to(device)
        model.load_state_dict(torch.load(args.checkpt)['state_dict'])
    else:
        model = build_model_tabular(args, 2).to(device)

        sd = torch.load(args.checkpt)['state_dict']
        fixed_sd = {}
        for k, v in sd.items():
            fixed_sd[k.replace('odefunc.odefunc', 'odefunc')] = v
        model.load_state_dict(fixed_sd)

    print(model)
    print("Number of trainable parameters: {}".format(count_parameters(model)))

    model.eval()
    p_samples = toy_data.inf_train_gen(args.data, batch_size=800**2)

    with torch.no_grad():
        sample_fn, density_fn = get_transforms(model)

        plt.figure(figsize=(10, 10))
        ax = ax = plt.gca()
        viz_flow.plt_samples(p_samples, ax, npts=800)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_filename = os.path.join(args.save, 'figs', 'true_samples.jpg')
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = ax = plt.gca()
        viz_flow.plt_flow_density(standard_normal_logprob, density_fn, ax, npts=800, memory=200, device=device)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_filename = os.path.join(args.save, 'figs', 'model_density.jpg')
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()

        plt.figure(figsize=(10, 10))
        ax = ax = plt.gca()
        viz_flow.plt_flow_samples(torch.randn, sample_fn, ax, npts=800, memory=200, device=device)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig_filename = os.path.join(args.save, 'figs', 'model_samples.jpg')
        utils.makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()
