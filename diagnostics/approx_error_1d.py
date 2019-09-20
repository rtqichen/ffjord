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

import numpy as np
import argparse
import os
import time

import torch
import torch.optim as optim

import lib.utils as utils
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import build_model_tabular

import seaborn as sns
sns.set_style("whitegrid")
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
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

parser.add_argument('--save', type=str, default='experiments/approx_error_1d')
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


def normal_log_density(x, mean=0, stdev=1):
    term = (x - mean) / stdev
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(stdev) + term * term)


def data_sample(batch_size):
    x1 = np.random.randn(batch_size) * np.sqrt(0.4) - 2.8
    x2 = np.random.randn(batch_size) * np.sqrt(0.4) - 0.9
    x3 = np.random.randn(batch_size) * np.sqrt(0.4) + 2.
    xs = np.concatenate([x1[:, None], x2[:, None], x3[:, None]], 1)
    k = np.random.randint(0, 3, batch_size)
    x = xs[np.arange(batch_size), k]
    return torch.tensor(x[:, None]).float().to(device)


def data_density(x):
    p1 = normal_log_density(x, mean=-2.8, stdev=np.sqrt(0.4))
    p2 = normal_log_density(x, mean=-0.9, stdev=np.sqrt(0.4))
    p3 = normal_log_density(x, mean=2.0, stdev=np.sqrt(0.4))
    return torch.log(p1.exp() / 3 + p2.exp() / 3 + p3.exp() / 3)


def model_density(x, model):
    x = x.to(device)
    z, delta_logp = model(x, torch.zeros_like(x))
    logpx = standard_normal_logprob(z) - delta_logp
    return logpx


def model_sample(model, batch_size):
    z = torch.randn(batch_size, 1)
    logqz = standard_normal_logprob(z)
    x, logqx = model(z, logqz, reverse=True)
    return x, logqx


def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    x = data_sample(batch_size)
    logpx = model_density(x, model)
    return -torch.mean(logpx)


def train():

    model = build_model_tabular(args, 1).to(device)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    end = time.time()
    best_loss = float('inf')
    model.train()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        loss = compute_loss(args, model)
        loss_meter.update(loss.item())

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)

        loss.backward()
        optimizer.step()

        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)

        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )
        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                test_nfe = count_nfe(model)
                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss, test_nfe)
                logger.info(log_message)

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    utils.makedirs(args.save)
                    torch.save({
                        'args': args,
                        'state_dict': model.state_dict(),
                    }, os.path.join(args.save, 'checkpt.pth'))
                model.train()

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                model.eval()

                xx = torch.linspace(-10, 10, 10000).view(-1, 1)
                true_p = data_density(xx)
                plt.plot(xx.view(-1).cpu().numpy(), true_p.view(-1).exp().cpu().numpy(), label='True')

                true_p = model_density(xx, model)
                plt.plot(xx.view(-1).cpu().numpy(), true_p.view(-1).exp().cpu().numpy(), label='Model')

                utils.makedirs(os.path.join(args.save, 'figs'))
                plt.savefig(os.path.join(args.save, 'figs', '{:06d}.jpg'.format(itr)))
                plt.close()

                model.train()

        end = time.time()

    logger.info('Training has finished.')


def evaluate():
    model = build_model_tabular(args, 1).to(device)
    set_cnf_options(args, model)

    checkpt = torch.load(os.path.join(args.save, 'checkpt.pth'))
    model.load_state_dict(checkpt['state_dict'])
    model.to(device)

    tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    errors = []
    with torch.no_grad():
        for tol in tols:
            args.rtol = tol
            args.atol = tol
            set_cnf_options(args, model)

            xx = torch.linspace(-15, 15, 500000).view(-1, 1).to(device)
            prob_xx = model_density(xx, model).double().view(-1).cpu()
            xx = xx.double().cpu().view(-1)
            dxx = torch.log(xx[1:] - xx[:-1])
            num_integral = torch.logsumexp(prob_xx[:-1] + dxx, 0).exp()
            errors.append(float(torch.abs(num_integral - 1.)))

            print(errors[-1])

    plt.figure(figsize=(5, 3))
    plt.plot(tols, errors, linewidth=3, marker='o', markersize=7)
    # plt.plot([-1, 0.2], [-1, 0.2], '--', color='grey', linewidth=1)
    plt.xscale("log", nonposx='clip')
    # plt.yscale("log", nonposy='clip')
    plt.xlabel('Solver Tolerance', fontsize=17)
    plt.ylabel('$| 1 - \int p(x) |$', fontsize=17)
    plt.tight_layout()
    plt.savefig('ode_solver_error_vs_tol.pdf')


if __name__ == '__main__':
    # train()
    evaluate()
