import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

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

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
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

parser.add_argument('--save', type=str, default='experiments/cnf')
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
        if args.spectral_norm: spectral_norm_power_iteration(model, 1)

        loss = compute_loss(args, model)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
            )
            loss = loss + reg_loss

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
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

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
                p_samples = toy_data.inf_train_gen(args.data, batch_size=2000)

                sample_fn, density_fn = get_transforms(model)

                plt.figure(figsize=(9, 3))
                visualize_transform(
                    p_samples, torch.randn, standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                    samples=True, npts=800, device=device
                )
                fig_filename = os.path.join(args.save, 'figs', '{:04d}.jpg'.format(itr))
                utils.makedirs(os.path.dirname(fig_filename))
                plt.savefig(fig_filename)
                plt.close()
                model.train()

        end = time.time()

    logger.info('Training has finished.')

    save_traj_dir = os.path.join(args.save, 'trajectory')
    logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = toy_data.inf_train_gen(args.data, batch_size=2000)
    save_trajectory(model, data_samples, save_traj_dir, device=device)
    trajectory_to_video(save_traj_dir)
