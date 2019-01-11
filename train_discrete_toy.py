import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time

import torch
import torch.optim as optim

import lib.layers as layers
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

from train_misc import standard_normal_logprob
from train_misc import count_parameters

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)

parser.add_argument('--depth', help='number of coupling layers', type=int, default=10)
parser.add_argument('--glow', type=eval, choices=[True, False], default=False)
parser.add_argument('--nf', type=eval, choices=[True, False], default=False)

parser.add_argument('--niters', type=int, default=100001)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=1000)
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

logger.info(args)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


def construct_model():

    if args.nf:
        chain = []
        for i in range(args.depth):
            chain.append(layers.PlanarFlow(2))
        return layers.SequentialFlow(chain)
    else:
        chain = []
        for i in range(args.depth):
            if args.glow: chain.append(layers.BruteForceLayer(2))
            chain.append(layers.CouplingLayer(2, swap=i % 2 == 0))
        return layers.SequentialFlow(chain)


def get_transforms(model):

    if args.nf:
        sample_fn = None
    else:

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

    model = construct_model().to(device)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.98)
    loss_meter = utils.RunningAverageMeter(0.98)

    end = time.time()
    best_loss = float('inf')
    model.train()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()

        loss = compute_loss(args, model)
        loss_meter.update(loss.item())

        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)

        if itr % args.log_freq == 0:
            log_message = (
                'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'.format(
                    itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
                )
            )
            logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                model.eval()
                test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f}'.format(itr, test_loss)
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
