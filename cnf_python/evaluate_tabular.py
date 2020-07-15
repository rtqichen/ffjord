import argparse
import os
import time
import torch

import lib.utils as utils
import lib.layers.odefunc as odefunc
import datasets
import numpy as np

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn
from train_tabular import *

# download data from https://zenodo.org/record/1161203#.XbiVGUVKhgi

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams', 'do']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='miniboone'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--hdim_factor', type=int, default=20) # default=10
parser.add_argument('--nhidden', type=int, default=2)      # default=1
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.') # defualt=3
parser.add_argument('--time_length', type=float, default=1.0) # default=1.0
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-8)
parser.add_argument('--rtol', type=float, default=1e-6)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)
parser.add_argument("--test_step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-6)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0
    args.train_T = False

logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size

if __name__ == '__main__':

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    data = load_data(args.data)
    data.tst.x = torch.from_numpy(data.tst.x)
    logger.info('test shape')
    logger.info(data.tst.x.shape)

    args.dims = '-'.join([str(args.hdim_factor * data.n_dims)] * args.nhidden)

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.n_dims, regularization_fns).to(device)
    set_cnf_options(args, model)

    for k in model.state_dict().keys():
        logger.info(k)

    if args.resume is not None:

        logger.info('Training has finished.')
        model = restore_model(model, args.resume).to(device)
        set_cnf_options(args, model)
    else:
        logger.info('must use --resume flag to provide the state_dict to evaluate')
        exit(1)

    logger.info(model)
    nWeights = count_parameters(model)
    logger.info("Number of trainable parameters: {}".format(nWeights))
    logger.info('Evaluating model on test set.')
    model.eval()

    override_divergence_fn(model, "brute_force")

    bInverse = True  # check one batch for inverse error, for speed

    with torch.no_grad():
        test_loss = utils.AverageMeter()
        test_nfe = utils.AverageMeter()
        for itr, x in enumerate(batch_iter(data.tst.x, batch_size=test_batch_size)):

            x = cvt(x)
            test_loss.update(compute_loss(x, model).item(), x.shape[0])
            test_nfe.update(count_nfe(model))

            if bInverse:  # check the ivnerse error
                z = model(x, reverse=False)  # push forward
                xpred = model(z, reverse=True)  # inverse
                logger.info('inverse norm for first batch: ')
                logger.info(torch.norm(xpred - x).item() / x.shape[0])
                bInverse = False

            logger.info('Progress: {:.2f}%'.format(100. * itr / (data.tst.x.shape[0] / test_batch_size)))
        log_message = '[TEST] Iter {:06d} | Test Loss {:.6f} | NFE {:.0f}'.format(itr, test_loss.avg, test_nfe.avg)
        logger.info(log_message)


