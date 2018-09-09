import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import lib.utils as utils
import lib.layers.odefunc as odefunc
from lib.custom_optimizers import Adam

import datasets

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument('--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='pinwheel')
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
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
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--max_epochs', type=int, default=2500)
parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=5000)
parser.add_argument('--weight_decay', type=float, default=1e-6)

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
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')


def compute_loss(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss


def restore_model(model, filename):
    checkpt = torch.load(filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpt["state_dict"])
    return model


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    data = load_data(args.data)
    train_loader = DataLoader(data.trn.x, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data.val.x, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(data.tst.x, batch_size=args.test_batch_size, shuffle=False)

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.n_dims, regularization_fns).to(device)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    itr = 0
    n_iters_without_improvement = 0
    for epoch in range(args.max_epochs):
        if args.early_stopping > 0 and n_iters_without_improvement > args.early_stopping:
            break

        time_meter = utils.RunningAverageMeter(0.93)
        loss_meter = utils.RunningAverageMeter(0.93)
        nfef_meter = utils.RunningAverageMeter(0.93)
        nfeb_meter = utils.RunningAverageMeter(0.93)
        tt_meter = utils.RunningAverageMeter(0.93)

        # Training loop.
        model.train()
        end = time.time()
        for _, x in enumerate(train_loader):
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            x = cvt(x)
            loss = compute_loss(x, model)
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

            if itr % args.log_freq == 0:
                log_message = (
                    'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
                    ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val,
                        nfef_meter.avg, nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
                    )
                )
                if len(regularization_coeffs) > 0:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

                logger.info(log_message)
            itr += 1
            end = time.time()

        # Validation loop.
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            val_loss = utils.AverageMeter()
            val_nfe = utils.AverageMeter()
            for _, x in enumerate(val_loader):
                x = cvt(x)
                val_loss.update(compute_loss(x, model).item(), x.shape[0])
                val_nfe.update(count_nfe(model))

            if val_loss.avg < best_loss:
                best_loss = val_loss.avg
                utils.makedirs(args.save)
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, os.path.join(args.save, 'checkpt.pth'))
                n_iters_without_improvement = 0
            else:
                n_iters_without_improvement += 1

            log_message = '[VAL] Epoch {:04d} | Val Loss {:.6f} | NFE {:.0f} | NoImproveEpochs {:02d}/{:02d}'.format(
                epoch, val_loss.avg, val_nfe.avg, n_iters_without_improvement, args.early_stopping
            )
            logger.info(log_message)

    logger.info('Training has finished.')
    model = restore_model(model, os.path.join(args.save, 'checkpt.pth')).to(device)
    set_cnf_options(args, model)

    with torch.no_grad():
        test_loss = utils.AverageMeter()
        test_nfe = utils.AverageMeter()
        for _, x in enumerate(test_loader):
            x = cvt(x)
            test_loss.update(compute_loss(x, model).item(), x.shape[0])
            test_nfe.update(count_nfe(model))
        log_message = '[TEST] Epoch {:04d} | Test Loss {:.6f} | NFE {:.0f}'.format(epoch, test_loss.avg, test_nfe.avg)
        logger.info(log_message)
