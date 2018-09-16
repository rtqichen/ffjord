import argparse
import os
import time

import torch

import lib.utils as utils
from lib.custom_optimizers import Adam
import lib.layers as layers

import datasets

from train_misc import standard_normal_logprob, count_parameters

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', choices=['power', 'gas', 'hepmass', 'miniboone', 'bsds300'], type=str, default='miniboone'
)

parser.add_argument('--depth', type=int, default=10)
parser.add_argument('--dims', type=str, default="100-100")
parser.add_argument('--nonlinearity', type=str, default="tanh")
parser.add_argument('--glow', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--test_batch_size', type=int, default=None)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--log_freq', type=int, default=10)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

test_batch_size = args.test_batch_size if args.test_batch_size else args.batch_size


def batch_iter(X, batch_size=args.batch_size, shuffle=False):
    """
    X: feature tensor (shape: num_instances x num_features)
    """
    if shuffle:
        idxs = torch.randperm(X.shape[0])
    else:
        idxs = torch.arange(X.shape[0])
    if X.is_cuda:
        idxs = idxs.cuda()
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


ndecs = 0


def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 100
        ndecs = 2
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 10**ndecs


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


def build_model(input_dim):
    hidden_dims = tuple(map(int, args.dims.split("-")))
    chain = []
    for i in range(args.depth):
        if args.glow: chain.append(layers.BruteForceLayer(input_dim))
        chain.append(layers.MaskedCouplingLayer(input_dim, hidden_dims, 'alternate', swap=i % 2 == 0))
        if args.batch_norm: chain.append(layers.MovingBatchNorm1d(input_dim, bn_lag=args.bn_lag))
    return layers.SequentialFlow(chain)


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

    logger.info('Using {} GPUs.'.format(torch.cuda.device_count()))

    data = load_data(args.data)
    data.trn.x = torch.from_numpy(data.trn.x)
    data.val.x = torch.from_numpy(data.val.x)
    data.tst.x = torch.from_numpy(data.tst.x)

    model = build_model(data.n_dims).to(device)

    if args.resume is not None:
        checkpt = torch.load(args.resume)
        model.load_state_dict(checkpt['state_dict'])

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    if not args.evaluate:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        time_meter = utils.RunningAverageMeter(0.98)
        loss_meter = utils.RunningAverageMeter(0.98)

        best_loss = float('inf')
        itr = 0
        n_vals_without_improvement = 0
        end = time.time()
        model.train()
        while True:
            if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                break

            for x in batch_iter(data.trn.x, shuffle=True):
                if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                    break

                optimizer.zero_grad()

                x = cvt(x)
                loss = compute_loss(x, model)
                loss_meter.update(loss.item())

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - end)

                if itr % args.log_freq == 0:
                    log_message = (
                        'Iter {:06d} | Epoch {:.2f} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | '.format(
                            itr,
                            float(itr) / (data.trn.x.shape[0] / float(args.batch_size)), time_meter.val, time_meter.avg,
                            loss_meter.val, loss_meter.avg
                        )
                    )
                    logger.info(log_message)
                itr += 1
                end = time.time()

                # Validation loop.
                if itr % args.val_freq == 0:
                    model.eval()
                    start_time = time.time()
                    with torch.no_grad():
                        val_loss = utils.AverageMeter()
                        for x in batch_iter(data.val.x, batch_size=test_batch_size):
                            x = cvt(x)
                            val_loss.update(compute_loss(x, model).item(), x.shape[0])

                        if val_loss.avg < best_loss:
                            best_loss = val_loss.avg
                            utils.makedirs(args.save)
                            torch.save({
                                'args': args,
                                'state_dict': model.state_dict(),
                            }, os.path.join(args.save, 'checkpt.pth'))
                            n_vals_without_improvement = 0
                        else:
                            n_vals_without_improvement += 1
                        update_lr(optimizer, n_vals_without_improvement)

                        log_message = (
                            '[VAL] Iter {:06d} | Val Loss {:.6f} | '
                            'NoImproveEpochs {:02d}/{:02d}'.format(
                                itr, val_loss.avg, n_vals_without_improvement, args.early_stopping
                            )
                        )
                        logger.info(log_message)
                    model.train()

        logger.info('Training has finished.')
        model = restore_model(model, os.path.join(args.save, 'checkpt.pth')).to(device)

    logger.info('Evaluating model on test set.')
    model.eval()

    with torch.no_grad():
        test_loss = utils.AverageMeter()
        for itr, x in enumerate(batch_iter(data.tst.x, batch_size=test_batch_size)):
            x = cvt(x)
            test_loss.update(compute_loss(x, model).item(), x.shape[0])
            logger.info('Progress: {:.2f}%'.format(itr / (data.tst.x.shape[0] / test_batch_size)))
        log_message = '[TEST] Iter {:06d} | Test Loss {:.6f} '.format(itr, test_loss.avg)
        logger.info(log_message)
