# !/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import time
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
import math
import random

import os

import datetime

import lib.utils as utils
import lib.layers.odefunc as odefunc

import vae_lib.models.VAE as VAE
import vae_lib.models.CNFVAE as CNFVAE
from vae_lib.optimization.training import train, evaluate
from vae_lib.utils.load_data import load_dataset
from vae_lib.utils.plotting import plot_training_curve

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser(description='PyTorch Sylvester Normalizing flows')

parser.add_argument(
    '-d', '--dataset', type=str, default='mnist', choices=['mnist', 'freyfaces', 'omniglot', 'caltech'],
    metavar='DATASET', help='Dataset choice.'
)

parser.add_argument(
    '-freys', '--freyseed', type=int, default=123, metavar='FREYSEED',
    help="""Seed for shuffling frey face dataset for test split. Ignored for other datasets.
                    Results in paper are produced with seeds 123, 321, 231"""
)

parser.add_argument('-nc', '--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--manual_seed', type=int, help='manual seed, if not given resorts to random seed.')

parser.add_argument(
    '-li', '--log_interval', type=int, default=10, metavar='LOG_INTERVAL',
    help='how many batches to wait before logging training status'
)

parser.add_argument(
    '-od', '--out_dir', type=str, default='snapshots', metavar='OUT_DIR',
    help='output directory for model snapshots etc.'
)

fp = parser.add_mutually_exclusive_group(required=False)
fp.add_argument('-te', '--testing', action='store_true', dest='testing', help='evaluate on test set after training')
fp.add_argument('-va', '--validation', action='store_false', dest='testing', help='only evaluate on validation set')
parser.set_defaults(testing=True)

# optimization settings
parser.add_argument(
    '-e', '--epochs', type=int, default=2000, metavar='EPOCHS', help='number of epochs to train (default: 2000)'
)
parser.add_argument(
    '-es', '--early_stopping_epochs', type=int, default=100, metavar='EARLY_STOPPING',
    help='number of early stopping epochs'
)

parser.add_argument(
    '-bs', '--batch_size', type=int, default=512, metavar='BATCH_SIZE', help='input batch size for training'
)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, metavar='LEARNING_RATE', help='learning rate')

parser.add_argument(
    '-w', '--warmup', type=int, default=100, metavar='N',
    help='number of epochs for warm-up. Set to 0 to turn warmup off.'
)
parser.add_argument('--max_beta', type=float, default=1., metavar='MB', help='max beta for warm-up')
parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB', help='min beta for warm-up')
parser.add_argument(
    '-f', '--flow', type=str, default='no_flow',
    choices=['planar', 'iaf', 'householder', 'orthogonal', 'triangular', 'cnf',
             'no_flow'], help="""Type of flows to use, no flows can also be selected"""
)
parser.add_argument(
    '-nf', '--num_flows', type=int, default=4, metavar='NUM_FLOWS',
    help='Number of flow layers, ignored in absence of flows'
)
parser.add_argument(
    '-nv', '--num_ortho_vecs', type=int, default=8, metavar='NUM_ORTHO_VECS',
    help=""" For orthogonal flow: How orthogonal vectors per flow do you need.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-nh', '--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',
    help=""" For Householder Sylvester flow: Number of Householder matrices per flow.
                    Ignored for other flow types."""
)
parser.add_argument(
    '-mhs', '--made_h_size', type=int, default=320, metavar='MADEHSIZE',
    help='Width of mades for iaf. Ignored for all other flows.'
)
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE', help='how many stochastic hidden units')
# gpu/cpu
parser.add_argument('--gpu_num', type=int, default=0, metavar='GPU', help='choose GPU to run on.')

# CNF settings
parser.add_argument(
    "--layer_type", type=str, default="concat",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='512-512')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=False)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.manual_seed is None:
    args.manual_seed = random.randint(1, 100000)
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
np.random.seed(args.manual_seed)

if args.cuda:
    # gpu device number
    torch.cuda.set_device(args.gpu_num)

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}


def run(args, kwargs):
    # ==================================================================================================================
    # SNAPSHOTS
    # ==================================================================================================================
    args.model_signature = str(datetime.datetime.now())[0:19].replace(' ', '_')
    args.model_signature = args.model_signature.replace(':', '_')

    snapshots_path = os.path.join(args.out_dir, 'vae_' + args.dataset + '_')
    snap_dir = snapshots_path + args.flow + '_' + str(args.gpu_num)

    if args.flow != 'no_flow':
        snap_dir += '_' + 'num_flows_' + str(args.num_flows)

    if args.flow == 'orthogonal':
        snap_dir = snap_dir + '_num_vectors_' + str(args.num_ortho_vecs)
    elif args.flow == 'orthogonalH':
        snap_dir = snap_dir + '_num_householder_' + str(args.num_householder)
    elif args.flow == 'iaf':
        snap_dir = snap_dir + '_madehsize_' + str(args.made_h_size)

    elif args.flow == 'permutation':
        snap_dir = snap_dir + '_' + 'kernelsize_' + str(args.kernel_size)
    elif args.flow == 'mixed':
        snap_dir = snap_dir + '_' + 'num_householder_' + str(args.num_householder)

    snap_dir = snap_dir + '__' + args.model_signature + '/'

    args.snap_dir = snap_dir

    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

    # logger
    utils.makedirs(args.snap_dir)
    logger = utils.get_logger(logpath=os.path.join(args.snap_dir, 'logs'), filepath=os.path.abspath(__file__))

    logger.info(args)

    # SAVING
    torch.save(args, snap_dir + args.flow + '.config')

    # ==================================================================================================================
    # LOAD DATA
    # ==================================================================================================================
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # ==================================================================================================================
    # SELECT MODEL
    # ==================================================================================================================
    # flow parameters and architecture choice are passed on to model through args

    if args.flow == 'no_flow':
        model = VAE.VAE(args)
    elif args.flow == 'planar':
        model = VAE.PlanarVAE(args)
    elif args.flow == 'iaf':
        model = VAE.IAFVAE(args)
    elif args.flow == 'orthogonal':
        model = VAE.OrthogonalSylvesterVAE(args)
    elif args.flow == 'householder':
        model = VAE.HouseholderSylvesterVAE(args)
    elif args.flow == 'triangular':
        model = VAE.TriangularSylvesterVAE(args)
    elif args.flow == 'cnf':
        model = CNFVAE.CNFVAE(args)
    else:
        raise ValueError('Invalid flow choice')

    if args.cuda:
        logger.info("Model on GPU")
        model.cuda()

    logger.info(model)

    optimizer = optim.Adamax(model.parameters(), lr=args.learning_rate, eps=1.e-7)

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    train_loss = []
    val_loss = []

    # for early stopping
    best_loss = np.inf
    best_bpd = np.inf
    e = 0
    epoch = 0

    train_times = []

    for epoch in range(1, args.epochs + 1):

        t_start = time.time()
        tr_loss = train(epoch, train_loader, model, optimizer, args, logger)
        train_loss.append(tr_loss)
        train_times.append(time.time() - t_start)
        logger.info('One training epoch took %.2f seconds' % (time.time() - t_start))

        v_loss, v_bpd = evaluate(val_loader, model, args, logger, epoch=epoch)

        val_loss.append(v_loss)

        # early-stopping
        if v_loss < best_loss:
            e = 0
            best_loss = v_loss
            if args.input_type != 'binary':
                best_bpd = v_bpd
            logger.info('->model saved<-')
            torch.save(model, snap_dir + args.flow + '.model')
            # torch.save(model, snap_dir + args.flow + '_' + args.architecture + '.model')

        elif (args.early_stopping_epochs > 0) and (epoch >= args.warmup):
            e += 1
            if e > args.early_stopping_epochs:
                break

        if args.input_type == 'binary':
            logger.info(
                '--> Early stopping: {}/{} (BEST: loss {:.4f})\n'.format(e, args.early_stopping_epochs, best_loss)
            )

        else:
            logger.info(
                '--> Early stopping: {}/{} (BEST: loss {:.4f}, bpd {:.4f})\n'.
                format(e, args.early_stopping_epochs, best_loss, best_bpd)
            )

        if math.isnan(v_loss):
            raise ValueError('NaN encountered!')

    train_loss = np.hstack(train_loss)
    val_loss = np.array(val_loss)

    plot_training_curve(train_loss, val_loss, fname=snap_dir + '/training_curve_%s.pdf' % args.flow)

    # training time per epoch
    train_times = np.array(train_times)
    mean_train_time = np.mean(train_times)
    std_train_time = np.std(train_times, ddof=1)
    logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time))

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================

    test_score_file = snap_dir + 'test_scores.txt'

    with open('experiment_log.txt', 'a') as ff:
        logger.info(args, file=ff)
        logger.info('Stopped after %d epochs' % epoch, file=ff)
        logger.info('Average train time per epoch: %.2f +/- %.2f' % (mean_train_time, std_train_time), file=ff)

    final_model = torch.load(snap_dir + args.flow + '.model')

    if args.testing:
        validation_loss, validation_bpd = evaluate(val_loader, final_model, args)
        test_loss, test_bpd = evaluate(test_loader, final_model, args, testing=True)

        with open('experiment_log.txt', 'a') as ff:
            logger.info('FINAL EVALUATION ON VALIDATION SET\n' 'ELBO (VAL): {:.4f}\n'.format(validation_loss), file=ff)
            logger.info('FINAL EVALUATION ON TEST SET\n' 'NLL (TEST): {:.4f}\n'.format(test_loss), file=ff)
            if args.input_type != 'binary':
                logger.info(
                    'FINAL EVALUATION ON VALIDATION SET\n'
                    'ELBO (VAL) BPD : {:.4f}\n'.format(validation_bpd), file=ff
                )
                logger.info('FINAL EVALUATION ON TEST SET\n' 'NLL (TEST) BPD: {:.4f}\n'.format(test_bpd), file=ff)

    else:
        validation_loss, validation_bpd = evaluate(val_loader, final_model, args)
        # save the test score in case you want to look it up later.
        _, _ = evaluate(test_loader, final_model, args, testing=True, file=test_score_file)

        with open('experiment_log.txt', 'a') as ff:
            logger.info(
                'FINAL EVALUATION ON VALIDATION SET\n'
                'ELBO (VALIDATION): {:.4f}\n'.format(validation_loss), file=ff
            )
            if args.input_type != 'binary':
                logger.info(
                    'FINAL EVALUATION ON VALIDATION SET\n'
                    'ELBO (VAL) BPD : {:.4f}\n'.format(validation_bpd), file=ff
                )


if __name__ == "__main__":

    run(args, kwargs)
