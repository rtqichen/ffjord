import argparse
import time
import os
import os.path
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from lib.odenvp import ODENVP
import lib.datasets as datasets
import lib.priors as priors
import lib.utils as utils
import lib.layers as layers

# Arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'celeba'])
parser.add_argument('-dataroot', type=str, default='data')
parser.add_argument('-imagesize', type=int, default=32)
parser.add_argument('-noise', type=eval, default=True, choices=[True, False])

parser.add_argument('-nepochs', help='Number of epochs for training', type=int, default=200)
parser.add_argument('-batchsize', help='Minibatch size', type=int, default=100)
parser.add_argument('-lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('-wd', help='Weight decay', type=float, default=1e-6)
parser.add_argument('-save', help='directory to save results', type=str, default='experiment1')
parser.add_argument('-cpu', action='store_true')
parser.add_argument('-val_batchsize', help='minibatch size', type=int, default=100)
parser.add_argument('-seed', type=int, default=None)

parser.add_argument('-resume', type=str, default=None)
parser.add_argument('-begin_epoch', type=int, default=0)

parser.add_argument('-n_resblocks', type=int, default=None)
parser.add_argument('-multiplier', type=int, default=None)

parser.add_argument('-nworkers', type=int, default=4)
parser.add_argument('-print_freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('-vis_freq', help='Visualize progress every so iterations', type=int, default=500)
args = parser.parse_args()

# flag to use cuda
use_cuda = not args.cpu

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


# Dataset and hyperparameters
if args.dataset == 'cifar10':
    im_dim = 3
    n_resblocks = 8 if args.n_resblocks is None else args.n_resblocks
    multiplier = 1 if args.multiplier is None else args.multiplier
    intermediate_dim = 64
    n_scale = 2
    alpha = 0.05
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.dataroot, train=True, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), add_noise
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers, pin_memory=use_cuda
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.dataroot, train=False, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
            ])
        ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers, pin_memory=use_cuda
    )
elif args.dataset == 'mnist':
    im_dim = 1
    n_resblocks = 4 if args.n_resblocks is None else args.n_resblocks
    multiplier = 1 if args.multiplier is None else args.multiplier
    intermediate_dim = 64
    n_scale = 3
    alpha = 1e-5
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=True,
            transform=transforms.Compose([transforms.Resize(args.imagesize),
                                          transforms.ToTensor(), add_noise])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers, pin_memory=use_cuda
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.dataroot, train=False, transform=transforms.Compose([
                transforms.Resize(args.imagesize),
                transforms.ToTensor(),
            ])
        ), batch_size=args.val_batchsize, shuffle=False, num_workers=args.nworkers, pin_memory=use_cuda
    )
elif args.dataset == 'celeba':
    im_dim = 3
    n_resblocks = 2 if args.n_resblocks is None else args.n_resblocks
    multiplier = 1 if args.multiplier is None else args.multiplier
    intermediate_dim = 64
    n_scale = 5
    alpha = 1e-5
    train_loader = torch.utils.data.DataLoader(
        datasets.CelebA(
            train=True, transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.imagesize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), add_noise
            ])
        ), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers, pin_memory=use_cuda
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CelebA(train=False, transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])), batch_size=args.batchsize, shuffle=True, num_workers=args.nworkers, pin_memory=use_cuda
    )

input_size = (args.batchsize, im_dim, args.imagesize, args.imagesize)
dataset_size = len(train_loader.dataset)

# Model
model = ODENVP(
    input_size,
    n_scale=n_scale,
    n_resblocks=n_resblocks,
    multiplier=multiplier,
    intermediate_dim=intermediate_dim,
    alpha=alpha,
)
prior = priors.Normal()
logger.info(model)

# Optimization
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)

if args.resume is not None:
    checkpt = torch.load(args.resume)
    model.load_state_dict(checkpt['state_dict'])
    optimizer.load_state_dict(checkpt['optimizer_state_dict'])
    # Manually move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

output_sizes = model.calc_output_size(input_size)
fixed_z = [prior.sample(size).detach() for size in output_sizes]

# CUDA
if use_cuda:
    model.cuda()
    fixed_z = [z.cuda() for z in fixed_z]


def get_num_evals(model):
    num_evals = 0
    for m in model.modules():
        if isinstance(m, layers.CNF):
            num_evals += m.num_evals()
    return num_evals


def compute_bits_per_dim(x):
    # batch_size = x.size(0)

    # z = f(x)
    zs, neg_logdetgrads = model.forward(x, 0)

    # log p(z)
    logpz = sum([prior.log_density(z).view(z.size(0), -1).sum(1) for z in zs])

    # log p(x)
    logpx = logpz - neg_logdetgrads

    logpx_per_dim = torch.mean(logpx) / (args.imagesize * args.imagesize * im_dim)
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def train(epoch):
    batch_time = utils.RunningAverageMeter(0.97)
    bpd_meter = utils.RunningAverageMeter(0.97)
    nfe_meter = utils.RunningAverageMeter(0.97)

    model.train()

    end = time.time()
    for i, x in enumerate(train_loader):
        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        if use_cuda:
            x = x.cuda(async=True)

        loss = compute_bits_per_dim(x)

        # update running averages
        bpd_meter.update(loss.item())
        nfe_meter.update(get_num_evals(model))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Bits/dim {bpd_meter.val:.4f} ({bpd_meter.avg:.4f})\t'
                'NFE {nfe_meter.val:.0f} ({nfe_meter.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, bpd_meter=bpd_meter, nfe_meter=nfe_meter
                )
            )
        if i % args.vis_freq == 0:
            visualize(epoch, i, x)


def validate(epoch):
    """
    Evaluates the cross entropy between p_data and p_model. Also evaluates the cross entropy if data is jittered.
    """
    bpd_meter_clean = utils.AverageMeter()
    bpd_meter_noisy = utils.AverageMeter()

    model.eval()

    start = time.time()
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            if use_cuda:
                x = x.cuda(async=True)

            bpd_clean = compute_bits_per_dim(x)
            bpd_noisy = compute_bits_per_dim(add_noise(x))

            bpd_meter_clean.update(bpd_clean.item(), x.size(0))
            bpd_meter_noisy.update(bpd_noisy.item(), x.size(0))
    val_time = time.time() - start
    logger.info(
        'Epoch: [{0}]\tTime {1:.2f}\t'
        'Test[Clean] bits/dim {bpd_meter_clean.avg:.4f}\t'
        'Test[Noisy] bits/dim {bpd_meter_noisy.avg:.4f}'.format(
            epoch, val_time, bpd_meter_clean=bpd_meter_clean, bpd_meter_noisy=bpd_meter_noisy
        )
    )


def visualize(epoch, itr, real_imgs):
    model.eval()
    utils.makedirs(os.path.join(args.save, 'imgs'))

    with torch.no_grad():
        # random samples
        fake_imgs = model(fixed_z, reverse=True)

        # imgs = torch.stack((real_imgs, fake_imgs), 1).view(-1, im_dim, args.imagesize, args.imagesize)
        imgs = torch.cat([real_imgs, fake_imgs], 0)

        filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)
        model.train()


def main():
    for epoch in range(args.begin_epoch, args.nepochs):
        train(epoch)
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args,
        }, os.path.join(args.save, 'models'), epoch)
        validate(epoch)


if __name__ == '__main__':
    main()
