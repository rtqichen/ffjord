import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.spectral_norm as spectral_norm
import lib.utils as utils

# go fast boi!!
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10"], type=str, default="mnist")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--conv", type=eval, default=True, choices=[True, False])

parser.add_argument(
    "--layer_type", type=str, default="ignore", choices=["ignore", "concat", "concatcoord", "hyper", "blend"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"])

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--time_length", type=float, default=None)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--data_size", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-3)
parser.add_argument("--lr_interval", type=float, default=2000)
parser.add_argument("--weight_decay", type=float, default=1e-6)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--logit", type=eval, default=True, choices=[True, False])

# Regularizations
parser.add_argument("--l2_coeff", type=float, default=0, help="L2 on dynamics.")
parser.add_argument("--dl2_coeff", type=float, default=0, help="Directional L2 on dynamics.")
parser.add_argument('-spectral_norm', action='store_true')

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def _add_spectral_norm(model):
    def recursive_apply_sn(parent_module):
        for child_name in list(parent_module._modules.keys()):
            child_module = parent_module._modules[child_name]
            classname = child_module.__class__.__name__
            if classname.find('Conv') != -1 and 'weight' in child_module._parameters:
                del parent_module._modules[child_name]
                parent_module.add_module(child_name, spectral_norm.spectral_norm(child_module, 'weight'))
            else:
                recursive_apply_sn(child_module)

    recursive_apply_sn(model)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def update_lr(optimizer, itr):
    lr = args.lr_min + 0.5 * (args.lr_max - args.lr_min) * (1 + np.cos(itr / args.num_epochs * np.pi))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.dataset == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(args.imagesize),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("==>>> total training batch number: {}".format(len(train_loader)))
    logger.info("==>>> total testing batch number: {}".format(len(test_loader)))
    return train_loader, test_loader, data_shape


def compute_bits_per_dim(x, model):

    zero = torch.zeros(x.shape[0], 1).to(x)

    assert args.logit, "logit=False not supported"

    # preprocessing layer
    logit_x, delta_logpx_logit_tranform = model.chain[-1](x, zero, reverse=True)

    # the rest of the layers
    z, delta_logp = model(logit_x, zero, reverse=True, inds=range(len(model.chain) - 2, -1, -1))

    # compute log p(z)
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)

    # compute log p(x)
    logpx_logit = logpz - delta_logp
    logpx = logpx_logit - delta_logpx_logit_tranform

    logpx_per_dim = torch.mean(logpx) / (x.nelement() / x.shape[0])  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim, torch.mean(logpx_logit)


def count_nfe(model):
    num_evals = 0
    for layer in model.chain:
        if isinstance(layer, layers.CNF):
            num_evals += layer.num_evals()
    return num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device)

    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)

    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    # build model
    # neural net that parameterizes the gradient of the flow
    gfunc = layers.ODEnet(
        hidden_dims=hidden_dims, input_shape=data_shape, strides=strides, conv=args.conv, layer_type=args.layer_type,
        nonlinearity=args.nonlinearity
    )
    # flow chain
    chain = [
        layers.CNF(
            odefunc=layers.ODEfunc(input_shape=data_shape, diffeq=gfunc, divergence_fn=args.divergence_fn),
            T=args.time_length,
        )
    ]
    if args.logit:
        chain.append(layers.SigmoidTransform(alpha=args.alpha))
    model = layers.SequentialFlow(chain)

    if args.spectral_norm:
        _add_spectral_norm(model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    model.to(device)

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    logp_logit_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)

    best_loss = float("inf")
    itr = (args.begin_epoch - 1) * len(train_loader)
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)

            # compute loss
            bits_per_dim, logit_loss = compute_bits_per_dim(x, model)
            bits_per_dim.backward()

            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(bits_per_dim.item())
            logp_logit_meter.update(logit_loss.item())
            steps_meter.update(count_nfe(model))

            if itr % args.log_freq == 0:
                logger.info(
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Logit LogP {:.4f}({:.4f}) | Steps {:.0f}({:.2f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, logp_logit_meter.val,
                        logp_logit_meter.avg, steps_meter.val, steps_meter.avg
                    )
                )

            itr += 1

        # compute test loss
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                logit_losses = []
                for (x, y) in test_loader:
                    x = cvt(x)
                    loss, logit_loss = compute_bits_per_dim(x, model)
                    losses.append(loss)
                    logit_losses.append(logit_loss.item())
                loss = np.mean(losses)
                logit_loss = np.mean(logit_losses)
                logger.info(
                    "Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}, Logit LogP {:.4f}".
                    format(epoch, time.time() - start, loss, logit_loss)
                )
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)
