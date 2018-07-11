import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import integrate

import lib.layers as layers
import lib.regularizations as regularizations
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument(
    "--data", choices=["swissroll", "8gaussians", "pinwheel", "circles", "moons", "mnist"], type=str, default="moons"
)
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--conv", type=eval, default=False, choices=[True, False])
parser.add_argument("--layer_type", type=str, default="ignore", choices=["ignore", "concat", "hyper", "blend"])
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"])
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--time_length", type=float, default=1.0)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--data_size", type=int, default=10000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr_max", type=float, default=1e-3)
parser.add_argument("--lr_min", type=float, default=1e-3)
parser.add_argument("--lr_interval", type=float, default=2000)
parser.add_argument("--weight_decay", type=float, default=1e-6)

parser.add_argument("--adjoint", type=eval, default=True, choices=[True, False])
parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--logit", type=eval, default=True, choices=[True, False])

# Regularizations
parser.add_argument("--l2_coeff", type=float, default=0, help="L2 on dynamics.")
parser.add_argument("--dl2_coeff", type=float, default=0, help="Directional L2 on dynamics.")

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()


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
    if args.data == "mnist":
        trans = tforms.Compose([tforms.ToTensor(), add_noise, lambda x: x.view(-1)])
        train_set = dset.MNIST(root="./data", train=True, transform=trans, download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans, download=True)
        if args.conv:
            data_shape = (1, 28, 28)
        else:
            data_shape = (784,)
    else:
        args.conv = False  # conv not supported for 2D datasets.
        dataset = toy_data.inf_train_gen(args.data, batch_size=args.data_size)
        dataset = [(d, 0) for d in dataset]  # add dummy labels
        num_train = int(args.data_size * .9)
        train_set, test_set = dataset[:num_train], dataset[num_train:]
        data_shape = (2,)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    print("==>>> total training batch number: {}".format(len(train_loader)))
    print("==>>> total testing batch number: {}".format(len(test_loader)))
    return train_loader, test_loader, data_shape


def compute_bits_per_dim(x, model):

    zero = torch.zeros(x.shape[0], 1).to(x)

    # preprocessing layer
    logit_x, delta_logpx_logit_tranform = model.chain[-1](x, zero, reverse=True)

    # the rest of the layers
    z, delta_logp = model(logit_x, zero, reverse=True, inds=range(len(model.chain) - 2, -1, -1))

    # compute log p(z)
    logpz = standard_normal_logprob(z).sum(1, keepdim=True)

    # compute log p(x)
    logpx_logit = logpz - delta_logp
    logpx = logpx_logit - delta_logpx_logit_tranform

    logpx_per_dim = torch.mean(logpx) / (x.nelement() / x.shape[0])  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim, torch.mean(logpx_logit)


def regularized_model(model):
    dict_of_regularizations = {}
    if args.l2_coeff != 0:
        dict_of_regularizations[regularizations.L2Regularization] = args.l2_coeff
    if args.dl2_coeff != 0:
        dict_of_regularizations[regularizations.DirectionalL2Regularization] = args.dl2_coeff

    for layer in model.chain:
        if isinstance(layer, layers.CNF):
            layer.odefunc


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

    _odeint = integrate.odeint_adjoint if args.adjoint else integrate.odeint
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    # build model
    chain = [
        layers.CNF(
            hidden_dims=hidden_dims,
            T=args.time_length,
            odeint=_odeint,
            input_shape=data_shape,
            strides=strides,
            conv=args.conv,
            layer_type=args.layer_type,
            divergence_fn=args.divergence_fn,
            nonlinearity=args.nonlinearity,
        )
    ]
    if args.logit:
        chain.append(layers.LogitTransform(alpha=args.alpha))
    model = layers.SequentialFlow(chain)

    print(model)
    print("Number of trainable parameters: {}".format(count_parameters(model)))

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
    fixed_z = cvt(torch.randn(100, 784))

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
                print(
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
                print("validating...")
                losses = []
                logit_losses = []
                for (x, y) in test_loader:
                    x = cvt(x)
                    loss, logit_loss = compute_bits_per_dim(x, model)
                    losses.append(loss)
                    logit_losses.append(logit_loss.item())
                loss = np.mean(losses)
                logit_loss = np.mean(logit_losses)
                print(
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
            if args.data == "mnist":
                generated_samples = model(fixed_z).view(-1, 1, 28, 28)
                save_image(generated_samples, fig_filename, nrow=10)
            else:
                generated_samples = toy_data.inf_train_gen(args.data, batch_size=10000)
                plt.figure(figsize=(9, 3))
                visualize_transform(
                    generated_samples, torch.randn, standard_normal_logprob, model, samples=True, device=device
                )
                plt.savefig(fig_filename)
