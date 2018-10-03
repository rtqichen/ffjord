from inspect import getsourcefile
import sys
import os
import math

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import argparse

import lib.layers as layers
import lib.odenvp as odenvp
import torch
import torchvision.transforms as tforms
import torchvision.datasets as dset
from torchvision.utils import save_image
import lib.utils as utils

from train_misc import add_spectral_norm, set_cnf_options, count_parameters

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--checkpt", type=str, required=True)
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="cifar10")
parser.add_argument("--dims", type=str, default="64,64,64")
parser.add_argument("--num_blocks", type=int, default=2, help='Number of stacked CNFs.')
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
)
parser.add_argument("--conv", type=eval, default=True, choices=[True, False])

parser.add_argument('--solver', type=str, default='dopri5')
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None)
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=-1.0)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

parser.add_argument('--ntimes', type=int, default=50)
parser.add_argument('--save', type=str, default='img_trajectory')

args = parser.parse_args()

BATCH_SIZE = 8 * 8


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            'data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    return train_set, data_shape


def create_model(args, data_shape):
    hidden_dims = tuple(map(int, args.dims.split(",")))

    model = odenvp.ODENVP(
        (BATCH_SIZE, *data_shape),
        n_blocks=args.num_blocks,
        intermediate_dims=hidden_dims,
        nonlinearity=args.nonlinearity,
        alpha=args.alpha,
        cnf_kwargs={"T": args.time_length, "train_T": args.train_T},
    )
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, data_shape = get_dataset(args)

    # build model
    model = create_model(args, data_shape)

    print(model)
    print("Number of trainable parameters: {}".format(count_parameters(model)))

    # restore parameters
    checkpt = torch.load(args.checkpt, map_location=lambda storage, loc: storage)
    pruned_sd = {}
    for k, v in checkpt['state_dict'].items():
        pruned_sd[k.replace('odefunc.odefunc', 'odefunc')] = v
    model.load_state_dict(pruned_sd)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)

    data_samples, _ = train_loader.__iter__().__next__()

    # cosine interpolate between 4 real images.
    z = data_samples[:4]
    print('Inferring base values for 4 example images.')
    z = model(z)

    phi0 = torch.linspace(0, 0.5, int(math.sqrt(BATCH_SIZE))) * math.pi
    phi1 = torch.linspace(0, 0.5, int(math.sqrt(BATCH_SIZE))) * math.pi
    phi0, phi1 = torch.meshgrid([phi0, phi1])
    phi0, phi1 = phi0.contiguous().view(-1, 1), phi1.contiguous().view(-1, 1)
    z = torch.cos(phi0) * (torch.cos(phi1) * z[0] + torch.sin(phi1) * z[1]) + \
        torch.sin(phi0) * (torch.cos(phi1) * z[2] + torch.sin(phi1) * z[3])
    print('Reconstructing images from latent interpolation.')
    z = model(z, reverse=True)

    non_cnf_layers = []

    utils.makedirs(args.save)
    img_idx = 0

    def save_imgs_figure(xs):
        global img_idx
        save_image(
            list(xs),
            os.path.join(args.save, "img_{:05d}.jpg".format(img_idx)), nrow=int(math.sqrt(BATCH_SIZE)), normalize=True,
            range=(0, 1)
        )
        img_idx += 1

    class FactorOut(torch.nn.Module):

        def __init__(self, factor_out):
            super(FactorOut, self).__init__()
            self.factor_out = factor_out

        def forward(self, x, reverse=True):
            assert reverse
            T = x.shape[0] // self.factor_out.shape[0]
            return torch.cat([x, self.factor_out.repeat(T, *([1] * (self.factor_out.ndimension() - 1)))], 1)

    time_ratio = 1.0
    print('Visualizing transformations.')
    with torch.no_grad():
        for idx, stacked_layers in enumerate(model.transforms):
            for layer in stacked_layers.chain:
                print(z.shape)
                print(non_cnf_layers)
                if isinstance(layer, layers.CNF):
                    # linspace over time, and visualize by reversing through previous non_cnf_layers.
                    cnf = layer
                    end_time = (cnf.sqrt_end_time * cnf.sqrt_end_time)
                    ntimes = int(args.ntimes * time_ratio)
                    integration_times = torch.linspace(0, end_time.item(), ntimes)
                    z_traj = cnf(z, integration_times=integration_times)

                    # reverse z(t) for all times to the input space
                    z_flatten = z_traj.view(ntimes * BATCH_SIZE, *z_traj.shape[2:])
                    for prev_layer in non_cnf_layers[::-1]:
                        z_flatten = prev_layer(z_flatten, reverse=True)
                    z_inv = z_flatten.view(ntimes, BATCH_SIZE, *data_shape)
                    for t in range(1, z_inv.shape[0]):
                        z_t = z_inv[t]
                        save_imgs_figure(z_t)
                    z = z_traj[-1]
                else:
                    # update z and place in non_cnf_layers.
                    z = layer(z)
                    non_cnf_layers.append(layer)
            if idx < len(model.transforms) - 1:
                d = z.shape[1] // 2
                z, factor_out = z[:, :d], z[:, d:]
                non_cnf_layers.append(FactorOut(factor_out))

                # After every factor out, we half the time for visualization.
                time_ratio = time_ratio / 2
