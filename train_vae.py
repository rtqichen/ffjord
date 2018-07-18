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
import torch.functional as F
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torch.utils.data import Dataset
from torchvision.utils import save_image

import integrate

import lib.layers as layers
import lib.regularizations as regularizations
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

# go fast boi!!
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("Continuous Normalizing Flow VAE")
parser.add_argument(
    "--data", choices=["mnist", "svhn"], type=str, default="mnist"
)
parser.add_argument("--dims", type=str, default="256,256")
parser.add_argument("--hidden_dim", type=int, default=64)

parser.add_argument("--layer_type", type=str, default="concat", choices=["ignore", "concat", "hyper", "blend"])
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "gated"])

parser.add_argument("--time_length", type=float, default=1.0)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)

parser.add_argument("--adjoint", type=eval, default=True, choices=[True, False])

# Regularizations
parser.add_argument("--l2_coeff", type=float, default=0, help="L2 on dynamics.")
parser.add_argument("--dl2_coeff", type=float, default=0, help="Directional L2 on dynamics.")

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/vae")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()


class GatedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedLinear, self).__init__()
        self.layer_f = nn.Linear(in_features, out_features)
        self.layer_g = nn.Linear(in_features, out_features)

    def forward(self, x):
            f = self.layer_f(x)
            g = torch.sigmoid(self.layer_g(x))
            return f * g


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GatedConv, self).__init__()
        self.layer_f = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=1, groups=groups
        )
        self.layer_g = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=1, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


class GatedConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1):
        super(GatedConvTranspose, self).__init__()
        self.layer_f = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups
        )
        self.layer_g = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, groups=groups
        )

    def forward(self, x):
        f = self.layer_f(x)
        g = torch.sigmoid(self.layer_g(x))
        return f * g


def binarize(x):
    """
    [0, 1] -> {0, 1}
    """
    noise = x.new().resize_as_(x).uniform_()
    binary = (noise < x).type_as(x)
    return binary



def binarized_mnist(path="./data/binarized_mnist.npz"):
    data = np.load(path)
    rs = lambda x: x.reshape([-1, 1, 28, 28])
    return rs(data['train_data']), rs(data['valid_data']), rs(data['test_data'])


def Encoder():
    return nn.Sequential(
        GatedConv(1,  32,  5, 1, 2),
        GatedConv(32, 32,  5, 2, 2),
        GatedConv(32, 64,  5, 1, 2),
        GatedConv(64, 64,  5, 2, 2),
        GatedConv(64, 64,  5, 1, 2),
        GatedConv(64, 64,  5, 1, 2),
        GatedConv(64, 256, 7, 1, 0)
    )


def Decoder():
    return nn.Sequential(
        GatedConvTranspose(64, 64, 7, 1, 0),
        GatedConvTranspose(64, 64, 5, 1, 2),
        GatedConvTranspose(64, 32, 5, 2, 2, 1),
        GatedConvTranspose(32, 32, 5, 1, 2),
        GatedConvTranspose(32, 32, 5, 2, 2, 1),
        GatedConvTranspose(32, 32, 5, 1, 2),
        GatedConvTranspose(32, 1,  1, 1, 0),
    )


class ODEnet(nn.Module):
    def __init__(self, hidden_dims, nonlinearity, layer_type, input_dim):
        super(ODEnet, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity == "gated":
            assert layer_type == "concat", "only concat layers are supported for gated nonlinearity"
            ls = []
            for i in range(len(hidden_dims)):
                dim_in = input_dim if i == 0 else hidden_dims[i - 1]
                dim_out = hidden_dims[i]
                cur_layer = layers.ConcatLinear([dim_in], dim_out, layer_type=GatedLinear)
                ls.append(cur_layer)
            ls.append(layers.ConcatLinear([hidden_dims[-1]], input_dim))
            self.layers = nn.ModuleList(ls)
        else:
            nonlinearity = {"tanh": nn.Tanh, "relu": nn.ReLU, "softplus": nn.Softplus, "elu": nn.ELU}[nonlinearity]
            self.nonlinearity = nonlinearity()
            base_layer = {"ignore": layers.IgnoreLinear, "hyper": layers.HyperLinear, "concat": layers.ConcatLinear,
                          "blend": layers.BlendLinear}[layer_type]
            ls = []
            for i in range(len(hidden_dims)):
                dim_in = input_dim if i == 0 else hidden_dims[i - 1]
                dim_out = hidden_dims[i]
                ls.append(base_layer([dim_in], dim_out))
            ls.append(base_layer([hidden_dims[-1]], input_dim))
            self.layers = nn.ModuleList(ls)

    def forward(self, t, y):
        dx = y
        if self.nonlinearity == "gated":
            for layer in self.layers:
                dx = layer(t, dx)
            return dx
        else:
            # for each non-terminal layer apply layer and apply nonlinearity
            for layer in self.layers[:-1]:
                dx = layer(t, dx)
                dx = self.nonlinearity(dx)
            dx = self.layers[-1](t, dx)
            return dx


class VAE(nn.Module):
    def __init__(self, hidden_dim, flow):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flow = flow
        dim_out = list(self.encoder.modules())[-1].out_channels
        self.hidden_dim = hidden_dim
        self.mu_layer = nn.Linear(dim_out, hidden_dim)
        self.logvar_layer = nn.Linear(dim_out, hidden_dim)
        self.nll = nn.BCEWithLogitsLoss(reduce=False)

    @staticmethod
    def _reparam(mu, logvar):
        eps = torch.normal(torch.zeros_like(mu), torch.ones_like(logvar))
        std = torch.exp(logvar / 2.)
        return mu + eps * std

    def forward(self, x, return_recons=False):
        # encode
        batch_size = x.size()[0]
        h = self.encoder(x).view(batch_size, -1)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        # sample z0 ~ N(mu, logvar)
        z0 = self._reparam(mu, logvar)

        # integrate flow
        zero = torch.zeros(x.shape[0], 1).to(x)
        zT, delta_logp = self.flow(z0, zero, reverse=False)
        logqz0 = standard_normal_logprob(z0).sum(dim=1)
        logqzT = logqz0 + delta_logp[:, 0]

        # get generaitve model likelihood and decode
        logpzT = standard_normal_logprob(zT).sum(dim=1)
        x_logit = self.decoder(zT[:, :, None, None])
        nll = self.nll(x_logit, x).view(batch_size, -1).sum(dim=1)
        logpx = -nll

        elbo = logpx + logpzT - logqzT
        if return_recons:
            return elbo, torch.sigmoid(x_logit)
        else:
            return elbo

    def sample(self, num_samples=100, z=None):
        if z is None:
            mu = torch.zeros(num_samples, self.hidden_dim)
            std = torch.ones(num_samples, self.hidden_dim)
            z = torch.normal(mu, std)

        x_logit = self.decoder(z[:, :, None, None])
        return torch.sigmoid(x_logit)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def get_dataset(args):
    if args.data == "mnist":
        train_set, valid_set, test_set = binarized_mnist()
        data_shape = (1, 28, 28)
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("==>>> total training batch number: {}".format(len(train_loader)))
    logger.info("==>>> total validation batch number: {}".format(len(valid_loader)))
    logger.info("==>>> total testing batch number: {}".format(len(test_loader)))
    return train_loader, valid_loader, test_loader, data_shape


def regularized_model(model):
    dict_of_regularizations = {}
    if args.l2_coeff != 0:
        dict_of_regularizations[regularizations.L2Regularization] = args.l2_coeff
    if args.dl2_coeff != 0:
        dict_of_regularizations[regularizations.DirectionalL2Regularization] = args.dl2_coeff

    for layer in model.chain:
        if isinstance(layer, layers.CNF):
            layer.odefunc = regularizations.RegularizationsContainer(layer.odefunc, dict_of_regularizations)
    return model


def get_regularization(model):
    reg_loss = 0
    for layer in model.chain:
        if isinstance(layer, layers.CNF):
            reg_loss += layer.odefunc.regularization_loss
    return reg_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # get deivce
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device)

    # load dataset
    train_loader, valid_loader, test_loader, data_shape = get_dataset(args)

    _odeint = integrate.odeint_adjoint if args.adjoint else integrate.odeint
    hidden_dims = tuple(map(int, args.dims.split(",")))

    gfunc = ODEnet(hidden_dims, args.nonlinearity, args.layer_type, args.hidden_dim)
    flow = layers.CNF(
        T=args.time_length,
        odeint=_odeint,
        input_shape=(args.hidden_dim,),
        gfunc=gfunc,
        divergence_fn=args.divergence_fn
    )
    vae = VAE(args.hidden_dim, flow)

    logger.info(vae)
    logger.info("Number of trainable parameters: {}".format(count_parameters(vae)))

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        vae.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    vae.to(device)

    # For visualization.
    fixed_z = cvt(torch.randn(100, np.prod(data_shape)))
    for valid_x in valid_loader:
        fixed_x = valid_x
        break

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)

    best_loss = float("inf")
    itr = (args.begin_epoch - 1) * len(train_loader)
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        for _, x in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()

            # cast data and move to device
            x = cvt(x)

            # compute loss
            elbo = vae(x)
            loss = -elbo.mean()
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(vae.flow.num_evals())

            if itr % args.log_freq == 0:
                logger.info(
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.4f}({:.4f}) | Steps {:.0f}({:.2f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val,
                        loss_meter.avg, steps_meter.val, steps_meter.avg
                    )
                )

            itr += 1

        # compute test loss
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for x in valid_loader:
                    x = cvt(x)
                    elbo = vae(x)
                    loss = -elbo.mean()
                    losses.append(loss.item())
                loss = np.mean(losses)
                logger.info(
                    "Epoch {:04d} | Time {:.4f}, Elbo {:.4f}".
                    format(epoch, time.time() - start, -loss)
                )
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": vae.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        with torch.no_grad():
            sample_filename = os.path.join(args.save, "samples", "{:04d}.jpg".format(epoch))
            recons_filename = os.path.join(args.save, "recons", "{:04d}.jpg".format(epoch))
            orig_filename = os.path.join(args.save, "recons", "orig.jpg".format(epoch))
            utils.makedirs(os.path.dirname(sample_filename))
            utils.makedirs(os.path.dirname(recons_filename))
            utils.makedirs(os.path.dirname(orig_filename))
            if args.data == "mnist":
                # samples
                generated_samples = vae.sample(z=fixed_z)
                save_image(generated_samples, sample_filename, nrow=10)
                # reconstructions
                x = cvt(fixed_x)
                elbo, recons = vae(x, return_recons=True)
                save_image(recons, recons_filename, nrow=10)
                save_image(fixed_x, orig_filename, nrow=10)

