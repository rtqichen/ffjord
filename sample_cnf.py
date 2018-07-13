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
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform

from collections import defaultdict

# go fast boi!!
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="samples/cnf")

parser.add_argument("--num_samples", type=int, default=10)

parser.add_argument("--num_interpolations", type=int, default=10)
parser.add_argument("--interpolation_steps", type=int, default=10)

parser.add_argument("--num_transformations", type=int, default=10)

new_args = parser.parse_args()


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def interpolate(x, y, num_interps):
    step = 1. / num_interps
    r = np.arange(0., 1. + step, step)
    vals = [x + (y - x) * t for t in r]
    vals = np.array(vals)
    return vals


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


if __name__ == "__main__":
    # restore parameters
    assert new_args.resume is not None
    if torch.cuda.is_available():
        checkpt = torch.load(new_args.resume)
    else:
        checkpt = torch.load(new_args.resume, map_location=lambda storage, location: storage)

    args = checkpt["args"]

    # get deivce
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device)

    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)

    _odeint = integrate.odeint_adjoint if args.adjoint else integrate.odeint
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    # build model
    cnf = layers.CNF(
        hidden_dims=hidden_dims, T=args.time_length, odeint=_odeint, input_shape=data_shape, strides=strides,
        conv=args.conv, layer_type=args.layer_type, divergence_fn=args.divergence_fn, nonlinearity=args.nonlinearity
    )
    model = layers.SequentialFlow([
        cnf,
        layers.LogitTransform(alpha=args.alpha),
    ])

    print(model)

    model.load_state_dict(checkpt["state_dict"])
    model.to(device)

    with torch.no_grad():
        print("Generating samples...")
        # generate the desired number of unconstrained samples
        for i in range(new_args.num_samples):
            fig_filename = os.path.join(new_args.save, "samples", "{:04d}.jpg".format(i))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(cvt(torch.randn(100, 784))).view(-1, 1, 28, 28)
            save_image(generated_samples, fig_filename, nrow=10)
            print(i)

        print("Generating interpolations...")
        # produce interpolations in latent space
        if new_args.num_interpolations > 0:
            ims = []
            # get latents for 1 batch
            for x, y in test_loader:
                x = cvt(x[:new_args.num_interpolations + 1])
                z = model(x, reverse=True)
                break
            latents = z.cpu().detach().numpy()
            for i in range(new_args.num_interpolations):
                interp = interpolate(latents[i], latents[i + 1], new_args.interpolation_steps)
                interp = cvt(torch.tensor(interp))
                generated_samples = model(interp).view(-1, 1, 28, 28)
                fig_filename = os.path.join(new_args.save, "interps", "{:04d}.jpg".format(i))
                utils.makedirs(os.path.dirname(fig_filename))
                save_image(generated_samples, fig_filename, nrow=new_args.interpolation_steps + 1)
                print(i)

        # produce data transformations (turn 1 into a 2)
        if new_args.num_transformations > 0:
            print("Generating transformations...")
            class_embeddings = defaultdict(list)
            # embed training data
            print("Embedding data...")
            for x, y in test_loader:
                z = model(cvt(x), reverse=True).cpu().detach().numpy()
                y = y.detach().numpy()
                for _z, _y in zip(z, y):
                    class_embeddings[_y].append(_z)

            # get mean embedding per class
            mean_embeddings = {y: np.array(z).mean(axis=0) for y, z in class_embeddings.items()}
            n_class = len(mean_embeddings.keys())

            for x, y in test_loader:
                x = cvt(x[:new_args.num_transformations])
                y = y[:new_args.num_transformations]
                z = model(x, reverse=True)
                break
            latents = z.cpu().detach().numpy()
            y = y.detach().numpy()
            all_new_latents = []
            print("Getting latents...")
            for latent, label in zip(latents, y):
                # get embedding for data point's true class
                class_embedding = mean_embeddings[label]
                for i in range(n_class):
                    cur_embedding = mean_embeddings[i]
                    new_embedding = latent + cur_embedding - class_embedding
                    all_new_latents.append(new_embedding)
                print('.')
            # generate new images
            all_new_latents = np.array(all_new_latents)
            all_new_latents = cvt(torch.tensor(all_new_latents))
            generated_samples = model(all_new_latents)
            print("Generating transformations...")
            for i in range(new_args.num_transformations):
                orig_im = x[i: i + 1]
                transformed_ims = generated_samples[i * n_class: (i + 1) * n_class]
                ims = torch.cat([orig_im, transformed_ims], 0).view(-1, 1, 28, 28)
                fig_filename = os.path.join(new_args.save, "transforms", "{:04d}.jpg".format(i))
                utils.makedirs(os.path.dirname(fig_filename))
                save_image(ims, fig_filename, nrow=n_class + 1)
                print(i)