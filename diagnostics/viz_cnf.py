from inspect import getsourcefile
import sys
import os
import subprocess

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.spectral_norm as spectral_norm
import lib.utils as utils


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
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
    return train_loader, test_loader, data_shape


def add_spectral_norm(model):
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


def build_model(args, state_dict):
    # load dataset
    train_loader, test_loader, data_shape = get_dataset(args)

    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    # neural net that parameterizes the velocity field
    if args.autoencode:

        def build_cnf():
            autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                hidden_dims=hidden_dims,
                input_shape=data_shape,
                strides=strides,
                conv=args.conv,
                layer_type=args.layer_type,
                nonlinearity=args.nonlinearity,
            )
            odefunc = layers.AutoencoderODEfunc(
                autoencoder_diffeq=autoencoder_diffeq,
                divergence_fn=args.divergence_fn,
                residual=args.residual,
                rademacher=args.rademacher,
            )
            cnf = layers.CNF(
                odefunc=odefunc,
                T=args.time_length,
                solver=args.solver,
            )
            return cnf
    else:

        def build_cnf():
            diffeq = layers.ODEnet(
                hidden_dims=hidden_dims,
                input_shape=data_shape,
                strides=strides,
                conv=args.conv,
                layer_type=args.layer_type,
                nonlinearity=args.nonlinearity,
            )
            odefunc = layers.ODEfunc(
                diffeq=diffeq,
                divergence_fn=args.divergence_fn,
                residual=args.residual,
                rademacher=args.rademacher,
            )
            cnf = layers.CNF(
                odefunc=odefunc,
                T=args.time_length,
                solver=args.solver,
            )
            return cnf

    chain = [layers.LogitTransform(alpha=args.alpha), build_cnf()]
    if args.batch_norm:
        chain.append(layers.MovingBatchNorm2d(data_shape[0]))
    model = layers.SequentialFlow(chain)

    if args.spectral_norm:
        add_spectral_norm(model)

    model.load_state_dict(state_dict)

    return model, test_loader.dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualizes experiments trained using train_cnf.py.")
    parser.add_argument("--checkpt", type=str, required=True)
    parser.add_argument("--nsamples", type=int, default=50)
    parser.add_argument("--ntimes", type=int, default=100)
    parser.add_argument("--save", type=str, default="imgs")
    args = parser.parse_args()

    checkpt = torch.load(args.checkpt, map_location=lambda storage, loc: storage)
    ck_args = checkpt["args"]
    state_dict = checkpt["state_dict"]

    model, test_set = build_model(ck_args, state_dict)
    real_samples = torch.stack([test_set[i][0] for i in range(args.nsamples)], dim=0)
    data_shape = real_samples.shape[1:]
    fake_latents = torch.randn(args.nsamples, *data_shape)

    # Transfer to GPU if available.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on {}".format(device))
    model.to(device)
    real_samples = real_samples.to(device)
    fake_latents = fake_latents.to(device)

    # Construct fake samples
    fake_samples = model(fake_latents, reverse=True).view(-1, *data_shape)
    samples = torch.cat([real_samples, fake_samples], dim=0)

    still_diffeq = torch.zeros_like(samples)
    im_indx = 0

    # Image-saving helper function
    def save_im(im, diffeq):
        global im_indx
        filename = os.path.join(current_dir, args.save, "flow_%05d.png" % im_indx)
        utils.makedirs(os.path.dirname(filename))

        diffeq = diffeq.clone()
        de_min, de_max = float(diffeq.min()), float(diffeq.max())
        diffeq.clamp_(min=de_min, max=de_max)
        diffeq.add_(-de_min).div_(de_max - de_min + 1e-5)

        assert im.shape == diffeq.shape
        shape = im.shape
        interleaved = torch.stack([im, diffeq]).transpose(0, 1).contiguous().view(2 * shape[0], *shape[1:])
        save_image(interleaved, filename, nrow=20, padding=0, range=(0, 1))
        im_indx += 1

    # Still frames with image samples.
    for _ in range(30):
        save_im(samples, still_diffeq)

    # Forward image to latent.
    logits = model.chain[0](samples)
    for i in range(1, len(model.chain)):
        assert isinstance(model.chain[i], layers.CNF)
        cnf = model.chain[i]
        tt = torch.linspace(cnf.integration_times[0], cnf.integration_times[-1], args.ntimes)
        z_t = cnf(logits, integration_times=tt)
        logits = z_t[-1]

        # transform back to image space
        im_t = model.chain[0](z_t.view(args.ntimes * args.nsamples * 2, *data_shape),
                              reverse=True).view(args.ntimes, 2 * args.nsamples, *data_shape)

        # save each step as an image
        for t, im in zip(tt, im_t):
            diffeq = cnf.odefunc(t, (im, None))[0]
            diffeq = model.chain[0](diffeq, reverse=True)
            save_im(im, diffeq)

    # Still frames with latent samples.
    latents = model.chain[0](logits, reverse=True)
    for _ in range(30):
        save_im(latents, still_diffeq)

    # Forward image to latent.
    for i in range(len(model.chain) - 1, 0, -1):
        assert isinstance(model.chain[i], layers.CNF)
        cnf = model.chain[i]
        tt = torch.linspace(cnf.integration_times[-1], cnf.integration_times[0], args.ntimes)
        z_t = cnf(logits, integration_times=tt)
        logits = z_t[-1]

        # transform back to image space
        im_t = model.chain[0](z_t.view(args.ntimes * args.nsamples * 2, *data_shape),
                              reverse=True).view(args.ntimes, 2 * args.nsamples, *data_shape)
        # save each step as an image
        for t, im in zip(tt, im_t):
            diffeq = cnf.odefunc(t, (im, None))[0]
            diffeq = model.chain[0](diffeq, reverse=True)
            save_im(im, -diffeq)

    # Combine the images into a movie
    bashCommand = r"ffmpeg -y -i {}/flow_%05d.png {}".format(
        os.path.join(current_dir, args.save), os.path.join(current_dir, args.save, "flow.mp4")
    )
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
