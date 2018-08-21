import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

LOW = -4
HIGH = 4


def plt_potential_func(potential, ax, npts=200, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow(prior_logdensity, transform, ax, npts=400, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    logqz = torch.sum(logqz, dim=1)[:, None]
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_axis_bgcolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=200, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)
    z, delta_logp = inverse_transform(x, zeros)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    plt.imshow(px)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, title="$x ~ q(x)$", device="cpu"):
    z = prior_sample(10000, 2).type(torch.float32).to(device)
    zk = transform(z).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=200)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_samples(samples, ax, title="$x ~ p(x)$"):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=200)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def visualize_samples(prior_sample, transform, device="cpu", num_samples=100, post_process=lambda x: x):
    z = prior_sample(num_samples).type(torch.float32).to(device)
    zk = transform(z)
    x = post_process(zk).cpu().numpy().reshape([-1, 28, 28, 1])
    x = np.clip(x, 0., 1.)
    x = x.reshape([10, 10, 28, 28, 1])
    x = np.hstack([_x for _x in x])
    x = np.hstack([_x for _x in x])
    return x


def visualize_transform(
    potential_or_samples, prior_sample, prior_density, transform, inverse_transform=None, samples=True, device="cpu"
):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")
    if samples:
        plt_samples(potential_or_samples, ax)
    else:
        plt_potential_func(potential_or_samples, ax)

    ax = plt.subplot(1, 3, 2, aspect="equal")
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")
    plt_flow_samples(prior_sample, transform, ax, device=device)
