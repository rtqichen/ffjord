import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CIFAR10 = "diagnostics/cifar10_multiscale.log"
CIFAR10_SN = "diagnostics/cifar10_multiscale_sn.log"

MNIST = "diagnostics/mnist_multiscale.log"
MNIST_SN = "diagnostics/mnist_multiscale_sn.log"


def get_values(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    losses = []
    nfes = []

    for line in lines:

        w = re.findall(r"Steps [^|(]*\([0-9\.]*\)", line)
        if w: w = re.findall(r"\([0-9\.]*\)", w[0])
        if w: w = re.findall(r"[0-9\.]+", w[0])
        if w:
            nfes.append(float(w[0]))

            w = re.findall(r"Bit/dim [^|(]*\([0-9\.]*\)", line)
            if w: w = re.findall(r"\([0-9\.]*\)", w[0])
            if w: w = re.findall(r"[0-9\.]+", w[0])
            if w:
                losses.append(float(w[0]))

    return losses, nfes


cifar10_loss, cifar10_nfes = get_values(CIFAR10)
cifar10_sn_loss, cifar10_sn_nfes = get_values(CIFAR10_SN)
mnist_loss, mnist_nfes = get_values(MNIST)
mnist_sn_loss, mnist_sn_nfes = get_values(MNIST_SN)

import brewer2mpl
line_colors = brewer2mpl.get_map('Set2', 'qualitative', 4).mpl_colors
dark_colors = brewer2mpl.get_map('Dark2', 'qualitative', 4).mpl_colors
plt.style.use('ggplot')

# CIFAR10 plot
plt.figure(figsize=(6, 7))
plt.scatter(cifar10_nfes, cifar10_loss, color=line_colors[1], label="w/o Spectral Norm")
plt.scatter(cifar10_sn_nfes, cifar10_sn_loss, color=line_colors[2], label="w/ Spectral Norm")

plt.ylim([3, 5])
plt.legend(fontsize=18)
plt.xlabel("NFE", fontsize=30)
plt.ylabel("Bits/dim", fontsize=30)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=16)
ax.yaxis.set_ticks([3, 3.5, 4, 4.5, 5])

plt.tight_layout()
plt.savefig('cifar10_sn_loss_vs_nfe.pdf')

# MNIST plot
plt.figure(figsize=(6, 7))
plt.scatter(mnist_nfes, mnist_loss, color=line_colors[1], label="w/o Spectral Norm")
plt.scatter(mnist_sn_nfes, mnist_sn_loss, color=line_colors[2], label="w/ Spectral Norm")

plt.ylim([0.9, 2])
plt.legend(fontsize=18)
plt.xlabel("NFE", fontsize=30)
plt.ylabel("Bits/dim", fontsize=30)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=16)
# ax.yaxis.set_ticks([3, 3.5, 4, 4.5, 5])

plt.tight_layout()
plt.savefig('mnist_sn_loss_vs_nfe.pdf')
