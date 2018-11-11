import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MNIST_SINGLESCALE = "diagnostics/mnist.log"
MNIST_MULTISCALE = "diagnostics/mnist_multiscale.log"


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


mnist_singlescale_loss, mnist_singlescale_nfes = get_values(MNIST_SINGLESCALE)
mnist_multiscale_loss, mnist_multiscale_nfes = get_values(MNIST_MULTISCALE)

import brewer2mpl
line_colors = brewer2mpl.get_map('Set2', 'qualitative', 4).mpl_colors
dark_colors = brewer2mpl.get_map('Dark2', 'qualitative', 4).mpl_colors

import seaborn as sns
sns.set_style("whitegrid")
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

plt.figure(figsize=(4, 2.6))
plt.scatter(mnist_singlescale_nfes[::10], mnist_singlescale_loss[::10], color=line_colors[1], label="Single FFJORD")
plt.scatter(mnist_multiscale_nfes[::10], mnist_multiscale_loss[::10], color=line_colors[2], label="Multiscale FFJORD")

plt.ylim([0.9, 1.25])
plt.legend(frameon=True, fontsize=10.5)
plt.xlabel("NFE", fontsize=18)
plt.ylabel("Bits/dim", fontsize=18)

ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=10)

plt.tight_layout()
plt.savefig('multiscale_loss_vs_nfe.pdf')
