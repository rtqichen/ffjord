import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal
import scipy.ndimage

# BASE = "experiments/cnf_mnist_64-64-128-128-64-64/logs"
# RESIDUAL = "experiments/cnf_mnist_64-64-128-128-64-64_residual/logs"
# RADEMACHER = "experiments/cnf_mnist_64-64-128-128-64-64_rademacher/logs"

BOTTLENECK = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64/logs"
BOTTLENECK_EST = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_ae-est/logs"
RAD_BOTTLENECK = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_rademacher/logs"
RAD_BOTTLENECK_EST = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_ae-est_rademacher/logs"

# ET_ALL = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_ae-est_residual_rademacher/logs"


def get_losses(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    losses = []

    for line in lines:
        w = re.findall(r"Bit/dim [^|(]*\([0-9\.]*\)", line)
        if w: w = re.findall(r"\([0-9\.]*\)", w[0])
        if w: w = re.findall(r"[0-9\.]+", w[0])
        if w:
            losses.append(float(w[0]))

    return losses


bottleneck_loss = get_losses(BOTTLENECK)
bottleneck_est_loss = get_losses(BOTTLENECK_EST)
rademacher_bottleneck_loss = get_losses(RAD_BOTTLENECK)
rademacher_bottleneck_est_loss = get_losses(RAD_BOTTLENECK_EST)

bottleneck_loss = scipy.signal.medfilt(bottleneck_loss, 21)
bottleneck_est_loss = scipy.signal.medfilt(bottleneck_est_loss, 21)
rademacher_bottleneck_loss = scipy.signal.medfilt(rademacher_bottleneck_loss, 21)
rademacher_bottleneck_est_loss = scipy.signal.medfilt(rademacher_bottleneck_est_loss, 21)

import seaborn as sns
sns.set_style("whitegrid")
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

import brewer2mpl
line_colors = brewer2mpl.get_map('Set2', 'qualitative', 4).mpl_colors
dark_colors = brewer2mpl.get_map('Dark2', 'qualitative', 4).mpl_colors
# plt.style.use('ggplot')

plt.figure(figsize=(4, 3))
plt.plot(np.arange(len(bottleneck_loss)) / 30, bottleneck_loss, ':', color=line_colors[1], label="Gaussian w/o Trick")
plt.plot(np.arange(len(bottleneck_est_loss)) / 30, bottleneck_est_loss, color=dark_colors[1], label="Gaussian w/ Trick")
plt.plot(np.arange(len(rademacher_bottleneck_loss)) / 30, rademacher_bottleneck_loss, ':', color=line_colors[2], label="Rademacher w/o Trick")
plt.plot(np.arange(len(rademacher_bottleneck_est_loss)) / 30, rademacher_bottleneck_est_loss, color=dark_colors[2], label="Rademacher w/ Trick")

plt.legend(frameon=True, fontsize=10.5, loc='upper right')
plt.ylim([1.1, 1.7])
# plt.yscale("log", nonposy='clip')
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Bits/dim", fontsize=18)
plt.xlim([0, 170])
plt.tight_layout()
plt.savefig('bottleneck_losses.pdf')
