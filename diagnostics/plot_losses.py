import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "experiments/cnf_mnist_64-64-128-128-64-64/logs"
RESIDUAL = "experiments/cnf_mnist_64-64-128-128-64-64_residual/logs"
RADEMACHER = "experiments/cnf_mnist_64-64-128-128-64-64_rademacher/logs"

BOTTLENECK = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64/logs"
BOTTLENECK_EST = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_ae-est/logs"
ET_ALL = "experiments/cnf_mnist_bottleneck_64-64-128-5-128-64-64_ae-est_residual_rademacher/logs"


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


base_loss = get_losses(BASE)
residual_loss = get_losses(RESIDUAL)
rademacher_loss = get_losses(RADEMACHER)
bottleneck_loss = get_losses(BOTTLENECK)
bottleneck_est_loss = get_losses(BOTTLENECK_EST)
all_losses = get_losses(ET_ALL)

plt.plot(np.arange(len(base_loss)), base_loss, label="Baseline")
plt.plot(np.arange(len(residual_loss)), residual_loss, label="Residual")
plt.plot(np.arange(len(rademacher_loss)), rademacher_loss, label="Rademacher")
plt.plot(np.arange(len(bottleneck_loss)), bottleneck_loss, label="Bottleneck")
plt.plot(np.arange(len(bottleneck_est_loss)), bottleneck_est_loss, label="Bottleneck (LowVarEst)")
plt.plot(np.arange(len(all_losses)), all_losses, label="Bottleneck+Rademacher+Residual")

plt.legend()
plt.ylim([1, 2])
plt.yscale("log", nonposy='clip')
plt.xlabel("Iteration")
plt.ylabel("Bits/dim")
plt.savefig('losses.pdf')
