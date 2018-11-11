import os.path
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage

import seaborn as sns
sns.set_style("whitegrid")
colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

dims = [16, 32, 48, 64]
dirs = [
    'vae_mnist_cnf_num_flows_4_256-256_num_blocks_1__2018-09-16_17_27_03',
    'vae_mnist_cnf_num_flows_4_256-256_num_blocks_1__2018-09-16_17_26_41',
    'vae_mnist_cnf_num_flows_4_256-256_num_blocks_1__2018-09-16_17_23_35',
    'vae_mnist_cnf_num_flows_4_256-256_num_blocks_1__2018-09-16_17_25_03',
]

nfe_all = []

for dim, dirname in zip(dims, dirs):
    with open(os.path.join('snapshots', dirname, 'logs'), 'r') as f:
        lines = f.readlines()

    nfes_ = []

    for line in lines:
        w = re.findall(r"NFE Forward [0-9]*", line)
        if w: w = re.findall(r"[0-9]+", w[0])
        if w:
            nfes_.append(float(w[0]))

    nfe_all.append(nfes_)

plt.figure(figsize=(4, 2.4))
for i, (dim, nfes) in enumerate(zip(dims, nfe_all)):
    nfes = np.array(nfes)
    xx = (np.arange(len(nfes)) + 1) / 50
    nfes = scipy.ndimage.gaussian_filter(nfes, 101)
    plt.plot(xx, nfes, '--', label='Dim {}'.format(dim))

plt.legend(frameon=True, fontsize=10.5)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('NFE', fontsize=18)
plt.xlim([0, 200])
plt.tight_layout()
plt.savefig("nfes_vs_dim_vae.pdf")
