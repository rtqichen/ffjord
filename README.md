# Free-form Jacobian of Reversible Dynamics (FFJORD)

Code for reproducing the experiments in the paper:

> Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." _International Conference on Learning Representations_ (2019).
> [[arxiv]](https://arxiv.org/abs/1810.01367) [[bibtex]](http://www.cs.toronto.edu/~rtqichen/bibtex/ffjord.bib)


## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Usage

Different scripts are provided for different datasets. To see all options, use the `-h` flag.

Toy 2d:
```
python train_toy.py --data 8gaussians --dims 64-64-64 --layer_type concatsquash --save experiment1
```

Tabular datasets from [MAF](https://github.com/gpapamak/maf):
```
python train_tabular.py --data miniboone --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 1000 --lr 1e-3
```

MNIST/CIFAR10:
```
python train_cnf.py --data mnist --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True
```

VAE Experiments (based on [Sylvester VAE](https://github.com/riannevdberg/sylvester-flows)):
```
python train_vae_flow.py --dataset mnist --flow cnf_rank --rank 64 --dims 1024-1024 --num_blocks 2
```

Glow / Real NVP experiments are run using `train_discrete_toy.py` and `train_discrete_tabular.py`.

## Datasets

### Tabular (UCI + BSDS300)
Follow instructions from https://github.com/gpapamak/maf and place them in `data/`.

### VAE datasets
Follow instructions from https://github.com/riannevdberg/sylvester-flows and place them in `data/`.

## Bespoke Flows

Here's a fun script that you can use to create your own 2D flow from an image!
```
python train_img2d.py --img imgs/github.png --save github_flow
```

<p align="center">
<img align="middle" src="./assets/github_flow.gif" width="400" height="400" />
</p>
