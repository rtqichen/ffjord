# Free-form Jacobian of Reversible Dynamics (FFJORD)

Experiment code for "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" \[[arxiv](https://arxiv.org/abs/1810.01367)\].

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
python train_cnf.py
```

VAE Experiments (based on [Sylvester VAE](https://github.com/riannevdberg/sylvester-flows)):
```
python train_vae_flow.py
```

Glow / Real NVP experiments are run using `train_discrete_toy.py` and `train_discrete_tabular.py`.

## Datasets

### Tabular (UCI + BSDS300)
Follow instructions from https://github.com/gpapamak/maf and place them in `data/`.

### VAE datasets
Follow instructions from https://github.com/riannevdberg/sylvester-flows and place them in `data/`.

## Bibtex
```
@article{grathwohl2018ffjord,
  title={FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models},
  author={Grathwohl, Will and Chen, Ricky T. Q. and Bettencourt, Jesse and Sutskever, Ilya and Duvenaud, David},
  journal={arXiv preprint arXiv:1810.01367},
  year={2018}
}
```
