# Free-form Jacobian of Reversible Dynamics (FFJORD)

Experiment code for "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models" \[[arxiv](https://arxiv.org/abs/1810.01367)\].

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Usage

Different scripts are provided for different datasets. To see all options, use the `-h` flag.

Toy 2d:
```
python train_toy.py
```

Tabular datasets from [MAF](https://github.com/gpapamak/maf):
```
python train_tabular.py
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

## Contact
Email rtqichen@cs.toronto.edu if you have questions about the code.

## Bibtex
```
@article{grathwohl2018ffjord,
  title={FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models},
  author={Grathwohl, Will and Chen, Ricky T. Q. and Bettencourt, Jesse and Sutskever, Ilya and Duvenaud, David},
  journal={arXiv preprint arXiv:1810.01367},
  year={2018}
}
```
