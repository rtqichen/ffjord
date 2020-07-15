# DOvsOD_NeuralODEs 
Discretize-Optimize vs Optimize-Discretize for Neural ODES

## Associated Publication

Discretize-Optimize vs. Optimize-Discretize for Time-Series Regression and Continuous Normalizing Flows
https://arxiv.org/abs/2006.00104

Please cite as
    
    @article{onken2020discretizeoptimize,
        title={Discretize-Optimize vs. Optimize-Discretize for Time-Series Regression and Continuous Normalizing Flows},
        author={Derek Onken and Lars Ruthotto},
        year={2020},
        journal = {arXiv preprint arXiv:2005.13420},
    }


## Setup

There are two problem types, each with its own setup instructions and coding language:
CNFs (in Python) and Time Series Regression (in Julia)

#### CNFs Python Setup:

```
cd cnf_python # run all commands for the cnfs from this location
virtualenv -p python3 neurEnv
source neurEnv/bin/activate
pip install -r cnf_python/requirements.txt
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```


commands to run:

```
python3 train_tabular.py --data miniboone --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 5000 --test_batch_size 1000 --lr 1e-3 --solver rk4 --step_size 0.25 --test_solver dopri5 --save experiments/cnf/DO/miniboone/rk4 
```


#### Times Series Julia Setup:



