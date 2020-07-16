# Discretize-Optimize vs Optimize-Discretize for Neural ODEs

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
pip install -r requirements.txt
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

train and evaluate toy DO model:
```
python3 train_toy.py --data 8gaussians --solver do --step_size 0.05 --save experiments/cnf/toy/8gaussians/do 

python evaluate_toy.py --data 8gaussians --resume experiments/cnf/toy/8gaussians/do/checkpt.pth --solver do --batch_size 2000
```

train and evaluate miniboone DO model:

```
python3 train_tabular.py --data miniboone --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 5000 --test_batch_size 1000 --lr 1e-3 --solver do --step_size 0.25 --test_solver do --test_step_size 0.10   --save experiments/cnf/miniboone/DO/rk4 

python evaluate_tabular.py --data miniboone --resume experiments/cnf/miniboone/DO/rk4/checkpt.pth --batch_size 1000
```

To run other models, you'll need to download the preprocessed data from Papamakarios's MAF paper found at https://zenodo.org/record/1161203#.XbiVGUVKhgi. Place the data in the data folder. We've done miniboone for you since it's small.


#### Time Series Julia Setup:

```
# start up julia, may need to adjust the path
exec '/Applications/Julia-1.2.app/Contents/Resources/julia/bin/julia'

# change into the time series directory
cd("path/times_series_julia/")

# setup necessary packages
using Pkg
Pkg.add("Flux")
Pkg.add("DiffEqFlux")
Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("Printf")
Pkg.add("LaTeXStrings")
```

Run the time series comparison:
```
include("compareBoth.jl")
```





