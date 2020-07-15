# DOvsOD_NeuralODEs 
DO vs OD for neural ODES in Times Series Regression and Conitnuous Normalizing Flows


Setup:

```
virtualenv -p python3 neurEnv
source neurEnv/bin/activate
pip install -r requirements.txt
pip install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```


commands to run:

```
python3 train_tabular.py --data miniboone --nhidden 2 --hdim_factor 20 --num_blocks 1 --nonlinearity softplus --batch_size 5000 --test_batch_size 1000 --lr 1e-3 --solver rk4 --step_size 0.25 --test_solver dopri5 --save experiments/cnf/DO/miniboone/rk4 
```


