![cubewalkers](https://repository-images.githubusercontent.com/515983983/3fc4e7c8-4be4-486c-a727-415c676afebb)
# cubewalkers
CUDA simulation of walkers on Boolean network state transition hypercubes. 

# citation
If you use `cubewalkers` in your research, please cite us! 

#### KH Park, FX Costa, LM Rocha, R Albert, JC Rozum. Models of cell processes are far from the edge of chaos. PRX Life (2023, accepted)

This citation will be updated when the paper appears.

# requirements and installation
Requires `CuPy` (https://cupy.dev/), which in turn requires the `CUDA Toolkit` (https://developer.nvidia.com/cuda-toolkit) and a compatable GPU (https://developer.nvidia.com/cuda-gpus). See the `CuPy` installation guide (https://docs.cupy.dev/en/stable/install.html) for further information.

After installing these prerequisites, install `cubewalkers` via
```
pip install git+https://github.com/jcrozum/cubewalkers
```
Note that this `pip` command ***WILL NOT*** install the `CuPy` and `CUDA Toolkit` dependencies automatically. These are hardware-specific and must be installed manually.


# basic usage
In-depth guides to using `cubewalkers` are available in the [Examples and Tutorials](https://github.com/jcrozum/cubewalkers/tree/main/Examples%20and%20Tutorials) directory. In addition, [this repository](https://github.com/kyuhyongpark/cubewalkers-analysis) showcases a real-world application of `cubewalkers`.
For a quick overview of syntax, a short example is shown below:
```python
import cubewalkers as cw

# Define the rules. Various formats are accepted . . .
rules = '''
X,  !X
Y,  X & !Z
Z,  Y'''

# For example, these two versions of `rules` are equivalent.
rules = '''
X* = !X
Y* = X and not Z
Z* = Y'''

# Alternatively, rules can be imported from a text file by converting to a string, e.g.,
# with open('/path/to/rule/file.bnet') as f:
#  rules = f.read()

# Convert the `rules` string into a `cubewalkers` model
model = cw.Model(rules)

# Specify the default number of time steps and walkers (independent simulations)
model.n_time_steps = 25
model.n_walkers = 5

# Simulate the model ensemble (synchronous update is the default)
model.simulate_ensemble()

# The simulation results are stored in `model.trajectories` as a
# `n_time_steps` x `n_variables` x `n_walkers` array of node values at
# each timestep for each node and walker
print(model.trajectories)

# Simulate again, but this time,
# 1) only keep the last 10 time steps
# 2) only track the average values of nodes in the ensemble
# 3) use asynchronous update
# Points 1) and 2) are useful for memory efficiency in large simulations.
model.simulate_ensemble(
  T_window=10,
  averages_only=True,
  maskfunction=cw.update_schemes.asynchronous)

# The `model.trajectories` variable is overwritten. It now contains
# a `T_window` x `n_variables` array of averaged node values at each timestep
# for each node, averaged across `n_walkers` simulations.
print(model.trajectories)
```
In addition to the basic simulation options shown above, `cubewalkers` supports probabilistic Boolean networks (noisy dynamics), control interventions, and estimation of various dynamical quantities. 
