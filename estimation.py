import os
import copy
import json
import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import jit
from jax import config
from jax.flatten_util import ravel_pytree
from adoptODE import train_adoptODE, simple_simulation

# config.update("jax_platform_name", "cpu")
config.update("jax_platform_name", "gpu")


def lorenz96(**kwargs_sys):
    vars = kwargs_sys["vars"]

    @jit
    def eom(y, t, params, iparams, exparams):
        p = params["p"]
        x = jnp.array([y[v] for v in vars])
        dx = jnp.array(jnp.roll(x, 1) * (jnp.roll(x, -1) - jnp.roll(x, 2)) - x + p)
        return dict(zip(vars, dx))

    @jit
    def loss(ys, params, iparams, exparams, targets):
        flat_fit = ravel_pytree(ys)[0]
        flat_target = ravel_pytree(targets)[0]
        return jnp.nanmean((flat_fit - flat_target) ** 2)

    def gen_params():
        return {}, {}, {}

    def gen_y0():
        y = kwargs_sys["init"]
        return dict(zip(vars, y))

    return eom, loss, gen_params, gen_y0, {}


# get commandline parameter "--every"
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--every", type=int, default=1)
args = parser.parse_args()
every = args.every


max_loops = 200
total_loops = 1000
trans = 3000
D = 420
N = 10000
dt = 0.01
p = 8.17
threshold = 10**-2
len_segs = 100
iguess_range = [-1, 4]
epochs = 3000
lr = 0
lr_y0 = 0.01
seed = 0

rng = np.random.default_rng(seed=seed)
vars = ["x" + str(i + 1).zfill(2) for i in range(D)]
vars_measured = ["x" + str(i + 1).zfill(2) for i in range(D) if i % every == 0]
kwargs_sys = {"N_sys": 1, "vars": vars, "init": rng.random(D)}


# Setting up system and training properties
num_segs = int(N / len_segs)
t_all = jnp.arange(0, (N + trans) * dt, dt)
t_evals = jnp.arange(0, len_segs * dt, dt)
kwargs_adoptODE = {"epochs": epochs, "lr": lr, "lr_y0": lr_y0}


name = "every" + str(every)
dir = os.path.join("results", name)
os.makedirs(dir)


estimated = np.zeros((D, N))
mse_true = np.zeros(num_segs)
mse_measured = np.zeros(num_segs)
counts = np.zeros(num_segs)
count = 0

true = simple_simulation(lorenz96, t_all, kwargs_sys, kwargs_adoptODE, params={"p": p})
true = np.array([true.ys[v][0][trans:] for v in vars])

i = 0
init_guess = "rand"
while i < num_segs and np.sum(counts) < total_loops:
    kwargs_sys["init"] = true[:, len_segs * i]
    dataset = simple_simulation(
        lorenz96,
        t_evals,
        kwargs_sys,
        kwargs_adoptODE,
        params={"p": p},
        params_train={"p": p},
    )
    ys_true = copy.deepcopy(dataset.ys)
    for v in sorted(list(set(vars) - set(vars_measured))):
        dataset.ys[v] = dataset.ys[v] * jnp.nan
        if init_guess == "rand":
            dataset.y0_train[v] = np.array(
                [rng.uniform(iguess_range[0], iguess_range[1])]
            )
        elif init_guess == "end":
            dataset.y0_train[v] = np.array([ys_sol[v][0, -1]])
    params_final, losses, errors, params_history = train_adoptODE(
        dataset, print_interval=100, save_interval=1
    )
    mse_measured_i = np.mean(
        (
            np.array([dataset.ys_sol[v].flatten() for v in vars_measured])
            - np.array([ys_true[v].flatten() for v in vars_measured])
        )
        ** 2
    )
    if mse_measured_i < threshold or count == max_loops - 1:
        init_guess = "end"
        ys_sol = copy.deepcopy(dataset.ys_sol)
        estimated[:, i * len_segs : (i + 1) * len_segs] = np.array(
            list(ys_sol.values())
        )[:, 0, :]
        mse_true[i] = np.mean(
            (ravel_pytree(dataset.ys_sol)[0] - ravel_pytree(ys_true)[0]) ** 2
        )
        mse_measured[i] = mse_measured_i
        counts[i] = count
        count = 0
        i += 1
    else:
        init_guess = "rand"
        count += 1


pd.DataFrame(estimated).to_csv(
    os.path.join(dir, "estimated.csv"), header=False, index=False
)
pd.DataFrame(mse_true).to_csv(
    os.path.join(dir, "mse_true.csv"), header=False, index=False
)
pd.DataFrame(mse_measured).to_csv(
    os.path.join(dir, "mse_measured.csv"), header=False, index=False
)
pd.DataFrame(true).to_csv(os.path.join(dir, "true.csv"), header=False, index=False)
pd.DataFrame(counts).to_csv(os.path.join(dir, "counts.csv"), header=False, index=False)


params = {}
params["every"] = every
params["max_loops"] = max_loops
params["total_loops"] = total_loops
params["trans"] = trans
params["D"] = D
params["N"] = N
params["dt"] = dt
params["p"] = p
params["threshold"] = threshold
params["len_segs"] = len_segs
params["iguess_range"] = iguess_range
params["epochs"] = epochs
params["lr"] = lr
params["lr_y0"] = lr_y0
params["seed"] = seed
params["vars"] = vars
params["vars_measured"] = vars_measured

with open(os.path.join(dir, "params.json"), "w") as f:
    json.dump(params, f, indent=4)
