import argparse
import copy
from glob import glob
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from jax import jit

from adoptODE import simple_simulation


def define_system(**kwargs_sys):
    def gen_y0():
        ini_state = jax.random.uniform(
            key=jax.random.key(kwargs_sys["seed_system"]),
            shape=(kwargs_sys["D"],),
            dtype=jnp.float32,
            minval=-1,
            maxval=4,
        )
        return {"state": ini_state}

    def gen_params():
        return {}, {}, {}

    @jit
    def eom(y, t, params, iparams, exparams):
        return {
            "state": (jnp.roll(y["state"], -1) - jnp.roll(y["state"], 2))
            * jnp.roll(y["state"], 1)
            - y["state"]
            + kwargs_sys["p"]
        }

    @jit
    def loss(ys, params, iparams, exparams, targets):
        x = ys["state"][..., :: kwargs_sys["observe_every"]]
        t_x = targets["state"][..., :: kwargs_sys["observe_every"]]
        return jnp.nanmean((x - t_x) ** 2)

    return eom, loss, gen_params, gen_y0, {}


def prediction_loop(
    system_kwargs: dict,
    adoptODE_kwargs: dict,
    results_filename: str,
    load_regex: str,
) -> None:

    for file in glob(load_regex):

        dataset = xr.open_dataset(file)
        if "initialization" in dataset.attrs.keys():
            continue
        if dataset.attrs["N_sys"] != 100: 
            continue
        if dataset.attrs["dt"] != system_kwargs["dt"]:
            continue

        print(f'processing file {file}')

        seed_system = dataset.attrs["seed_system"]

        last_ic = dataset.reconstruction.isel(
            segment=-1, seed_system=0, time=(dataset.segment.size - 1) * 100
        )

        prediction = simple_simulation(
            define_system,
            np.arange(
                dataset.time[(dataset.segment.size - 1) * 100],
                dataset.time[(dataset.segment.size - 1) * 100] + 6.5,
                step=dataset.attrs["dt"],
            ),
            system_kwargs,
            adoptODE_kwargs,
            y0={"state": last_ic.values},
        )

        last_ic_true = dataset.ground_truth.isel(
            segment=-1, seed_system=0, time=(dataset.segment.size - 1) * 100
        )
        truth = simple_simulation(
            define_system,
            np.arange(
                dataset.time[(dataset.segment.size - 1) * 100],
                dataset.time[(dataset.segment.size - 1) * 100] + 6.5,
                step=dataset.attrs["dt"],
            ),
            system_kwargs,
            adoptODE_kwargs,
            y0={"state": last_ic_true.values},
        )
        dataset.close()
        iteration_result = xr.Dataset(
            {
                "ground_truth": xr.DataArray(
                    truth.ys["state"][
                        jnp.newaxis,
                        ...,
                    ],
                    dims=["seed_system", "n_sys", "time", "variable"],
                    coords={
                        "n_sys": np.arange(0, system_kwargs["N_sys"]),
                        "time": truth.t_evals,
                        "variable": np.arange(1, system_kwargs["D"] + 1),
                        "seed_system": [seed_system],
                    },
                ),
                "reconstruction": xr.DataArray(
                    prediction.ys["state"][jnp.newaxis, ...],  # type: ignore
                    dims=["seed_system", "n_sys", "time", "variable"],
                    coords={
                        "n_sys": np.arange(0, system_kwargs["N_sys"]),
                        "time": prediction.t_evals,
                        "variable": np.arange(1, system_kwargs["D"] + 1),
                        "seed_system": [seed_system],
                    },
                ),
            },
        )
        saved_dset = xr.open_dataset(
            os.path.join("results/", results_filename), engine="h5netcdf"
        )
        merged_dset = xr.merge([saved_dset, iteration_result])
        saved_dset.close()
        merged_dset.to_netcdf(
            os.path.join("results/", results_filename), engine="h5netcdf"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe_every", type=int, default=1)
    parser.add_argument("--D", type=int, default=120)
    parser.add_argument("--N_sys", type=int, default=100)
    parser.add_argument("--N_time_steps", default=600)
    parser.add_argument("--dt", type=float, default=0.0065)
    parser.add_argument("--initialization", type=str, default="observed_dist")

    args = parser.parse_args()

    kwargs_sys = {
        "N_sys": args.N_sys,
        "D": args.D,
        "p": 8.17,
        "trans_steps": 10000,
        "N_time_steps": int(args.N_time_steps),
        "dt": args.dt,
        "len_segs": 100,
        "observe_every": args.observe_every,
    }

    kwargs_adoptODE = {
        "lr": 0.05,
        "epochs": 3000,
        "lr_y0": 0.05,
    }

    kwargs_adoptODE_to_save = kwargs_adoptODE.copy()

    # get current date and time in YYYY-MM-DD_HH-MM-SS format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reconstruction_attrs = {**kwargs_sys, **kwargs_adoptODE_to_save}

    dset = xr.Dataset(attrs=reconstruction_attrs)
    savename = f"{timestamp}-long_term_prediction_D{args.D}-observe_every{args.observe_every}.h5"
    load_regex = (
        f"results/*-D{args.D}-observe_every{args.observe_every}-seed_system*.h5"
    )
    dset.to_netcdf(
        os.path.join(
            "results/",
            savename,
        ),
        engine="h5netcdf",
    )
    prediction_loop(
        kwargs_sys,
        adoptODE_kwargs=kwargs_adoptODE,
        results_filename=savename,
        load_regex=load_regex,
    )
