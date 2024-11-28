import argparse
import copy
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from jax import jit

import adoptODE
from adoptODE import dataset_adoptODE, simple_simulation, train_adoptODE


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


def gen_dataset(
    dataset_gt: adoptODE.Framework.dataset_adoptODE,  # type: ignore
    system_kwargs: dict,
    adoptODE_kwargs: dict,
    params: np.ndarray,
    num_segment: int = 0,
):

    mask = np.zeros(dataset_gt.ys["state"].shape, dtype=bool)
    mask[..., :: system_kwargs["observe_every"]] = 1
    mask_y0 = mask[:, 0, :]

    len_segs = system_kwargs["len_segs"]
    segment_evals = dataset_gt.t_evals[:len_segs]
    ys = {"state": dataset_gt.ys["state"] * jnp.nan}
    ys["state"][mask] = dataset_gt.ys["state"][mask]
    ys["state"] = dataset_gt.ys["state"][
        :, num_segment * len_segs : (num_segment + 1) * len_segs, :
    ]

    y0_train = {"state": dataset_gt.ys["state"][:, 0] * 0}

    if num_segment == 0:
        y0_train["state"][~mask_y0] = params.flat
        y0_train["state"][mask_y0] = dataset_gt.y0["state"][mask_y0]
    else:
        y0_train["state"] = params
        y0_train["state"][mask_y0] = dataset_gt.ys["state"][
            :, num_segment * len_segs, :
        ][mask_y0]

    y0_lower_bound = jnp.full(y0_train["state"].shape, -jnp.inf)
    y0_lower_bound = y0_lower_bound.at[mask_y0].set(y0_train["state"][mask_y0])

    y0_upper_bound = jnp.full(y0_train["state"].shape, jnp.inf)
    y0_upper_bound = y0_upper_bound.at[mask_y0].set(y0_train["state"][mask_y0])

    adoptODE_kwargs["lower_b_y0"] = {"state": y0_lower_bound}
    adoptODE_kwargs["upper_b_y0"] = {"state": y0_upper_bound}

    return dataset_adoptODE(
        define_system,
        ys,
        segment_evals,
        system_kwargs,
        adoptODE_kwargs,
        y0_train=y0_train,
    )


def training_loop(
    dataset_gt,
    system_kwargs: dict,
    adoptODE_kwargs: dict,
) -> xr.Dataset:

    initialization_key = jax.random.key(system_kwargs["seed_optimization"])

    all_values = dataset_gt.ys["state"][
        ..., :: system_kwargs["observe_every"]
    ].flatten()

    init_params = np.array(
        jax.random.choice(
            initialization_key, all_values, shape=dataset_gt.y0_train["state"].shape
        )
    )

    init_params = np.delete(
        init_params, np.s_[:: system_kwargs["observe_every"]], axis=-1
    )

    num_segments = system_kwargs["N_time_steps"] // system_kwargs["len_segs"]
    xr_datasets = []
    for segment in range(num_segments):

        print(f"Segment {segment+1}/{num_segments}")

        dataset_rec = gen_dataset(
            dataset_gt,
            system_kwargs,
            adoptODE_kwargs=adoptODE_kwargs,
            params=init_params,
            num_segment=segment,
        )

        params_final, losses, _, params_history = train_adoptODE(
            dataset_rec, save_interval=10, print_interval=100  # type: ignore
        )
        y0_hist = np.array([elem["y0"]["state"] for elem in params_history]).swapaxes(
            0, 1
        )

        losses = np.array(losses)
        reconstructed_dataset_sim = simple_simulation(
            define_system,
            np.array(
                [*dataset_rec.t_evals, dataset_rec.t_evals[-1] + system_kwargs["dt"]]
            ),
            system_kwargs,
            adoptODE_kwargs,
            y0=params_final["y0"],
        )
        init_params = reconstructed_dataset_sim.ys["state"][:, -1, :]  # type: ignore
        xr_datasets.append(
            xr.Dataset(
                {
                    "ground_truth": xr.DataArray(
                        dataset_gt.ys["state"][
                            jnp.newaxis,
                            :,
                            segment
                            * system_kwargs["len_segs"] : (segment + 1)
                            * system_kwargs["len_segs"],
                            ...,
                            jnp.newaxis,
                        ],
                        dims=["seed_system", "n_sys", "time", "variable", "segment"],
                        coords={
                            "n_sys": np.arange(0, system_kwargs["N_sys"]),
                            "time": dataset_gt.t_evals[
                                segment * 100 : (segment + 1) * 100
                            ],
                            "variable": np.arange(1, system_kwargs["D"] + 1),
                            "segment": [segment + 1],
                            "seed_system": [system_kwargs["seed_system"]],
                        },
                    ),
                    "reconstruction": xr.DataArray(
                        reconstructed_dataset_sim.ys["state"][jnp.newaxis, :, :-1, :, jnp.newaxis],  # type: ignore
                        dims=["seed_system", "n_sys", "time", "variable", "segment"],
                        coords={
                            "n_sys": np.arange(0, system_kwargs["N_sys"]),
                            "time": dataset_gt.t_evals[
                                segment * 100 : (segment + 1) * 100
                            ],
                            "variable": np.arange(1, system_kwargs["D"] + 1),
                            "segment": [segment + 1],
                            "seed_system": [system_kwargs["seed_system"]],
                        },
                    ),
                    "losses": xr.DataArray(
                        np.array(losses).T[jnp.newaxis, ..., jnp.newaxis],
                        dims=["seed_system", "n_sys", "epoch", "segment"],
                        coords={
                            "n_sys": np.arange(0, system_kwargs["N_sys"]),
                            "epoch": np.arange(0, 10 * len(losses), 10),
                            "segment": [segment + 1],
                            "seed_system": [system_kwargs["seed_system"]],
                        },
                    ),
                    "y0_history": xr.DataArray(
                        y0_hist[jnp.newaxis, ...],
                        dims=["seed_system", "n_sys", "epoch", "variable"],
                        coords={
                            "n_sys": np.arange(0, system_kwargs["N_sys"]),
                            "epoch": np.arange(0, 10 * len(losses), 10),
                            "variable": np.arange(1, system_kwargs["D"] + 1),
                            "seed_system": [system_kwargs["seed_system"]],
                        },
                    ),
                },
            )
        )

    return xr.concat(xr_datasets, dim="segment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--observe_every", type=int, default=1)
    parser.add_argument("--D", type=int, default=120)
    parser.add_argument("--N_sys", type=int, default=100)
    parser.add_argument("--seed_system", type=int, default=42)
    parser.add_argument("--N_time_steps", default=600)

    args = parser.parse_args()

    print(f"seed_system = {args.seed_system}")

    kwargs_sys = {
        "N_sys": args.N_sys,
        "D": args.D,
        "p": 8.17,
        "trans_steps": 1000,
        "N_time_steps": int(args.N_time_steps),
        "dt": 0.0065,
        "len_segs": 100,
        "observe_every": args.observe_every,
        "seed_system": args.seed_system,
    }

    t_evals = jnp.arange(
        0,
        (kwargs_sys["N_time_steps"] + kwargs_sys["trans_steps"]) * kwargs_sys["dt"],
        kwargs_sys["dt"],
    )

    kwargs_adoptODE = {
        "lr": 0.05,
        "epochs": 3000,
        "lr_y0": 0.05,
        "custom_scheduel_y0": optax.cosine_decay_schedule(
            0.05, 3000, alpha=1e-3, exponent=1.0
        ),
    }

    kwargs_adoptODE_to_save = kwargs_adoptODE.copy()

    seed = np.random.randint(0, np.iinfo(np.int32).max)
    kwargs_sys["seed_optimization"] = seed

    dataset_base = simple_simulation(
        define_system, t_evals, kwargs_sys, kwargs_adoptODE
    )

    # make a deep copy to keep the ground truth data
    dataset_ground_truth = copy.deepcopy(dataset_base)

    # remove the transient phase from the ground truth data
    dataset_ground_truth.ys["state"] = dataset_ground_truth.ys["state"][
        :, kwargs_sys["trans_steps"] :
    ]
    dataset_ground_truth.y0["state"] = dataset_ground_truth.ys["state"][:, 0, :]
    dataset_ground_truth.t_evals = jnp.arange(
        0, (kwargs_sys["N_time_steps"]) * kwargs_sys["dt"], kwargs_sys["dt"]
    )
    # get current date and time in YYYY-MM-DD_HH-MM-SS format
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reconstruction = training_loop(
        dataset_ground_truth, kwargs_sys, adoptODE_kwargs=kwargs_adoptODE
    )
    reconstruction.attrs = {**kwargs_sys, **kwargs_adoptODE_to_save}
    reconstruction.attrs["custom_scheduel_y0"] = "cosine_decay"

    for key, val in reconstruction.attrs.items():
        if val is None:
            reconstruction.attrs[key] = ["None"]
    reconstruction.to_netcdf(
        f"results/{timestamp}-D{args.D}-observe_every{args.observe_every}-seed_system{args.seed_system}.h5",
        engine="h5netcdf",
    )
