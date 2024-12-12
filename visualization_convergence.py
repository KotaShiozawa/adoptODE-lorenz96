import argparse
from datetime import datetime
from pathlib import Path
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Set the font size parameters
plt.rcParams.update(
    {
        "font.size": 9,  # Adjust this value to match your LaTeX document font size
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
    }
)


def exponential(x, a, b):
    return a * np.exp(b * x)


def hyperbolic_tan(x, a, b, c, d):
    return a * np.tanh(b * (x - c)) + d


def plot_convergence(
    e_true: xr.DataArray, threshold: float, savename: str, title: str = ""
):
    lower_threshold = 10 ** (np.log10(threshold) - 1)
    upper_threshold = 10 ** (np.log10(threshold) + 1)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    mn = (e_true.mean(dim="time").mean(dim="seed_system") > threshold).sum(dim="n_sys")
    lower = (e_true.mean(dim="time").mean(dim="seed_system") > lower_threshold).sum(
        dim="n_sys"
    )
    upper = (e_true.mean(dim="time").mean(dim="seed_system") > upper_threshold).sum(
        dim="n_sys"
    )
    ax.errorbar(e_true.segment, mn, yerr=[mn - lower, upper - mn], fmt="o", color="k")

    ax.set_ylabel(
        r"#unconverged trajectories ($E_{true} > 10^{%d}$)" % (np.log10(threshold))
    )
    ax.set_xlabel("segment")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)


def plot_mse_time_resolved(
    e_true: xr.DataArray, savename: str, title: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for segment in range(e_true.segment.size):
        e_true.isel(segment=segment).isel(seed_system=0).plot(
            ax=ax, hue="n_sys", x="time", add_legend=False, color="grey", alpha=0.3
        )  # type: ignore

    ax.set_yscale("log")
    ax.set_ylabel("true mean squared error")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)


def calc_mse_true(data: xr.Dataset) -> xr.DataArray:
    return ((data.reconstruction - data.ground_truth) ** 2).mean(dim="variable")


def calc_mse_obs(data: xr.Dataset, observe_every: int) -> xr.DataArray:
    return (
        (
            data.reconstruction.isel(variable=slice(0, -1, observe_every))
            - data.ground_truth.isel(variable=slice(0, -1, observe_every))
        )
        ** 2
    ).mean(dim="variable")


def collect_results(filepath: str, dt: float) -> xr.Dataset:
    files = glob(filepath)

    datasets = []
    for file in files:
        dataset = xr.open_dataset(file)
        if dataset.attrs["dt"] != dt:
            continue

        observe_every = dataset.attrs["observe_every"]
        seed_system = dataset.attrs["seed_system"]

        mse_true = calc_mse_true(dataset)
        mse_obs = calc_mse_obs(dataset, observe_every)
        datasets.append(
            xr.Dataset({"mse_true": mse_true, "mse_obs": mse_obs}).expand_dims(
                "seed_system", coords={"seed_system": [seed_system]}
            )
        )
        dataset.close()

    return xr.concat(datasets, dim="seed_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--dt", type=float, default=0.01)

    args = parser.parse_args()

    mse_results = collect_results(args.filepath, args.dt)
    observation_fraction = mse_results.attrs["observe_every"]
    D = mse_results.attrs["D"]

    now = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")

    plot_convergence(
        mse_results.mse_true,
        args.threshold,
        f"plots/{now}_convergence_observe_every{observation_fraction}-D{D}-dt{args.dt}-combined",
        title=f"D = {D}, every {observation_fraction}th variable observed",
    )
    plot_mse_time_resolved(
        mse_results.mse_true,
        f"plots/{now}_mse_observe_every{observation_fraction}-D{D}-dt{args.dt}-combined",
        title=f"D = {D}, every {observation_fraction}th variable observed",
    )
