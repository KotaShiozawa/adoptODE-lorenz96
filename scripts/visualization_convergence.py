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
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    (e_true.mean(dim="time") > threshold).sum(dim="n_sys").plot(
        ax=ax, hue="seed_system", x="segment", color="grey", alpha=0.3, add_legend=False
    )

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
        for seed_system in range(e_true.seed_system.size):
            e_true.isel(segment=segment).isel(seed_system=seed_system).plot(
                ax=ax, hue="n_sys", x="time", add_legend=False, color="grey", alpha=0.3
            )  # type: ignore

    ax.set_yscale("log")
    ax.set_ylabel("true mean squared error")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)

def plot_mse_violin(e_true: xr.DataArray, e_obs: xr.DataArray, savename: str) -> None:
    """plots a violin plot of the MSE for each estimated segment"""


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
        print(f"file = {file}")
        dataset = xr.open_dataset(file)
        if dataset.attrs["dt"] != dt:
            dataset.close()
            continue

        print(f"n_sys = {dataset.n_sys.size}")

        observe_every = dataset.attrs["observe_every"]

        mse_true = calc_mse_true(dataset)
        mse_obs = calc_mse_obs(dataset, observe_every)
        datasets.append(
            xr.Dataset({"mse_true": mse_true, "mse_obs": mse_obs}, attrs=dataset.attrs)
        )
        dataset.close()

    return xr.concat(datasets, dim="seed_system")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--threshold", type=float, default=1e-5)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--observe_every", type=int, default=3)

    args = parser.parse_args()
    observation_fraction = args.observe_every

    filepath = f"{args.filepath}/2024-12*observe_every{observation_fraction}*.h5"

    mse_results = collect_results(filepath, args.dt)
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
