import argparse
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

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

    # ydata = (e_true.mean(dim="time") > threshold).sum(dim="n_sys").values.flatten()
    # xdata = e_true.segment.values

    # popt, pcov = curve_fit(hyperbolic_tan, xdata, ydata, p0=[100, -0.2, 10, 100])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    (e_true.mean(dim="time") > threshold).sum(dim="n_sys").plot.scatter(
        ax=ax, x="segment", color="grey", marker="d"
    )
    # x = np.linspace(xdata.min(), xdata.max(), 100)
    # ax.plot(x, hyperbolic_tan(x, *popt), "r--", label=r"fit: $f(x)\sim \tanh(x)$")
    ax.set_ylabel(
        r"#unconverged trajectories ($E_{true} > 10^{%d}$)" % (np.log10(threshold))
    )
    ax.set_xlabel("segment")
    # ax.legend(frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)


def plot_mse_time_resolved(
    e_true: xr.DataArray, savename: str, title: str = ""
) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for segment in range(e_true.segment.size):
        e_true.isel(segment=segment).plot(
            ax=ax, hue="n_sys", x="time", add_legend=False, color="grey", alpha=0.3
        )

    ax.set_yscale("log")
    ax.set_ylabel("true mean squared error")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)


def calc_mse_true(data: xr.Dataset) -> xr.DataArray:
    return ((data.reconstruction - data.ground_truth) ** 2).mean(dim="variable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    parser.add_argument("--threshold", type=float, default=1e-5)

    args = parser.parse_args()

    dataset = xr.open_dataset(args.filepath)

    observe_every = dataset.attrs["observe_every"]
    D = dataset.attrs["D"]
    seed_system = dataset.attrs["seed_system"]

    mse_true = calc_mse_true(dataset)
    dataset.close()

    now = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")

    plot_convergence(
        mse_true,
        args.threshold,
        f"plots/{now}_convergence_observe_every{observe_every}-D{D}-seed_{seed_system}",
        title=f"D = {D}, every {observe_every}th variable observed",
    )
    plot_mse_time_resolved(
        mse_true,
        f"plots/{now}_mse_observe_every{observe_every}-D{D}-seed_{seed_system}",
        title=f"D = {D}, every {observe_every}th variable observed",
    )
