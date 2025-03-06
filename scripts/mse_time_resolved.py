"""plots time resolved mse"""

from glob import glob

import matplotlib.pyplot as plt
import xarray as xr
from util import git_dir


def mse_time_resolved_plot(
    segment_lengths: tuple[int, ...] = (35, 45, 55, 65, 75, 100)
) -> None:
    """collects data for segment lengths and plots time resolved mse

    Args:
        segment_lengths (tuple[int, ...], optional): segment lengths to use. Defaults to (35, 45, 55, 65, 75, 100).
    """
    fig, axs = plt.subplots(nrows=2, ncols=3)

    for ax, segment_length in zip(axs.flat, segment_lengths):  # type: ignore
        for file in glob(f"{git_dir()}/data/01_simulations/*quartiles_uniform.h5"):
            data = xr.open_dataset(file)
            if data.attrs["len_segs"] == segment_length:
                error = (
                    (
                        (
                            data.ground_truth.isel(variable=slice(0, -1, 3))
                            - data.reconstruction.isel(variable=slice(0, -1, 3))
                        )
                        ** 2
                    )
                    .mean(dim="variable")
                    .mean(dim="segment")
                )
                error.plot(
                    ax=ax,
                    x="time",
                    hue="n_sys",
                    color="grey",
                    alpha=0.5,
                    add_legend=False,
                )
                ax.set_yscale("log")
                ax.set_title(r"$t_S = %.2f$" % (segment_length / 100))
                ax.set_xlabel("Time")
                ax.set_ylabel(r"$E_{\mathrm{true}}$")
                break
    fig.tight_layout()
    fig.savefig(f"{git_dir()}/plots/mse_time_resolved.png")


if __name__ == "__main__":
    mse_time_resolved_plot()
