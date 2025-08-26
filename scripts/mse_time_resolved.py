"""plots time resolved mse"""

from glob import glob

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib.lines import Line2D
from util import git_dir

plt.style.use(f'{git_dir()}/scripts/paper_2col.mplstyle')

def mse_time_resolved_plot_all_segment_lengths(
    segment_lengths: tuple[int, ...] = (35, 45, 55, 65, 75, 100)
) -> None:
    """collects data for segment lengths and plots time resolved mse

    Args:
        segment_lengths (tuple[int, ...], optional): segment lengths to use. Defaults to (35, 45, 55, 65, 75, 100).
    """
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)

    for ax, segment_length in zip(axs.flat, segment_lengths):  # type: ignore
        for file in glob(f"{git_dir()}/data/01_simulations/*quartiles_uniform_restrict_to_hyperplanes.h5"):
            data = xr.open_dataset(file)
            if data.attrs["len_segs"] == segment_length and data.time.size > 200:
                error = (
                    (
                        (
                            data.ground_truth
                            - data.reconstruction
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
                if segment_length in [35, 65]:
                    ax.set_ylabel(r"$E_{\mathrm{true}}$")
                else:
                    ax.set_ylabel("")
                if segment_length == 75:
                    ax.set_xlabel("time")
                else:
                    ax.set_xlabel("")
                data.close()
                break
            data.close()
    fig.tight_layout()
    fig.savefig(f"{git_dir()}/plots/mse_time_resolved_all_segment_lengths.png", dpi=600)


def mse_time_resolved_plot_one_length(segment_length: int = 65) -> None:
    """collects data for segment lengths and plots time resolved mse

    Args:
        segment_lengths (tuple[int, ...], optional): segment lengths to use. Defaults to (35, 45, 55, 65, 75, 100).
    """
    fig, ax = plt.subplots() # type: ignore
    vline_locs = None
    emin = None
    emax = None
    for file in glob(f"{git_dir()}/data/01_simulations/*quartiles_uniform_restrict_to_hyperplanes.h5"):
        if "long_term_prediction" in file:
            continue
        print(f'processing {file}')
        data = xr.open_dataset(file)
        if data.attrs["len_segs"] == segment_length and data.time.size > 200:
            error = (
                np.sqrt(
                    (
                        (
                            data.ground_truth
                            - data.reconstruction
                    )
                    ** 2
                    ).sum(dim="variable")
                )
                .mean(dim="segment")
            )
            
            error.plot(
                ax=ax,
                x="time",
                hue="n_sys",
                color="tab:red",
                alpha=0.3,
                add_legend=False,
            )
            vline_locs = np.arange(0, data.segment.size+1, 1)*segment_length/100
            emin = error.min().values
            emax = error.max().values

        data.close()
    if vline_locs is not None and emin is not None and emax is not None:
        ax.vlines(
            x=vline_locs,
            ymin=emin,
            ymax=emax,
            color='black',
            linestyle='dashed',
        )
    ax.set_xlabel("time")
    ax.set_yscale("log")
    ax.set_ylabel(r"$E_{\mathrm{true}}$")
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(f"{git_dir()}/plots/mse_time_resolved_segment_length{segment_length}.png", dpi=600)


if __name__ == "__main__":
    # mse_time_resolved_plot_all_segment_lengths()
    for segment_length in [65]:
        mse_time_resolved_plot_one_length(segment_length=segment_length)
