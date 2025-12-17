"""plots lyapunov exponents, Kaplan-Yorke dimension, autocorrelation and mutual information"""

import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from util import git_dir

plt.style.use(f"{git_dir()}/scripts/paper_2col.mplstyle")


def lyapunov_plot():
    lyapunov_data = {}

    with h5py.File(f"{git_dir()}/data/02_analysis/lyapunovs.h5") as f:
        for dimension in f.keys():  # pylint: disable=C0206
            lyapunov_data[int(dimension)] = {}
            lyapunov_data[int(dimension)]["spectrum"] = np.array(
                f[dimension]["spectrum"]
            )
            lyapunov_data[int(dimension)]["kaplanyorke"] = np.array(
                f[dimension]["ky_dim"]
            )

    fig, ax0 = plt.subplots()

    ax1 = ax0.twinx()
    for dimension, lyapdata in lyapunov_data.items():
        if dimension != 4:
            ax1.scatter(
                [dimension],
                1 / np.max(lyapdata["spectrum"]),
                marker="d",
                color="tab:red",
                s=8,
            )
            ax0.scatter(
                [dimension],
                lyapdata["kaplanyorke"],
                marker="o",
                color="tab:blue",
                s=8,
            )

    ax0.set_xlabel(r"dimension $D$")
    ax0.set_ylabel(r"$\Delta^{(KY)}$", color="tab:blue")
    ax0.tick_params(axis="y", labelcolor="tab:blue", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:red", color="tab:red")
    ax1.spines["left"].set_color("tab:blue")
    ax0.spines["left"].set_color("tab:blue")
    ax1.spines["right"].set_color("tab:red")
    ax0.spines["right"].set_color("tab:red")

    ax1.set_ylabel(r"$1/\lambda_{\mathrm{max}}$", color="tab:red")
    ax1.set_xticks([0, 10, 20, 30, 40, 50, 60, 80, 100, 120])

    plt.tight_layout()
    plt.savefig(f"{git_dir()}/plots/lyapunov_time_and_ky_dim.eps", dpi=300)
    fig.savefig(f"{git_dir()}/plots/lyapunov_time_and_ky_dim.png", dpi=300)


def autocorrelation_plot():
    autocorrelation_data = {}
    mutualinformation_data = {}

    with h5py.File(f"{git_dir()}/data/02_analysis/autocor_and_mi.h5", "r") as f:
        for dimension in f.keys():  # pylint: disable=C0206
            if dimension == "time":
                continue
            autocorrelation_data[int(dimension)] = {}
            mutualinformation_data[int(dimension)] = {}
            autocorrelation_data[int(dimension)]["mean"] = np.array(
                f[dimension]["autocor"]["mean"]
            )
            autocorrelation_data[int(dimension)]["std"] = np.array(
                f[dimension]["autocor"]["std"]
            )
            mutualinformation_data[int(dimension)]["mean"] = np.array(
                f[dimension]["mutualinfo"]["mean"]
            )
            mutualinformation_data[int(dimension)]["std"] = np.array(
                f[dimension]["mutualinfo"]["std"]
            )
        taus = np.array(f["time"])

    dimension = np.array(list(autocorrelation_data.keys()))

    # Normalize the dimension values to the range [0, 1]
    norm = plt.Normalize(min(dimension), max(dimension))

    # Create a colormap
    cmap = cm.viridis  # pylint: disable=E1101

    # Map the dimension values to colors
    colors = cmap(norm(dimension))

    fig = plt.figure(figsize=(3.416, 3.416))
    gs = GridSpec(nrows=11, ncols=8)

    ax1 = fig.add_subplot(gs[0:3, 1:-1])
    ax2 = fig.add_subplot(gs[3:6, 1:-1])
    colorbar_ax = fig.add_subplot(gs[:6, -1])
    colorbar_ax.axis("off")
    legend_ax = fig.add_subplot(gs[6:7, :])
    zerocrossing_ax = fig.add_subplot(gs[8:10, 1:])

    for (dim_acor, autocor), (dim_mi, mutualinfo) in zip(
        autocorrelation_data.items(), mutualinformation_data.items()
    ):
        ax1.plot(
            taus * 0.01,
            mutualinfo["mean"],
            color=colors[dim_mi - 5],
            zorder=dim_mi,
        )
        ax1.fill_between(
            taus * 0.01,
            mutualinfo["mean"] - mutualinfo["std"],
            mutualinfo["mean"] + mutualinfo["std"],
            alpha=0.5,
            color=colors[dim_mi - 5],
            zorder=dim_mi,
        )
        ax2.plot(
            taus * 0.01,
            autocor["mean"],
            color=colors[dim_acor - 5],
            zorder=dim_acor,
        )
        ax2.fill_between(
            taus * 0.01,
            autocor["mean"] - autocor["std"],
            autocor["mean"] + autocor["std"],
            alpha=0.5,
            color=colors[dim_acor - 5],
            zorder=dim_acor,
        )
        first_zero_crossing = np.where(autocor["mean"] < 0)[0][0]
        zerocrossing_ax.scatter(
            dim_acor,
            taus[first_zero_crossing] * 0.01,
            color="tab:blue",
            s=8,
            marker="o",
            label="first zero-crossing" if dim_acor == dimension[0] else None,
        )
        first_minimum = find_peaks(-autocor["mean"])[0][0]
        zerocrossing_ax.scatter(
            dim_acor,
            taus[first_minimum] * 0.01,
            color="tab:red",
            s=8,
            marker="d",
            label="first minimum" if dim_acor == dimension[0] else None,
        )

    # axs[0].sharex(axs[1])
    ax1.set_xticklabels([])
    ax2.set_xlabel(r"delay $\tau$")
    ax1.set_ylabel(r"$I(y(t), y(t-\tau))$")
    ax2.set_ylabel(r"$C(y(t), y(t-\tau))$")
    zerocrossing_ax.set_xlabel(r"dimension $D$")
    zerocrossing_ax.set_ylabel(r"$\tau$")

    # Add a colorbar to the right of the top axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=colorbar_ax,
        orientation="vertical",
        aspect=20,
        fraction=0.5,
        anchor=(-1, 0.5),
    )
    cbar.set_label(r"$D$")

    # Add labels (a) and (b) to the top left of the subplots
    ax1.text(
        -0.32,
        1.1,
        "(a)",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    ax2.text(
        -0.32,
        1.1,
        "(b)",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    zerocrossing_ax.text(
        -0.27,
        1.1,
        "(c)",
        transform=zerocrossing_ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    handles, labels = zerocrossing_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncols=2, bbox_to_anchor=[0.5, -2])
    legend_ax.axis("off")
    plt.subplots_adjust(hspace=1.2, top=0.95, bottom=0.05)
    fig.savefig(f"{git_dir()}/plots/autocorrelation_and_mi.eps", dpi=600)
    fig.savefig(f"{git_dir()}/plots/autocorrelation_and_mi.png", dpi=600)


if __name__ == "__main__":
    lyapunov_plot()
    autocorrelation_plot()
