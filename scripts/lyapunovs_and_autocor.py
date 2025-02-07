"""plots lyapunov exponents, Kaplan-Yorke dimension, autocorrelation and mutual information"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import h5py
from scipy.signal import find_peaks

plt.style.use("../paper_2col.mplstyle")


def lyapunov_plot():
    lyapunov_data = {}

    with h5py.File("../data/02_analysis/lyapunovs.h5") as f:
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
    plt.savefig("lyapunov_time_and_ky_dim.eps", dpi=300)
    fig.savefig("lyapunov_time_and_ky_dim.png", dpi=300)


def autocorrelation_plot():
    autocorrelation_data = {}
    mutualinformation_data = {}

    with h5py.File("../data/02_analysis/autocor_and_mi.h5", "r") as f:
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

    fig, axs = plt.subplots(nrows=2, figsize=(3.416, 3.416 * 0.8))

    for dimension, autocor in autocorrelation_data.items():
        axs[0].plot(
            taus * 0.01,
            autocor["mean"],
            color=colors[dimension - 5],
            zorder=dimension,
        )
        axs[0].fill_between(
            taus * 0.01,
            autocor["mean"] - autocor["std"],
            autocor["mean"] + autocor["std"],
            alpha=0.5,
            color=colors[dimension - 5],
            zorder=dimension,
        )
        first_zero_crossing = np.where(autocor["mean"] < 0)[0][0]
        axs[1].scatter(
            dimension,
            taus[first_zero_crossing] * 0.01,
            color="tab:blue",
            s=8,
            marker="o",
            label="first zero-crossing" if dimension == dimension[0] else None,
        )
        first_minimum = find_peaks(-autocor["mean"])[0][0]
        axs[1].scatter(
            dimension,
            taus[first_minimum] * 0.01,
            color="tab:red",
            s=8,
            marker="d",
            label="first minimum" if dimension == dimension[0] else None,
        )

    axs[0].set_xlabel(r"delay $\tau$")
    axs[0].set_ylabel(r"$C(y(t), y(t-\tau))$")
    axs[1].set_xlabel(r"dimension $D$")
    axs[1].set_ylabel(r"$\tau$")

    # Add a colorbar to the right of the top axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[0], orientation="vertical", aspect=10)
    cbar.set_label(r"$D$")

    # Add labels (a) and (b) to the top left of the subplots
    axs[0].text(
        -0.32,
        1.1,
        "(a)",
        transform=axs[0].transAxes,
        fontsize=12,
        verticalalignment="top",
    )
    axs[1].text(
        -0.27,
        1.1,
        "(b)",
        transform=axs[1].transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    axs[1].legend(loc="center", ncols=2, bbox_to_anchor=(0.5, 1.25))

    plt.subplots_adjust(hspace=1.1)
    fig.savefig("../plots/autocorrelation.eps", dpi=300)
    fig.savefig("../plots/autocorrelation.png", dpi=300)

    fig, ax = plt.subplots()

    for dimension, mutualinfo in mutualinformation_data.items():
        ax.plot(
            taus * 0.01,
            mutualinfo["mean"],
            color=colors[dimension - 5],
            zorder=dimension,
        )
        ax.fill_between(
            taus * 0.01,
            mutualinfo["mean"] - mutualinfo["std"],
            mutualinfo["mean"] + mutualinfo["std"],
            alpha=0.5,
            color=colors[dimension - 5],
            zorder=dimension,
        )

    ax.set_xlabel(r"delay $\tau$")
    ax.set_ylabel(r"$I(y(t), y(t-\tau))$")
    # Add a colorbar to the right of the top axis
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", aspect=30, pad=0.3)
    cbar.set_label(r"$D$")
    fig.tight_layout()
    fig.savefig("../plots/mutualinformation.eps", dpi=300)
    fig.savefig("../plots/mutualinformation.png", dpi=300)


if __name__ == "__main__":
    lyapunov_plot()
    autocorrelation_plot()
