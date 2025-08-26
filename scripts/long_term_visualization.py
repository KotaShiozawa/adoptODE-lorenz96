import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from glob import glob
import os

from util import git_dir

plt.style.use(f"{git_dir()}/scripts/paper_2col.mplstyle")

def plot_nrmse_time_resolved(results_filename: str) -> None:
    
    dataset = xr.open_dataset(results_filename)
    dataset["time"] = (dataset["time"] - dataset.attrs["len_segs"]/100) / 0.55 # rescale to lyapunov time

    time_norm = np.sqrt(
        (dataset.ground_truth**2).sum(dim="variable")
    ).mean(dim="time").mean(dim="seed_system")

    nrmse_data = np.sqrt(
        (
            (
                dataset.ground_truth 
                - dataset.reconstruction
            )**2
        ).sum(dim="variable")
    ) / time_norm


    average_d0 = nrmse_data.sel(
        time=0, method="nearest"
    ).mean(dim="seed_system").mean(dim="n_sys").values

    min_seed_sys_idx = np.abs(
        nrmse_data-average_d0
    ).sel(
        time=0, method="nearest"
    ).min(
        dim="n_sys"
    ).idxmin(dim="seed_system")

    min_n_sys_idx = np.abs(
        nrmse_data-average_d0
    ).sel(
        time=0, method="nearest",
        seed_system=min_seed_sys_idx,
    ).idxmin(dim="n_sys")

    max_dev_variable = np.abs(
        dataset.ground_truth.sel(
            seed_system=min_seed_sys_idx.values
        ) - dataset.reconstruction.sel(
            seed_system=min_seed_sys_idx.values, 
            n_sys=min_n_sys_idx.values
        )
    ).mean(dim="time").idxmax(dim="variable").values

    min_dev_variable = np.abs(
        dataset.ground_truth.sel(
            seed_system=min_seed_sys_idx.values
        ) - dataset.reconstruction.sel(
            seed_system=min_seed_sys_idx.values, 
            n_sys=min_n_sys_idx.values
        )
    ).mean(dim="time").idxmin(dim="variable").values


    fig, (ax1, ax0) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(3.416, 3.416 * 1.0)
    ) # type: ignore
    colors = ["tab:blue", "tab:red"]
    deviations = ["min", "max"]
    for i, variable in enumerate([min_dev_variable, max_dev_variable]):
        dataset.ground_truth.sel(
            seed_system=min_seed_sys_idx.values,
            variable=variable,
        ).plot(
            ax=ax0,
            x="time",
            color=colors[i],
            label=r"$n=n_{\text{%s}}$"%(deviations[i]),
        )
        dataset.reconstruction.sel(
            seed_system=min_seed_sys_idx.values,
            n_sys=min_n_sys_idx.values,
            variable=variable,
        ).plot(
            ax=ax0,
            x="time",
            color=colors[i],
            # label=r"$\hat{y}_{%d}(t)$"%(variable+1),
            linestyle="--",
        )
    ax0.set_title("")
    ax0.set_ylabel(r"$y^{(n)}(t; y_{0, i_0}, \hat{y}_{0, i_0, j_0})$")
    ax0.legend(ncol=2)
    for seed_system in nrmse_data.seed_system:
        nrmse_data.sel(
            seed_system=seed_system,
            time=slice(-100, 12.0)
        ).plot(
            ax=ax1,
            x="time",
            hue="n_sys",
            color="tab:orange",
            alpha=0.4,
            add_legend=False,
        )
    nrmse_data.sel(
        seed_system=min_seed_sys_idx.values,
        n_sys=min_n_sys_idx.values,
        time=slice(-100, 12.0)
    ).plot(
        ax=ax1,
        x="time",
        color="tab:purple",
        add_legend=False,
        lw=2,
        linestyle="dashed",
    )
    ax1.plot(
        np.arange(0, 12.0, 0.1),
        average_d0 * np.exp(np.arange(0, 12.0, 0.1)),
        color="black",
    )
    ax1.axhline(0.2, color="black", linestyle="--")
    ax0.set_xlabel(r"$\lambda t$")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$d(t)$") # = ||y(t)-\hat{y}(t)||_2 / \langle||y(t)||_2\rangle_t$")
    ax1.set_title("")
    ax1.set_yticks([1e-4, 1e-2, 0.2, 1e0])
    ax1.set_yticklabels(["$10^{-4}$", "$10^{-2}$", "0.2", "$10^{0}$"])
    ax1.set_xlim(nrmse_data.time.min().values, 12.0)
    handles_and_labels = [
        Line2D([0], [0], color="black", label= r"$\langle d(t=0)\rangle_{y_0, \hat{y}_0}\exp(\lambda t)$"),
        Line2D([0], [0], color="red", alpha=0.4, label=r"$d(t; \lbrace y_{0, i}\rbrace, \lbrace\hat{y}_{0, i, j}\rbrace)$"),
        Line2D([0], [0], color="tab:purple", linestyle="dashed", lw=2, label=r"$d(t; y_{0, i_0}, \hat{y}_{0, i_0, j_0})$"),
    ]
    ax1.legend(handles=handles_and_labels, loc="lower right")
    ax1.set_ylim(top=2.0)
    ax1.set_xlabel("")

    ax1.text(-0.2, 1.05, "(a)", transform=ax1.transAxes)
    ax0.text(-0.2, 1.05, "(b)", transform=ax0.transAxes)

    fig.tight_layout()

    fig.savefig(f"{git_dir()}/plots/nrmse_time_resolved-{dataset.attrs['len_segs']}.png", dpi=600)
    dataset.close()

    
if __name__ == "__main__":
    
    for len_segs in [35, 45, 55, 65, 75, 100]:
        list_of_files = glob(f"{git_dir()}/data/01_simulations/*long_term_prediction*len_segs{len_segs}-*")
        latest_file = max(list_of_files, key=os.path.getctime)

        plot_nrmse_time_resolved(
            results_filename=latest_file,
        )