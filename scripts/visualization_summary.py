from glob import glob
import git

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.optimize import curve_fit

from util import git_dir  #

plt.style.use("paper_2col.mplstyle")


def collect_files(
    observe_every: int,
    dt: float = 0.0065,
    N_sys: int = 100,
    len_segs: int = 100,
    initialization: str = "quartiles_uniform",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joints = []
    totals = []
    fits = []
    for file in glob(f"{git_dir()}/data/01_simulations/202*every{observe_every}*.h5"):
        data = xr.open_dataset(file)
        if (
            len(data.data_vars) == 0
            or data.attrs["dt"] != dt
            or data.attrs["N_sys"] != N_sys
            or data.attrs["len_segs"] != len_segs
            or data.attrs["initialization"] != initialization
        ):
            data.close()
            continue
        print(f"Processing {file}")
        mse_true = (
            ((data.reconstruction - data.ground_truth) ** 2)
            # .mean(dim="time")
            .mean(dim="variable")
        )
        mse_obs = (
            ((data.reconstruction - data.ground_truth) ** 2).isel(
                variable=slice(0, -1, observe_every)
            )
            # .mean(dim="time")
            .mean(dim="variable")
        )
        mse_true_ta = mse_true.mean(dim="time")
        mse_obs_ta = mse_obs.mean(dim="time")

        segments = []
        mse = []
        true = []
        down_slope = []
        down_slope_err = []
        up_slope = []
        up_slope_err = []

        for segment in range(data.segment.size):
            mse.append(mse_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["true"]
                * len(mse_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(mse_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            mse.append(mse_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["observed"]
                * len(mse_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(mse_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            )

            mnidx = (
                mse_true.isel(segment=segment)
                .mean(dim="n_sys")
                .idxmin(dim="time")
                .values[0]
            )

            def func(x, a, b):
                return a * x + b

            fit_data = np.log(
                mse_true.isel(segment=segment).sel(time=slice(0, mnidx)).dropna("time")
            ).mean(dim="n_sys")
            fit_err = (
                np.log(
                    mse_true.isel(segment=segment)
                    .sel(time=slice(0, mnidx))
                    .dropna("time")
                )
                .std(dim="n_sys")
                .values
            )
            fit_x = fit_data.time.values.flatten()

            if mnidx > segment * len_segs / 100 + 0.01:
                popt, pcov = curve_fit(
                    func,
                    fit_x.flatten(),
                    fit_data.values[0],
                    p0=[-1.0, 13],
                    sigma=fit_err[0],
                )
                down_slope.append(popt[0])
                down_slope_err.append(np.sqrt(np.diag(pcov))[0])
            else:
                down_slope.append(np.nan)
                down_slope_err.append(np.nan)

            if mnidx < segment * len_segs / 100 + len_segs / 100 - 0.02:
                fit_data_2 = np.log(
                    mse_true.isel(segment=segment)
                    .sel(time=slice(mnidx, 15.4))
                    .dropna("time")
                ).mean(dim="n_sys")
                fit_err_2 = (
                    np.log(
                        mse_true.isel(segment=segment)
                        .sel(time=slice(mnidx, 15.4))
                        .dropna("time")
                    )
                    .std(dim="n_sys")
                    .values
                )
                fit_x_2 = fit_data_2.time.values.flatten()
                popt_2, pcov_2 = curve_fit(
                    func, fit_x_2.flatten(), fit_data_2.values[0], sigma=fit_err_2[0]
                )

                up_slope.append(popt_2[0])
                up_slope_err.append(np.sqrt(np.diag(pcov_2))[0])

            else:
                up_slope.append(np.nan)
                up_slope_err.append(np.nan)

        mse = [item for sublist in mse for item in sublist]
        true = [item for sublist in true for item in sublist]
        segments = [item for sublist in segments for item in sublist]

        mse_df = pd.DataFrame(
            {
                "mse": mse,
                "type": true,
                "segment": segments,
            }
        )

        fit_df = pd.DataFrame(
            {
                "up_slope": up_slope,
                "down_slope": down_slope,
                "up_slope_err": up_slope_err,
                "down_slope_err": down_slope_err,
                "segment": data.segment.values,
            }
        )
        fits.append(fit_df)

        if np.any(np.isinf(mse_df.mse)):
            print(f"Found inf values in {file}")
            continue

        if np.any(np.isinf(mse_df.mse)):
            print(f"Found inf values in {file}")
            continue

        totals.append(mse_df)
        data.close()

    return pd.concat(totals), pd.concat(fits)


def plot_results():

    fig, (ax, dist_ax) = plt.subplots(nrows=2, figsize=(6.9, 4.0))  # type: ignore
    fmts = ["o-", "s-", "^-", "v-", "D-", "X-"]

    comp_dfs = []

    middle_threshold = 1e-4
    lower_threshold = 1e-5
    upper_threshold = 1e-3

    for i, len_segs in enumerate([35, 45, 55, 65, 75, 100]):
        total_df = pd.read_hdf(
            f"{git_dir()}/data/02_analysis/total_df3_dt0.01-len_segs{len_segs}.h5"
        )
        total_df["time"] = total_df["segment"] * len_segs / 100
        success_rates = {middle_threshold: [], upper_threshold: [], lower_threshold: []}

        for segment in total_df["segment"].unique():
            segment_df = total_df[total_df["segment"] == segment]
            denom = len(segment_df) / 2

            success_rate = (
                len(
                    segment_df[
                        np.logical_and(
                            segment_df["mse"] < middle_threshold,
                            segment_df["type"] == "true",
                        )
                    ]
                )
                / denom
            )
            success_rate_upper = (
                len(
                    segment_df[
                        np.logical_and(
                            segment_df["mse"] < upper_threshold,
                            segment_df["type"] == "true",
                        )
                    ]
                )
                / denom
            )
            success_rate_lower = (
                len(
                    segment_df[
                        np.logical_and(
                            segment_df["mse"] < lower_threshold,
                            segment_df["type"] == "true",
                        )
                    ]
                )
                / denom
            )
            # print(f'number of initial conditions considered for segment {segment}: {denom/100}')

            success_rates[middle_threshold].append(success_rate)
            success_rates[upper_threshold].append(success_rate_upper)
            success_rates[lower_threshold].append(success_rate_lower)

        success_rates[upper_threshold] = np.array(success_rates[upper_threshold])  # type: ignore
        success_rates[middle_threshold] = np.array(success_rates[middle_threshold])  # type: ignore
        success_rates[lower_threshold] = np.array(success_rates[lower_threshold])  # type: ignore

        ax.errorbar(
            total_df.time.unique(),
            success_rates[middle_threshold],
            yerr=(
                success_rates[middle_threshold] - success_rates[lower_threshold],  # type: ignore
                success_rates[upper_threshold] - success_rates[middle_threshold],  # type: ignore
            ),
            fmt=fmts[i],
            label=r"$t_s=%.2f$" % (len_segs / 100),
            capsize=3,
            capthick=1,
        )

        if len_segs in [100, 65]:
            total_df["len_segs"] = [len_segs] * len(total_df)
            comp_dfs.append(total_df)

    comp_df = pd.concat(comp_dfs)
    sns.violinplot(
        ax=dist_ax,
        data=comp_df[comp_df.type == "true"],
        x="time",
        y="mse",
        hue="len_segs",
        palette=["tab:red", "tab:brown"],
        inner=None,
        split=True,
        width=1,
        dodge=True,
        density_norm="width",
        log_scale=True,
    )

    ax.legend(
        bbox_to_anchor=(0.5, 1.25),
        loc="center",
        ncol=6,
    )

    current_xticks = dist_ax.get_xticks()
    dist_ax.set_xticks(current_xticks[::3])
    dist_ax.hlines(middle_threshold, -0.5, 28.5, color="black", linestyle="--")
    dist_ax.hlines(upper_threshold, -0.5, 28.5, color="black", linestyle="--")
    dist_ax.hlines(lower_threshold, -0.5, 28.5, color="black", linestyle="--")
    dist_ax.set_ylabel(r"$E_{\mathrm{true}}$")
    import matplotlib as mpl

    for i, violin in enumerate(dist_ax.findobj(mpl.collections.PolyCollection)):  # type: ignore
        if np.all(np.abs(violin.get_facecolor()[0] - colors.to_rgba("tab:red")) < 0.1):
            violin.set_hatch("////")

    ax.set_ylabel("success rate")
    ax.set_xlim(-0.5, 7.5)
    ax.set_xlabel("")
    dist_ax.set_xlim(-0.5, 19.5)
    handles, labels = dist_ax.get_legend_handles_labels()
    dist_ax.legend(
        handles,
        ["0.65", "1.0"],
        title=r"$t_S$",
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
    )
    dist_ax.legend_.findobj(mpl.patches.Rectangle)[0].set_hatch("////")  # type: ignore

    # Add label (a) to the top right of the axes
    ax.text(-0.15, 1.0, "(a)", transform=ax.transAxes, ha="left", va="bottom")

    # Add label (b) to the top right of the axes
    dist_ax.text(-0.15, 1.0, "(b)", transform=dist_ax.transAxes, ha="left", va="bottom")

    fig.tight_layout()
    fig.savefig(f"{git_dir()}/plots/lorenz96_success_rates.png", dpi=300)
    fig.savefig(f"{git_dir()}/plots/lorenz96_success_rates.eps")


if __name__ == "__main__":

    for len_segs in [35, 45, 55, 65, 75, 100]:
        print(f"len_segs = {len_segs}")
        total_df, fit_df = collect_files(3, dt=0.01, len_segs=len_segs)
        total_df.to_hdf(
            f"{git_dir()}/data/02_analysis/total_df3_dt0.01-len_segs{len_segs}.h5",
            key="data",
        )
        fit_df.to_hdf(
            f"{git_dir()}/data/02_analysis/fit_df_dt0.01-len_segs{len_segs}.h5",
            key="data",
        )

    plot_results()
