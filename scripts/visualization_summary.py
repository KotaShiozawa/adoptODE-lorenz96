from glob import glob
import git
from datetime import datetime
import argparse


import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from scipy.optimize import curve_fit

from util import git_dir  #

plt.style.use(f"{git_dir()}/scripts/paper_2col.mplstyle")


def collect_files(
    observe_every: int,
    dt: float = 0.0065,
    N_sys: int = 100,
    len_segs: int = 100,
    initialization: str = "quartiles_uniform",
    D: int = 120,
) -> pd.DataFrame:

    totals = []

    for file in glob(f"{git_dir()}/data/01_simulations/202*-D{D}*every{observe_every}*{initialization}*.h5"):
        data = xr.open_dataset(file)
        if "initialization" not in data.attrs:
            print(f'skipping {file} because of missing initialization attribute')
            data.close()
            continue
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
        e_true = (
            (
                (data.reconstruction - data.ground_truth) ** 2
            ).mean(dim="variable")
        )
        e_obs = (
            (
                (
                    data.reconstruction.isel(variable=slice(0, -1, observe_every))
                    - data.ground_truth.isel(variable=slice(0, -1, observe_every))
                ) ** 2
            ).mean(dim="variable")
        )
        e_true_ta = e_true.mean(dim="time")
        e_obs_ta = e_obs.mean(dim="time")

        segments = []
        error = []
        true = []

        for segment in range(data.segment.size):
            error.append(e_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["true"]
                * len(e_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(e_true_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            error.append(e_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["observed"]
                * len(e_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(e_obs_ta.isel(segment=segment).mean(dim="seed_system").values)
            )

        error = [item for sublist in error for item in sublist]
        true = [item for sublist in true for item in sublist]
        segments = [item for sublist in segments for item in sublist]

        error_df = pd.DataFrame(
            {
                "error": error,
                "type": true,
                "segment": segments,
            }
        )

        if np.any(np.isinf(error_df.error)):
            print(f"Found inf values in {file}")
            continue

        if np.any(np.isinf(error_df.error)):
            print(f"Found inf values in {file}")
            continue

        totals.append(error_df)
        data.close()

    return pd.concat(totals) # , pd.concat(fits)


def add_subplot_axes(ax, rect, facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.25
    # y_labelsize *= rect[3]**0.25
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_results(
        D_system: tuple[int, ...] | int = 120, 
        len_segs: tuple[int, ...] | int = (35, 45, 55, 65, 75, 100), 
        observe_every: tuple[int, ...] | int = 3,
        middle_threshold: float = 1e-5,
        lower_threshold: float = 1e-6,
        upper_threshold: float = 1e-4,
    ):

    fig, (ax, dist_ax) = plt.subplots(nrows=2, figsize=(6.9, 4.5))  # type: ignore
    fmts = ["o", "s", "^", "v", "D", "X", "p", "P", "d"]

    comp_dfs = []
    fit_a = []
    fit_b = []
    fit_c = []
    varied_param = None
    varied_param_name = None
    varied_param_variable_name = None
    example_params = None
    distax_xlims = None
    if isinstance(D_system, (tuple, list)):
        varied_param = D_system
        varied_param_name = "D"
        varied_param_variable_name = "D"
        example_params = [D_system[0], D_system[-1]]
        distax_xlims = (-1, 12.5)
    elif isinstance(len_segs, (tuple, list)):
        varied_param = len_segs
        varied_param_name = r"t_s"
        varied_param_variable_name = "len_segs"
        example_params = [65, 100]
        distax_xlims = (-1, 19.5)
    elif isinstance(observe_every, (tuple, list)):
        varied_param = observe_every
        varied_param_name = r"\text{every}"
        varied_param_variable_name = "every"
        example_params = [3, 5]
        distax_xlims = (-1, 12.5)

    for i, param in enumerate(varied_param):
        if varied_param_variable_name == "D":
            D = param
            len_seg = len_segs
            every = observe_every
        elif varied_param_variable_name == "len_segs":
            len_seg = param
            D = D_system
            every = observe_every
        elif varied_param_variable_name == "every":
            every = param
            len_seg = len_segs
            D = D_system
        else:
            raise ValueError(f"One of D, len_segs, or observe_every must be varied! Got D = {D_system}, len_segs = {len_segs}, observe_every = {observe_every}.")
        total_df = pd.read_hdf(
            f"{git_dir()}/data/02_analysis/total_df3_dt0.01-len_segs{len_seg}_D{D}_observe_every{every}.h5"
        )
        total_df["time"] = total_df["segment"] * len_seg / 100
        success_rates = {middle_threshold: [], upper_threshold: [], lower_threshold: []}

        for segment in total_df["segment"].unique():
            segment_df = total_df[total_df["segment"] == segment]
            denom = len(segment_df) / 2

            success_rate = (
                len(
                    segment_df[
                        np.logical_and(
                            segment_df["error"] < middle_threshold,
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
                            segment_df["error"] < upper_threshold,
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
                            segment_df["error"] < lower_threshold,
                            segment_df["type"] == "true",
                        )
                    ]
                )
                / denom
            )

            success_rates[middle_threshold].append(success_rate)
            success_rates[upper_threshold].append(success_rate_upper)
            success_rates[lower_threshold].append(success_rate_lower)

        success_rates[upper_threshold] = np.array(success_rates[upper_threshold])  # type: ignore
        success_rates[middle_threshold] = np.array(success_rates[middle_threshold])  # type: ignore
        success_rates[lower_threshold] = np.array(success_rates[lower_threshold])  # type: ignore

        def sigmoid(x, a, b, c):
            return a / (1 + np.exp(-b * (x - c)))
        x = total_df.time.unique()
        y = success_rates[middle_threshold]
        popt, _ = curve_fit(sigmoid, x, y, p0=[1, 1, 1])
        fit_a.append(popt[0])
        fit_b.append(popt[1])
        fit_c.append(popt[2])

        ax.errorbar(
            total_df.time.unique(),
            success_rates[middle_threshold],
            yerr=(
                success_rates[middle_threshold] - success_rates[lower_threshold],  # type: ignore
                success_rates[upper_threshold] - success_rates[middle_threshold],  # type: ignore
            ),
            fmt=fmts[i],
            label=r"$%s=%d$" %(varied_param_name, param),
            capsize=3,
            capthick=1,
            color=f"C0{i}",
        )
        ax.plot(
            np.linspace(x.min()-1, x.max()+1, 100),
            sigmoid(np.linspace(x.min()-1, x.max()+1, 100), *popt),
            color=f"C0{i}",
        )

        if param in example_params:
            total_df[varied_param_variable_name] = [param] * len(total_df)
            comp_dfs.append(total_df)

    comp_df = pd.concat(comp_dfs)
    sns.violinplot(
        ax=dist_ax,
        data=comp_df[comp_df.type == "true"],
        x="time",
        y="error",
        hue=varied_param_variable_name,
        palette=[
            colors.to_rgba(f"C0{varied_param.index(example_params[0])}"),
            colors.to_rgba(f"C0{varied_param.index(example_params[1])}"),
        ],
        inner=None,
        split=True,
        width=1,
        dodge=True,
        density_norm="width",
        log_scale=True,
    )
    ncol = 6 if len(varied_param) == 6 else 5
    ax.legend(
        bbox_to_anchor=(0.5, 1.25),
        loc="center",
        ncol=ncol,
    )

    current_xticks = dist_ax.get_xticks()
    dist_ax.set_xticks(current_xticks[::3])
    dist_ax.hlines(middle_threshold, -1, 28.5, color="black", linestyle="--")
    dist_ax.hlines(upper_threshold, -1, 28.5, color="black", linestyle="--")
    dist_ax.hlines(lower_threshold, -1, 28.5, color="black", linestyle="--")
    dist_ax.set_ylabel(r"$E_{\mathrm{true}}$")
    import matplotlib as mpl

    for i, violin in enumerate(dist_ax.findobj(mpl.collections.PolyCollection)):  # type: ignore
        if np.all(np.abs(violin.get_facecolor()[0] - colors.to_rgba(f"C0{varied_param.index(example_params[0])}")) < 0.1):
            violin.set_hatch("////")

    ax.set_ylabel("success rate")
    ax.set_xlim(-0.5, 7.5)
    ax.set_xlabel("")
    dist_ax.set_xlim(*distax_xlims)
    handles, labels = dist_ax.get_legend_handles_labels()
    dist_ax.legend(
        handles,
        example_params,
        title=r"$%s$"%(varied_param_name),
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
    )
    dist_ax.legend_.findobj(mpl.patches.Rectangle)[0].set_hatch("////")  # type: ignore

    # Add label (a) to the top right of the axes
    ax.text(-0.15, 1.0, "(a)", transform=ax.transAxes, ha="left", va="bottom")

    # Add label (b) to the top right of the axes
    dist_ax.text(-0.15, 1.0, "(c)", transform=dist_ax.transAxes, ha="left", va="bottom")
    
    ############################# plot fitted parameters #############################

    fit_params_by_param = {
        "a": fit_a,
        "b": fit_b,
        "c": fit_c,
    }
    rect = [0.7, 0.22, 0.25, 0.28] if varied_param_variable_name != "every" else [0.06, 0.57, 0.15, 0.28]
    fit_ax = add_subplot_axes(ax, rect)
    color_cycle = ["tab:cyan", "navy", "darkgoldenrod"]

    for i, (fit_param, values) in enumerate(fit_params_by_param.items()):
        fit_ax.plot(varied_param, values, marker=fmts[i], label=fit_param, color=color_cycle[i])

    # lazily create some empty space in the legend
    for i, (fit_param, values) in enumerate(fit_params_by_param.items()):
        fit_ax.plot(varied_param, values, marker=fmts[i], label="\t", color="white", alpha=0.0)

    fit_ax.set_xlabel(f"${varied_param_name}$", labelpad=-2)
    fit_ax.text(-0.25, 0.95, "(b)", transform=fit_ax.transAxes, ha="left", va="bottom")

    sigmoid_illustration = plt.imread(f"{git_dir()}/plots/sigmoid.png")
    im = OffsetImage(sigmoid_illustration, zoom=0.17)
    ab = AnnotationBbox(
        im,
        (1.375, -0.05) if varied_param_variable_name != "every" else (6.61, -0.65),
        xycoords=fit_ax.transAxes,
        boxcoords="axes fraction",
        pad=0,
        frameon=False,
        bboxprops=dict(edgecolor="none"),
    )
    fit_ax.add_artist(ab)

    legend = fit_ax.legend(
        bbox_to_anchor=(1.4, 0.7) if varied_param_variable_name != "every" else (6.66, 0.1),
        loc="center",
        title="sigmoid fit",
    )
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0.0, 0.0))
    
    # get current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    fig.tight_layout()
    fig.savefig(f"{git_dir()}/plots/{dt_string}_lorenz96_success_rates_{varied_param_variable_name}.png", dpi=300)
    fig.savefig(f"{git_dir()}/plots/{dt_string}_lorenz96_success_rates_{varied_param_variable_name}.eps")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--recalculate_all", type=bool, default=False)
    parser.add_argument("--recalculate_len_segs", type=bool, default=False)
    parser.add_argument("--recalculate_D", type=bool, default=False)
    parser.add_argument("--recalculate_observe_every", type=bool, default=False)
    args = parser.parse_args()

    observe_every = 3
    D = 120
    len_segs = (35, 45, 55, 65, 75, 100)
    if args.recalculate_all or args.recalculate_len_segs:
        for len_segs in len_segs: 
            print(f"observe_every = {observe_every}")
            total_df = collect_files(dt=0.01, len_segs=len_segs, D=D, observe_every=observe_every)
            total_df.to_hdf(
                f"{git_dir()}/data/02_analysis/total_df_dt0.01-len_segs{len_segs}_D{D}_observe_every{observe_every}.h5",
                key="data",
            )

    plot_results(
        D_system=D,
        len_segs=len_segs,
        observe_every=3,
    )

    len_segs = 65
    D = 120
    observe_every = (3, 4, 5, 6)
    if args.recalculate_all or args.recalculate_observe_every:
        for every in observe_every:
            print(f"observe_every = {observe_every}")
            total_df = collect_files(dt=0.01, len_segs=len_segs, D=D, observe_every=every)
            total_df.to_hdf(
                f"{git_dir()}/data/02_analysis/total_df_dt0.01-len_segs{len_segs}_D{D}_observe_every{every}.h5",
                key="data",
            )
    plot_results(
        D_system=D,
        len_segs=len_segs,
        observe_every=observe_every,
    )
    
    len_segs = 65
    observe_every = 3
    D_system = (60, 90, 120, 150, 180, 210, 240, 270, 300)
    if args.recalculate_all or args.recalculate_D:
        for D in D_system:
            print(f"D = {D}")
            total_df = collect_files(dt=0.01, len_segs=len_segs, D=D, observe_every=observe_every)
            total_df.to_hdf(
                f"{git_dir()}/data/02_analysis/total_df_dt0.01-len_segs{len_segs}_D{D}_observe_every{observe_every}.h5",
                key="data",
            )
    plot_results(
        D_system=D_system,
        len_segs=len_segs,
        observe_every=observe_every,
    )

