from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D

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


def collect_files(observe_every: int):
    joints = []
    totals = []
    for file in glob(
        f"results/incremental_initial_condition/*every{observe_every}*.h5"
    ):

        print(f"Processing {file}")
        data = xr.open_dataset(file)
        mse_true = (
            ((data.reconstruction - data.ground_truth) ** 2)
            .mean(dim="time")
            .mean(dim="variable")
        )
        mse_obs = (
            ((data.reconstruction - data.ground_truth) ** 2)
            .isel(variable=slice(0, -1, observe_every))
            .mean(dim="time")
            .mean(dim="variable")
        )

        segments = []
        mse = []
        true = []

        for segment in range(data.segment.size):
            mse.append(mse_true.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["true"]
                * len(mse_true.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(mse_true.isel(segment=segment).mean(dim="seed_system").values)
            )
            mse.append(mse_obs.isel(segment=segment).mean(dim="seed_system").values)
            true.append(
                ["observed"]
                * len(mse_obs.isel(segment=segment).mean(dim="seed_system").values)
            )
            segments.append(
                [segment]
                * len(mse_obs.isel(segment=segment).mean(dim="seed_system").values)
            )

        mse = [item for sublist in mse for item in sublist]
        true = [item for sublist in true for item in sublist]
        segments = [item for sublist in segments for item in sublist]

        mse_df = pd.DataFrame({"mse": mse, "type": true, "segment": segments})

        mse_df["oom"] = np.around(np.log10(mse_df.mse))
        totals.append(mse_df)
        min_ooms = [
            np.min(mse_df.loc[mse_df.segment == segment].loc[mse_df.type == type].oom)
            for segment in range(6)
            for type in ["true", "observed"]
        ]
        max_ooms = [
            np.max(mse_df.loc[mse_df.segment == segment].loc[mse_df.type == type].oom)
            for segment in range(6)
            for type in ["true", "observed"]
        ]
        joint = pd.concat(
            [
                mse_df.loc[mse_df.segment == segment]
                .loc[mse_df.type == type]
                .drop(columns="mse")
                .mode()
                .dropna()
                for segment in range(6)
                for type in ["true", "observed"]
            ]
        )
        joint["min_oom"] = min_ooms
        joint["max_oom"] = max_ooms
        joints.append(joint)
        data.close()

    joint_df = pd.concat(joints)
    joint_df["oom"] = 10 ** joint_df["oom"]
    joint_df["min_oom"] = 10 ** joint_df["min_oom"]
    joint_df["max_oom"] = 10 ** joint_df["max_oom"]
    return joint_df, pd.concat(totals)


def plot_results(joint_df, total_df, savename):

    # Set the figure size to half the page width (assuming a typical LaTeX page width of 6.5 inches)
    fig, axs = plt.subplots(
        figsize=(6.5, 3.25), ncols=2, sharey=True
    )  # Adjust the height as needed

    ax0, ax1 = axs

    sns.violinplot(
        ax=ax0,
        data=total_df,
        x="segment",
        y="mse",
        hue="type",
        log_scale=True,
        palette="Set2",
        inner=None,
        split=True,
        width=1,
        dodge=True,
    )
    ax0.set_xlabel("Consecutive segments")
    ax0.set_ylabel("Mean squared error ")
    ax0.legend(frameon=False, bbox_to_anchor=(0.5, 1.15), ncols=2, loc="lower center")

    sns.lineplot(
        ax=ax1, data=joint_df, hue="type", x="segment", y="oom", palette="Set2"
    )
    sns.lineplot(
        ax=ax1,
        data=joint_df,
        hue="type",
        x="segment",
        y="min_oom",
        linestyle="--",
        palette="Set2",
    )
    sns.lineplot(
        ax=ax1,
        data=joint_df,
        hue="type",
        x="segment",
        y="max_oom",
        linestyle=":",
        palette="Set2",
    )
    ax1.set_yscale("log")
    ax0.set_xlabel("Consecutive segments")
    ax1.set_ylabel(r"$\mathcal{O}(E)$")

    # Create custom legend handles with the new colors
    handles = [
        Line2D([0], [0], color="black", lw=1.5, label=r"$\mathcal{O}$(Highest mode)"),
        Line2D([0], [0], color="black", lw=1.5, linestyle="--", label="Minimum"),
        Line2D([0], [0], color="black", lw=1.5, linestyle=":", label="Maximum"),
    ]

    # Add the custom legend to the plot, positioned on top of the plot
    ax1.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.15),
        ncols=2,
        frameon=False,
    )

    # Adjust layout to make room for the legend
    fig.tight_layout()
    fig.savefig(f"{savename}.png", dpi=300)


if __name__ == "__main__":

    mse_df3, total_df3 = collect_files(3)
    plot_results(joint_df=mse_df3, total_df=total_df3, savename="mse_every3")

    mse_df4, total_df4 = collect_files(4)
    plot_results(joint_df=mse_df4, total_df=total_df4, savename="mse_every4")
