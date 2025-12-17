from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

from util import git_dir

plt.style.use(f'{git_dir()}/scripts/paper_2col.mplstyle')


def collect_files(observe_every: int, len_segs: int, dt: float=0.01):
    joints = []
    for file in glob(
        f"{git_dir()}/data/01_simulations/202*every{observe_every}*.h5"
    ):
        if "long_term_prediction" in file:
            continue
        data = xr.open_dataset(file)
        if (
            len(data.data_vars) == 0
            or data.attrs["len_segs"] != len_segs
            or data.attrs["dt"] != dt
            ):
            data.close()
            continue
        print(f"Processing {file}")

        e_true = (
            (
                (
                    data.reconstruction.isel(segment=0, n_sys=slice(0, 100))
                    - data.ground_truth.isel(segment=0)
                ) ** 2
            ).mean(dim="variable")
        )
        e_obs = (
            (
                (
                    data.reconstruction.isel(segment=0, n_sys=slice(0, 100), variable=slice(0, -1, observe_every)) 
                    - data.ground_truth.isel(segment=0, variable=slice(0, -1, observe_every))
                ) ** 2
            ).mean(dim="variable")
        )
        e_true_ta = e_true.mean(dim="time")
        e_obs_ta = e_obs.mean(dim="time")

        joint = pd.DataFrame(
            {
                "seed_system": [data.attrs["seed_system"]] * 100,
                "e_true": e_true_ta.values.flatten(),
                "e_obs": e_obs_ta.values.flatten(),
                "initialization": [data.attrs["initialization"]]* 100,
            }
        )

        joints.append(joint)
        data.close()

    if len(joints) > 0:
        joint_df = pd.concat(joints)
        return joint_df

    return None

LABELS = {
    "observed_dist": "observed distribution",
    "uniform-1_4": r"uniform distribution $\in[-1, 4]$",
    "quartiles_uniform": r"uniform distribution $\in [P_{25}, P_{75}]$",
    "quartiles_observed": r"observed $\lbrace Q2, Q3\rbrace$"
}

if __name__ == "__main__":

    len_segs = 45 # autocorellation zero-crossing
    joint_df = collect_files(3, len_segs=len_segs)
    if joint_df is not None:
        joint_df.to_hdf(f"{git_dir()}/data/02_analysis/initialization_{len_segs}.h5", key="data")
        for label in np.unique(joint_df.initialization):
            joint_df.loc[joint_df.initialization == label, "initialization"] = LABELS[label]
        fig, ax = plt.subplots()
        sns.histplot(
            joint_df,
            x="e_true",
            hue="initialization",
            log_scale=True,
            stat="density",
            element="step",
        )
        ax.set_ylabel('density')
        ax.set_xlabel(r'$E_{\mathrm{true}}$')
        fig.tight_layout()
        plt.savefig(f'{git_dir()}/plots/initialization_{len_segs}.png', dpi=600)
