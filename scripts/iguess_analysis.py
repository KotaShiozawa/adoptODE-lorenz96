from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

# Set the font size parameters
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 9,
    }
)


def collect_files(observe_every: int, len_segs: int, dt: float = 0.01):
    joints = []
    for file in glob(f"results/202*every{observe_every}*.h5"):
        if "long_term_prediction" in file:
            continue
        data = xr.open_dataset(file)
        if (
            len(data.data_vars) == 0
            or "initialization" not in data.attrs.keys()
            or (
                "initialization" in data.attrs.keys()
                and data.attrs["initialization"] == "observed_dist"
                and data.segment.size < 10
            )
            or data.attrs["len_segs"] != len_segs
            or data.attrs["dt"] != dt
        ):
            data.close()
            continue
        print(f"Processing {file}")

        mse_true = (
            (data.reconstruction.isel(segment=0) - data.ground_truth.isel(segment=0))
            ** 2
        ).mean(dim="variable")
        mse_obs = (
            (
                (
                    data.reconstruction.isel(segment=0)
                    - data.ground_truth.isel(segment=0)
                )
                ** 2
            )
            .isel(variable=slice(0, -1, observe_every))
            .mean(dim="variable")
        )
        mse_true_ta = mse_true.mean(dim="time")
        mse_obs_ta = mse_obs.mean(dim="time")

        joint = pd.DataFrame(
            {
                "seed_system": [data.attrs["seed_system"]] * data.n_sys.size,
                "mse_true": mse_true_ta.values.flatten(),
                "mse_obs": mse_obs_ta.values.flatten(),
                "initialization": [data.attrs["initialization"]] * data.n_sys.size,
            }
        )

        joints.append(joint)
        data.close()

    if len(joints) > 0:
        joint_df = pd.concat(joints)
        return joint_df

    return None


if __name__ == "__main__":

    for len_segs in [22, 33, 43, 65, 76, 100]:
        print(f"len_segs = {len_segs}")
        joint_df = collect_files(3, len_segs=len_segs)
        if joint_df is not None:
            print(joint_df)
            joint_df.to_hdf(f"results/initialization_{len_segs}.h5", key="data")

            fig, ax = plt.subplots()
            sns.histplot(
                joint_df,
                x="mse_true",
                hue="initialization",
                log_scale=True,
                palette="Set2",
                multiple="stack",
            )
            fig.tight_layout()
            plt.savefig(f"plots/initialization_{len_segs}.png")
