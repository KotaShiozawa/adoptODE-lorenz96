from glob import glob

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import xarray as xr

plt.style.use("paper_2col.mplstyle")


def collect_files(observe_every: int, len_segs: int, dt: float = 0.01):
    joints = []
    for file in glob(f"../data/01_simulations/202*every{observe_every}*.h5"):
        if "long_term_prediction" in file:
            continue
        data = xr.open_dataset(file)
        if (
            len(data.data_vars) == 0
            or data.attrs["len_segs"] != len_segs
            or data.attrs["dt"] != dt
            or data.segment.size != 1
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
        return pd.concat(joints)

    return None


if __name__ == "__main__":

    for segment_length in [22, 33, 43, 65, 76, 100]:
        print(f"len_segs = {segment_length}")
        joint_df = collect_files(3, len_segs=segment_length)
        if joint_df is not None:
            print(joint_df)
            joint_df.to_hdf(
                f"../data/02_analysis/initialization_{segment_length}.h5", key="data"
            )

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
            plt.savefig(f"../plots/initialization_{segment_length}.png")
