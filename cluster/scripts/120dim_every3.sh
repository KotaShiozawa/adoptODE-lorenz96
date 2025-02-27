#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=inga.kottlarz@ds.mpg.de
#SBATCH --output=cluster/outputs/%j_lorenz96_120dim_%A-%a.out
#SBATCH --job-name=lorenz96_120dim_%A-%a
#SBATCH --error=cluster/error/%j_lorenz96_120dim_%A-%a.err
#SBATCH --time 01:00:00
#SBATCH -p gpu
#SBATCH -G RTX5000
#SBATCH --mem 48G
#SBATCH -a 1-50

sleep $SLURM_ARRAY_TASK_ID
source .venv/bin/activate
python scripts/sebastian.py --observe_every=3 --N_sys=500 --D=120 --N_time_steps=10000 --dt=0.01 --len_segs=45 --initialization=observed_dist --cut_off=45
