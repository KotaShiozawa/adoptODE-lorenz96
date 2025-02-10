#!/bin/bash
#SBATCH --time 01:50:00
#SBATCH --qos 2h
#SBATCH --partition medium
#SBATCH --mem 128G
#SBATCH -C cascadelake
#SBATCH -o cluster/error/%j_visualization.out
#SBATCH -e cluster/error/%j_visualization.err

source .venv/bin/activate
python scripts/visualization_summary.py
