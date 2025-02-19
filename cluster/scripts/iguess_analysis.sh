#!/bin/bash
#SBATCH --time 01:30:00
#SBATCH --qos 2h
#SBATCH --partition medium
#SBATCH --mem 128G
#SBATCH -C cascadelake
#SBATCH -o cluster/error/%j_iguess_analysis.out
#SBATCH -e cluster/error/%j_iguess_analysis.err

source .venv/bin/activate
python scripts/iguess_analysis.py
