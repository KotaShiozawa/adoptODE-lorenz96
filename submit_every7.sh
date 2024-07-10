#!/bin/bash
#SBATCH --job-name=lorenz96_every7
#SBATCH --output=lorenz96_every7_%j.out
#SBATCH --error=lorenz96_every7_%j.err
#SBATCH --time 120
#SBATCH --qos 2h
#SBATCH -p gpu
#SBATCH -G RTX5000
#SBATCH --mem 32G


source .venv/bin/activate
python estimation.py --every=7