#!/bin/bash
#SBATCH --job-name=lorenz96_every7
#SBATCH --output=lorenz96_every7_%j.out
#SBATCH --error=lorenz96_every7_%j.err
#SBATCH --partition=gpu
#SBATCH --G=RTX5000:1
#SBATCH --time=120
#SBATCH --mem=32G
#SBATCH --qos=short


source .venv/bin/activate
python estimation.py --every=7