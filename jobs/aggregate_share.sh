#!/bin/bash
#SBATCH -J wnq-agg
#SBATCH -p share
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm_logs/%x_%j.out
#SBATCH -e slurm_logs/%x_%j.err
# If you need an account, uncomment and set:
##SBATCH -A eecs

set -euo pipefail

source venv/bin/activate

python -m src.exp.aggregate --runs-root runs --out-root results --dataset cifar100