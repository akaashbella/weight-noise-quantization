#!/bin/bash
#SBATCH -J wnq-eval-noise
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH --array=0-79%4
#SBATCH --mail-type=ALL,TIMELIMIT_90
#SBATCH --mail-user=bellaak@oregonstate.edu
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail

MODELS=(resnet50 mobilenetv3_large convnext_tiny efficientnet_b0)
REGIMES=(clean noisy)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

idx=${SLURM_ARRAY_TASK_ID}
seed=${SEEDS[$(( idx % 10 ))]}
tmp=$(( idx / 10 ))
regime=${REGIMES[$(( tmp % 2 ))]}
model=${MODELS[$(( tmp / 2 ))]}

RUN_DIR="runs/cifar100/${model}/${regime}/seed_${seed}"

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

echo "Starting EVAL_NOISE: ${RUN_DIR}"
python -m src.exp.eval_noise \
  --run-dir "$RUN_DIR" \
  --checkpoint best \
  --split test \
  --n-repeats 1