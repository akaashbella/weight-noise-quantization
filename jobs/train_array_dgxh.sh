#!/bin/bash
#SBATCH -J wnq-train
#SBATCH -p dgxh
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-79%2
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

# Single-GPU default for batch 128
LR=0.05

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

echo "Starting TRAIN: model=${model} regime=${regime} seed=${seed} lr=${LR}"
python -m src.exp.train \
  --model "$model" \
  --train-regime "$regime" \
  --seed "$seed" \
  --lr "$LR" \
  --epochs 200 \
  --batch-size-per-gpu 128 \
  --num-workers 8