#!/usr/bin/env bash
# Example: run eval_noise on an existing run dir with multiple repeats.
# Run from project root.

python -m src.exp.eval_noise \
  --run-dir runs/cifar100/resnet50/clean/seed_0 \
  --checkpoint best \
  --split test \
  --n-repeats 3
