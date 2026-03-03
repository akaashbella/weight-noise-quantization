#!/usr/bin/env bash
# Example: run PTQ eval on an existing run dir with FP32 baseline computed in-script.
# Run from project root.

python -m src.exp.eval_ptq \
  --run-dir runs/cifar100/resnet50/clean/seed_0 \
  --checkpoint best \
  --split test \
  --compute-fp32-baseline
