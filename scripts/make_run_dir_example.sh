#!/usr/bin/env bash
# Example: create a run directory and write config (training stub only).
# Run from project root.

python -m src.exp.train --model resnet50 --train-regime clean --seed 0 --lr 0.05
