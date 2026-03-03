#!/usr/bin/env bash
# Example: run full aggregation (index, summaries, correlation, plots).
# Run from project root.

python -m src.exp.aggregate --runs-root runs --out-root results --dataset cifar100
