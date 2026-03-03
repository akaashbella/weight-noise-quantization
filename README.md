# weight-noise-quantization

Experiment pipeline for CIFAR-100: train CNNs with optional Gaussian weight noise, evaluate under eval-time noise sweeps and weights-only PTQ (int8), then aggregate and analyze correlations between noise sensitivity and quantization drop.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

## Repo layout

- **`src/models/`** — Torchvision backbones (ResNet50, MobileNetV3-Large, ConvNeXt-Tiny, EfficientNet-B0), registry, build/forward helpers.
- **`src/data/`** — CIFAR-100 at 224×224, ImageNet-style transforms, train/val/test dataloaders.
- **`src/exp/`** — Config, paths, training loop (clean/noisy), eval_noise (α sweep + curve features), eval_ptq (W8A16), aggregation, correlation analysis, plotting.
- **`scripts/`** — Sanity checks and example commands.

## Usage

**Train** (clean or noisy regime):

```bash
python -m src.exp.train --model resnet50 --train-regime clean --seed 0 --lr 0.05
python -m src.exp.train --model resnet50 --train-regime noisy --seed 0 --lr 0.05
```

Optional: `--epochs`, `--batch-size-per-gpu`, `--data-root`, `--num-workers`, `--out`, `--resume`, etc.

**Eval under weight noise** (sweep α, write curve features):

```bash
python -m src.exp.eval_noise --run-dir runs/cifar100/resnet50/clean/seed_0 --checkpoint best --split test --n-repeats 3
```

**Eval PTQ** (weights-only int8, drop vs FP32):

```bash
python -m src.exp.eval_ptq --run-dir runs/cifar100/resnet50/clean/seed_0 --checkpoint best --split test --compute-fp32-baseline
```

**Aggregate** (index runs, summaries, correlation, plots):

```bash
python -m src.exp.aggregate --runs-root runs --out-root results --dataset cifar100
```

Outputs: `results/cifar100/summary_*.csv`, `correlation_analysis.csv`, `plots/*.png`.

## Research idea

Weight noise and quantization both perturb Conv/Linear weights. The pipeline tests whether **noise sensitivity** (AUC, init_slope, alpha_50, collapse_alpha) correlates with **PTQ accuracy drop**, and whether **noisy training** improves quantization robustness.
