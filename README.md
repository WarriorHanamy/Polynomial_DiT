# Polynomial DiT

This repository is a submodule of [https://github.com/SYSU-HILAB/am-planner](https://github.com/SYSU-HILAB/am-planner).

## Setup

### Download Checkpoints

```bash
cd checkpoints
python download.py
```

This will download pre-trained model weights from HuggingFace.

## Two Approaches

This repository provides two methods for trajectory prediction:

### 1. Discrete Points Fitting

Predicts trajectory as a set of discrete waypoints (4 points).

```bash
python demo_points.py
```

### 2. Polynomial Parameter Fitting

Predicts trajectory as polynomial coefficients, generating smooth continuous curves.

```bash
python demo_polynomial.py
```

## Output

Results are saved in the `logs/` directory.

