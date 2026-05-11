# Polynomial DiT

This repository is a submodule of [https://github.com/SYSU-HILAB/am-planner](https://github.com/SYSU-HILAB/am-planner).

## Setup

```bash
uv sync
```

### Download Checkpoints

```bash
uv run download-checkpoints
```

This will download pre-trained model weights from HuggingFace.

## Two Approaches

This repository provides two methods for trajectory prediction:

### 1. Discrete Points Fitting

Predicts trajectory as a set of discrete waypoints (4 points).

```bash
uv run demo points
```

### 2. Polynomial Parameter Fitting

Predicts trajectory as polynomial coefficients, generating smooth continuous curves.

```bash
uv run demo trajectory
```

## Output

Results are saved in the `logs/` directory.

