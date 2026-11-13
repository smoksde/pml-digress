import os
from pathlib import Path

import torch

# Environment overrides
_ROOT = Path(__file__).resolve().parent.parent
# raw QM9, only read to build the cache
DATA_DIR = Path(os.environ.get("DATASETS_DATASET_DIR", _ROOT / "qm9_pyg")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", _ROOT / "outputs")).expanduser()
# QM9 target to condition on, e.g. "gap"; unset = unconditional
LABEL_PROPERTY = os.environ.get("LABEL_PROPERTY") or None

# Paths
CACHE_PATH = OUTPUT_DIR / "cache" / "qm9_cache.pt"
RUNS_DIR = OUTPUT_DIR / "runs"
PLOTS_DIR = OUTPUT_DIR / "plots"
BASELINES_DIR = OUTPUT_DIR / "baselines"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 0

# Data
MAX_ATOMS = 9
NUM_EDGES = MAX_ATOMS * (MAX_ATOMS - 1) // 2  # upper-triangle atom pairs
QM9_SUBSET_SIZE = 50_000
TRAIN_FRACTION = 0.9  # rest is the test set, only for final evaluation
VAL_FRACTION = (
    0.1  # share of the train part held out for validation / checkpoint selection
)
MARGINAL_SAMPLE_SIZE = 2_000  # train molecules used to estimate the noise marginals
QM9_TARGETS = {  # column in the PyG QM9 `y` tensor
    "mu": 0,
    "alpha": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "r2": 5,
    "zpve": 6,
    "u0": 7,
    "u": 8,
    "h": 9,
    "g": 10,
    "cv": 11,
}

# Model and diffusion
MODEL_CONFIG = {
    "type": "mlp",
    "params": {"hidden_dim": 256, "time_dim": 64, "y_dim": 1 if LABEL_PROPERTY else 0},
}
DIFFUSION_STEPS = 200
COSINE_SCHEDULE_OFFSET = 0.008

# Training
TRAIN_STEPS = 10_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EDGE_LOSS_WEIGHT = 5.0
GRAD_CLIP = 10.0

LOG_INTERVAL = 500  # train/val loss
CHECKPOINT_INTERVAL = 200
EVAL_INTERVAL = (
    1000  # sample quality: validity, uniqueness, novelty, distribution distances
)
EVAL_SAMPLES = 1000
VAL_T_BUCKETS = 10  # val loss is also reported at this many fixed diffusion steps

# Sampling
NUM_SAMPLES = 100

# Valence-heuristic baseline
HEURISTIC_P_UPGRADE = 0.3
HEURISTIC_P_RING = 0.3
