import os
from pathlib import Path

import torch

# Environment overrides
_ROOT = Path(__file__).resolve().parent.parent
# raw QM9 (PyG), only read to build the cache
DATA_DIR = Path(os.environ.get("DATASETS_DATASET_DIR", _ROOT / "qm9_pyg")).expanduser()
# MOSES csv files (train.csv, test.csv, test_scaffolds.csv); downloaded here if missing
MOSES_DIR = Path(os.environ.get("MOSES_DIR", _ROOT / "moses")).expanduser()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", _ROOT / "outputs")).expanduser()
# QM9 target to condition on, e.g. "gap"; unset = unconditional (conditional generation left out for now)
LABEL_PROPERTY = os.environ.get("LABEL_PROPERTY") or None
# worker processes used when building a dataset cache (RDKit parsing of ~1.9M MOSES SMILES)
DATA_WORKERS = int(os.environ.get("DATA_WORKERS", "1"))

# Paths
CACHE_DIR = OUTPUT_DIR / "cache"
RUNS_DIR = OUTPUT_DIR / "runs"
BASELINES_DIR = OUTPUT_DIR / "baselines"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 0

# Data
DATASET = "qm9"
MAX_ATOMS = 9  # QM9 only; other datasets take it from their DatasetInfo
NUM_EDGES = MAX_ATOMS * (MAX_ATOMS - 1) // 2
QM9_SUBSET_SIZE = 50_000  # the Milestone 1 subset (--subset-size 50000); M2 default is the full dataset
TRAIN_FRACTION = 0.9  # QM9: the rest is the test set, only for final evaluation
VAL_FRACTION = 0.1  # QM9: share of the train part held out for validation / checkpoint selection
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

# Diffusion
DIFFUSION_STEPS = 500  # DiGress paper default
LEGACY_DIFFUSION_STEPS = 200  # M1 checkpoints trained with this
TRANSITION = "marginal"
COSINE_SCHEDULE_OFFSET = 0.008

# Models
DEFAULT_MODEL = "digress"
M1_MODEL_PARAMS = {"hidden_dim": 256, "time_dim": 64}  # mlp / equivariant_mlp / simple_graph_transformer
# Graph Transformer sizes from DiGress authors (see their model/discrete.yaml, experiment/qm9_no_h.yaml, experiment/moses.yaml)
DIGRESS_PRESETS = {
    "qm9": {
        "n_layers": 9,
        "hidden_mlp_dims": {"X": 256, "E": 128, "y": 128},
        "hidden_dims": {"dx": 256, "de": 64, "dy": 64, "n_head": 8, "dim_ffX": 256, "dim_ffE": 128, "dim_ffy": 128},
        "dropout": 0.1,
    },
    "moses": {
        "n_layers": 12,
        "hidden_mlp_dims": {"X": 256, "E": 128, "y": 256},
        "hidden_dims": {"dx": 256, "de": 64, "dy": 128, "n_head": 8, "dim_ffX": 256, "dim_ffE": 128, "dim_ffy": 256},
        "dropout": 0.1,
    },
}

# Training
TRAIN_STEPS = 10_000
BATCH_SIZE = 256  # 256 for M1 models
LEARNING_RATE = 1e-3  # M1 models (using Adam)
# defaults for the DiGress model when --lr / --batch-size are not given (DiGress authors uses AdamW and lr 2e-4)
DIGRESS_TRAIN_PRESETS = {"qm9": {"lr": 2e-4, "batch_size": 512}, "moses": {"lr": 2e-4, "batch_size": 256}}
WEIGHT_DECAY = 1e-12  # DiGress authors AdamW default
EDGE_LOSS_WEIGHT = 5.0  # DiGress paper
GRAD_CLIP = 10.0 # same as in M1, maybe turn off later / change when using with normal Adam

LOG_INTERVAL = 500  # train/val loss
CHECKPOINT_INTERVAL = 200
EVAL_INTERVAL = 1000  # /interval : sample quality: validity, uniqueness, novelty, distribution distances
EVAL_SAMPLES = 1000
VAL_T_BUCKETS = 10  # val loss is also reported at this many fixed diffusion steps
VAL_SIZE = 10_000  # cap on validation molecules per loss evaluation (MOSES validation set has ~176k)
TEST_EVAL_SIZE = 10_000  # cap on test molecules for the test loss / denoising metrics
REFERENCE_SIZE = 10_000  # cap on reference molecules for histogram TVs and SNN/Frag/Scaf

# Sampling
NUM_SAMPLES = 100
SAMPLE_BATCH_SIZE = 512  # graphs are denoised in parallel
FINAL_SAMPLES = {"qm9": 1_000, "moses": 2_500}  # eval size (smaller for now, paper uses qm9 10k and MOSES 25k)

# Valence-heuristic baseline
HEURISTIC_P_UPGRADE = 0.3
HEURISTIC_P_RING = 0.3
