"""
new file: api.py

One combined entry for training shared by both python -m digress.cli.train and train_apply.train_apply

"""
import random
from dataclasses import asdict, replace

import torch

from digress.config import (
    BATCH_SIZE,
    DEFAULT_MODEL,
    DIGRESS_TRAIN_PRESETS,
    LEARNING_RATE,
    REFERENCE_SIZE,
    RUNS_DIR,
)
from digress.features import FeatureConfig
from digress.report.logger import RunLogger
from digress.training.checkpoint import build_run, load_checkpoint, resolve_settings
from digress.training.loop import TrainConfig, train, val_ts


def run_description(run, run_name, cfg):
    data = run.data
    return {"run_name": run_name, **asdict(cfg), **run.settings,
            "dataset_key": data.info.key, "model": run.model_config, "max_atoms": data.info.max_atoms,
            "train_molecules": len(data.train_idx), "val_molecules": len(data.val_idx),
            "test_molecules": len(data.test_idx), "reference_size": REFERENCE_SIZE,
            "parameters": sum(p.numel() for p in run.model.parameters()),
            "train_bonds_per_mol": (data.edge_vecs[data.train_idx[:20_000]] > 0).sum(1).float().mean().item(),
            "resumed_from_step": run.start_step}


def _fill_defaults(cfg, model_type, dataset):
    preset = DIGRESS_TRAIN_PRESETS[dataset] if model_type == "digress" else {"lr": LEARNING_RATE,
                                                                             "batch_size": BATCH_SIZE}
    return replace(cfg, lr=preset["lr"] if cfg.lr is None else cfg.lr,
                   batch_size=preset["batch_size"] if cfg.batch_size is None else cfg.batch_size)


def run_training(run_name, cfg=None, *, model=None, dataset=None, subset_size=None, aromatic=None,
                 diffusion_steps=None, transition=None, optimizer=None, weight_decay=None, hidden_dim=None,
                 n_layers=None, dropout=None, features=None, runs_dir=RUNS_DIR):
    """
    Trains or resumes outputs/runs/<run_name>. Arguments left None take the checkpoint's value when
    resuming and defaults otherwise; cfg.lr/cfg.batch_size = None take the model/dataset presets.

    Model args (model, hidden_dim, n_layers, dropout, features) only apply to new runs.
    Returns (run, run_dir, finished)
    """
    cfg = cfg or TrainConfig(lr=None, batch_size=None)
    run_dir = runs_dir / run_name
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ckpt = load_checkpoint(run_dir / "checkpoints" / "last.pt")
    model_type = ckpt["model_config"]["type"] if ckpt is not None else (model or DEFAULT_MODEL)
    if ckpt is not None and model not in (None, model_type):
        raise ValueError(f"run {run_name!r} uses model {model_type!r}; use a new run name for {model!r}")
    request = {"dataset": dataset, "subset_size": subset_size, "aromatic": aromatic,
               "diffusion_steps": diffusion_steps, "transition": transition, "optimizer": optimizer,
               "weight_decay": weight_decay}
    settings = resolve_settings(request, ckpt, model_type)
    cfg = _fill_defaults(cfg, model_type, settings["dataset"])
    if isinstance(features, dict):
        features = FeatureConfig.from_dict(features)
    run = build_run(ckpt, cfg.lr, settings, model_type, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
                    features=features)
    if run.start_step:
        print(f"resuming {run_name} from step {run.start_step}/{cfg.steps}", flush=True)

    logger = RunLogger(run_dir, run.start_step, val_ts(run.T), run.info)
    logger.write_config(run_description(run, run_name, cfg))
    finished = train(run, run_dir, logger, cfg)
    return run, run_dir, finished
