"""
changes:

a run has data, settings and the info and T properties.

SETTING_KEYS saved at ckpt["settings"]

resolve_settings() sets None values to not given/ specified, using the defaults and passed values if any.

LEGACY_SETTINGS for Milestone 1 QM9 setting

make_optimizer adds like in authors' code AdamW with amsgrad (for consistency, might not be needed if clipping grad.)

load_for_sampling() replaced by load_run() taking a path as argument, returning a complete run in eval. modew
"""

from dataclasses import dataclass, field

import torch

from digress.config import (
    DEVICE,
    DIFFUSION_STEPS,
    LABEL_PROPERTY,
    LEGACY_DIFFUSION_STEPS,
    QM9_SUBSET_SIZE,
    TRANSITION,
    WEIGHT_DECAY,
)
from digress.data.dataset import get_dataset
from digress.data.marginals import bootstrap_stats
from digress.diffusion.discrete import make_diffusions
from digress.models.registry import create_model, new_model_config

MARGINAL_KEYS = ("atom_marginal", "bond_marginal", "atom_count_dist")

SETTING_KEYS = ("dataset", "subset_size", "aromatic", "label_property", "diffusion_steps", "transition", "optimizer",
                "weight_decay")
# Milestone 1 settings
LEGACY_SETTINGS = {"dataset": "qm9", "subset_size": QM9_SUBSET_SIZE, "aromatic": False, "label_property": None,
                   "diffusion_steps": LEGACY_DIFFUSION_STEPS, "transition": "marginal", "optimizer": "adam",
                   "weight_decay": 0.0}


def default_settings(model_type):
    return {"dataset": "qm9", "subset_size": None, "aromatic": False, "label_property": LABEL_PROPERTY,
            "diffusion_steps": DIFFUSION_STEPS, "transition": TRANSITION,
            "optimizer": "adamw" if model_type == "digress" else "adam",
            "weight_decay": WEIGHT_DECAY if model_type == "digress" else 0.0}


@dataclass
class Run:
    model: torch.nn.Module
    opt: torch.optim.Optimizer
    model_config: dict
    marginals: dict
    atom_diffusion: object
    bond_diffusion: object
    data: object
    settings: dict = field(default_factory=dict)
    start_step: int = 0
    best_val: float = float("inf")

    @property
    def info(self):
        return self.data.info

    @property
    def T(self):
        return self.atom_diffusion.T


def checkpoint_settings(ckpt):
    if "settings" in ckpt:
        return {**LEGACY_SETTINGS, **ckpt["settings"]}
    # Milestone 1 checkpoint: only label was stored (its "optimizer" is optimizer state)
    return {**LEGACY_SETTINGS, "label_property": ckpt.get("label_property")}


def resolve_settings(requested, ckpt, model_type):
    """
    Settings of the run: From checkpoint when resuming (explicit values, conflicting requests yield error);
    Otherwise the requested values additionally to the defaults. requested maps keys to a value or None (meaning not specified/ given)
    """
    requested = {k: v for k, v in (requested or {}).items() if v is not None}
    if ckpt is None:
        return {**default_settings(model_type), **requested}
    settings = checkpoint_settings(ckpt)
    conflicts = {k: (settings[k], v) for k, v in requested.items() if settings.get(k) != v}
    if conflicts:
        raise ValueError(f"cannot resume with different settings (checkpoint, requested): {conflicts}; "
                         "use a new run name")
    return settings


def make_optimizer(model, name, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


def load_checkpoint(path):
    if not path.exists():
        return None
    return torch.load(path, map_location=DEVICE, weights_only=False)


def save_checkpoint(path, run, step):
    torch.save(
        {
            "model": run.model.state_dict(),
            "model_config": run.model_config,
            "settings": run.settings,
            "label_property": run.settings["label_property"],
            "y_mean": run.data.y_mean,
            "y_std": run.data.y_std,
            "optimizer": run.opt.state_dict(),
            **run.marginals,
            "step": step,
            "best_val": run.best_val,
        },
        path,
    )


def build_run(ckpt, lr, settings=None, model_type=None, hidden_dim=None, n_layers=None, dropout=None, features=None):
    """Rebuilds a run from a checkpoint, or creates a new one from `settings` (see resolve_settings)."""
    if ckpt is not None:
        settings = checkpoint_settings(ckpt) if settings is None else settings
    data = get_dataset(settings["dataset"], settings["subset_size"], settings["aromatic"], settings["label_property"])
    if ckpt is not None:
        model_config = ckpt["model_config"]
    else:
        y_dim = 1 if settings["label_property"] else 0
        model_config = new_model_config(model_type, settings["dataset"], settings["diffusion_steps"], y_dim,
                                        hidden_dim, n_layers, dropout, features)
    model = create_model(model_config, data.info).to(DEVICE)
    opt = make_optimizer(model, settings["optimizer"], lr, settings["weight_decay"])
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        for group in opt.param_groups:
            group["lr"] = lr
        marginals = {k: ckpt[k] for k in MARGINAL_KEYS}
    else:
        marginals = dict(zip(MARGINAL_KEYS, bootstrap_stats(data)))
    run = Run(model, opt, model_config, marginals,
              *make_diffusions(marginals, settings["transition"], settings["diffusion_steps"]), data, settings)
    if ckpt is not None:
        run.start_step, run.best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
    return run


def load_run(path):
    """Run from a checkpoint file, model in eval mode (when sampling/ evaluating)"""
    ckpt = load_checkpoint(path)
    if ckpt is None:
        raise FileNotFoundError(path)
    run = build_run(ckpt, lr=0.0)
    run.model.eval()
    return ckpt, run
