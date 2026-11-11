from dataclasses import dataclass

import torch

from digress.config import DEVICE, DIFFUSION_STEPS, LABEL_PROPERTY, MODEL_CONFIG
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.marginals import bootstrap_stats
from digress.diffusion.discrete import DiscreteDiffusion
from digress.models.registry import create_model

MARGINAL_KEYS = ("atom_marginal", "bond_marginal", "atom_count_dist")


@dataclass
class Run:
    model: torch.nn.Module
    opt: torch.optim.Optimizer
    model_config: dict
    marginals: dict
    atom_diffusion: DiscreteDiffusion
    bond_diffusion: DiscreteDiffusion
    start_step: int = 0
    best_val: float = float("inf")


def new_model_config(model_type=None, hidden_dim=None):
    params = dict(MODEL_CONFIG["params"])
    if model_type not in (None, MODEL_CONFIG["type"]):
        params.pop("hidden_dim")
    if hidden_dim is not None:
        params["hidden_dim"] = hidden_dim
    return {"type": model_type or MODEL_CONFIG["type"], "params": params}


def load_checkpoint(path):
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    if ckpt.get("label_property") != LABEL_PROPERTY:
        raise ValueError(
            f"run was trained with label {ckpt.get('label_property')!r} but "
            f"LABEL_PROPERTY={LABEL_PROPERTY!r}; set LABEL_PROPERTY to match or use a new run name"
        )
    return ckpt


def save_checkpoint(path, run, data, step):
    torch.save(
        {
            "model": run.model.state_dict(),
            "model_config": run.model_config,
            "label_property": LABEL_PROPERTY,
            "y_mean": data.y_mean,
            "y_std": data.y_std,
            "optimizer": run.opt.state_dict(),
            **run.marginals,
            "step": step,
            "best_val": run.best_val,
        },
        path,
    )


def build_run(ckpt, lr, model_type=None, hidden_dim=None):
    model_config = (
        ckpt["model_config"]
        if ckpt is not None
        else new_model_config(model_type, hidden_dim)
    )
    model = create_model(model_config).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        marginals = {k: ckpt[k] for k in MARGINAL_KEYS}
    else:
        marginals = dict(
            zip(MARGINAL_KEYS, bootstrap_stats(NUM_ATOM_TYPES, NUM_BOND_TYPES))
        )
    run = Run(
        model,
        opt,
        model_config,
        marginals,
        DiscreteDiffusion(marginals["atom_marginal"], DIFFUSION_STEPS),
        DiscreteDiffusion(marginals["bond_marginal"], DIFFUSION_STEPS),
    )
    if ckpt is not None:
        run.start_step, run.best_val = ckpt["step"], ckpt.get("best_val", float("inf"))
    return run


def load_for_sampling(path):
    ckpt = torch.load(path, map_location=DEVICE)
    model = create_model(ckpt["model_config"]).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    atom_diffusion = DiscreteDiffusion(ckpt["atom_marginal"], DIFFUSION_STEPS)
    bond_diffusion = DiscreteDiffusion(ckpt["bond_marginal"], DIFFUSION_STEPS)
    return ckpt, model, atom_diffusion, bond_diffusion
