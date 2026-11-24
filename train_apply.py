"""
New file train_apply.py

like in task sheet, to dispatch training (with model selection on val loss) and returning predictions for the test data. E.g.

from train_apply import train_apply
    out = train_apply("digress", "qm9")                       # paper setting on QM9
    out = train_apply("digress", "qm9", transition="uniform") # noise-model comparison using uniform rather than default marginal
    out = train_apply("digress", "moses", subset_size=200_000, max_minutes=110) ....

Every argument except method and dataset has a default. Calling again with same run_name resumes the run from its last checkpoint and then evaluates-.

"""

from digress.config import (
    DIFFUSION_STEPS,
    EDGE_LOSS_WEIGHT,
    EVAL_INTERVAL,
    EVAL_SAMPLES,
    FINAL_SAMPLES,
    GRAD_CLIP,
    RANDOM_SEED,
    TEST_EVAL_SIZE,
    TRAIN_STEPS,
    VAL_SIZE,
)
from digress.features import FeatureConfig
from digress.models.registry import MODEL_REGISTRY

METHODS = tuple(sorted(MODEL_REGISTRY))


def _default_run_name(method, dataset, transition, features, aromatic, subset_size, diffusion_steps, seed):
    parts = [method, dataset, transition, f"T{diffusion_steps}"]
    if features.structural != "none":
        parts.append(features.structural)
    if features.molecular:
        parts.append("mol")
    if features.node_index:
        parts.append("nodeidx")
    if aromatic:
        parts.append("arom")
    if subset_size:
        parts.append(f"sub{subset_size}")
    return "_".join(parts + [f"s{seed}"])


def train_apply(method="digress", dataset="qm9", *, run_name=None,
                # data options
                subset_size=None, aromatic=False,
                # diffusion options
                diffusion_steps=DIFFUSION_STEPS, transition="marginal",
                # model (for new runs only!)
                n_layers=None, hidden_dim=None, dropout=None,
                extra_features="none", molecular_features=False, node_index_features=False,
                # training options
                steps=TRAIN_STEPS, batch_size=None, lr=None, edge_loss_weight=EDGE_LOSS_WEIGHT, grad_clip=GRAD_CLIP,
                eval_interval=EVAL_INTERVAL, eval_samples=EVAL_SAMPLES, val_size=VAL_SIZE, max_minutes=None,
                seed=RANDOM_SEED,
                # evaluation options
                num_samples=None, max_test=TEST_EVAL_SIZE, distribution_metrics=True):
    """Trains method on dataset and return predictions for its test split.

    Returns a dict with
      samples:   (atoms, adj, mask) of `num_samples` generated graphs (default values: 10k QM9 / 25k MOSES)
      smiles:    their SMILES (relaxed validity, largest fragment; None = invalid) note: maybe even add the non-relaxed from milestone 1?
      metrics:   test loss (overall / per t), sample quality, distribution metrics (see digress.evaluation.test)
      denoising: per-noise-level clean-graph prediction metrics on the test graphs
      run_dir, finished (False if stopped by max_minutes; call again with the same run_name to continue)
    """
    from digress.evaluation.test import evaluate_run, save_samples
    from digress.training.api import run_training
    from digress.training.checkpoint import load_run
    from digress.training.loop import TrainConfig

    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    features = FeatureConfig(extra_features, molecular_features, node_index_features)
    run_name = run_name or _default_run_name(method, dataset, transition, features, aromatic, subset_size,
                                             diffusion_steps, seed)
    cfg = TrainConfig(steps=steps, batch_size=batch_size, lr=lr, seed=seed, edge_loss_weight=edge_loss_weight,
                      grad_clip=grad_clip, eval_interval=eval_interval, eval_samples=eval_samples,
                      val_size=val_size, max_minutes=max_minutes)
    _, run_dir, finished = run_training(
        run_name, cfg, model=method, dataset=dataset, subset_size=subset_size, aromatic=aromatic,
        diffusion_steps=diffusion_steps, transition=transition, hidden_dim=hidden_dim, n_layers=n_layers,
        dropout=dropout, features=features)

    # model selection: checkpoint with the lowest validation loss
    _, best = load_run(run_dir / "checkpoints" / "best.pt")
    num_samples = num_samples or FINAL_SAMPLES[dataset]
    results, samples, smiles = evaluate_run(best, edge_loss_weight, num_samples, max_test, distribution_metrics)
    save_samples(run_dir / "train_apply_samples.pt", *samples, smiles)
    return {"samples": samples, "smiles": smiles, "metrics": results, "denoising": results["denoising"],
            "run_dir": run_dir, "finished": finished}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="train_apply from the command line (all other args: defaults)")
    parser.add_argument("--method", default="digress", choices=METHODS)
    parser.add_argument("--dataset", default="qm9", choices=("qm9", "moses"))
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--transition", default="marginal", choices=("marginal", "uniform"))
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--max-minutes", type=float)
    args = parser.parse_args()
    out = train_apply(args.method, args.dataset, steps=args.steps, transition=args.transition,
                      subset_size=args.subset_size, max_minutes=args.max_minutes)
    print(json.dumps({k: v for k, v in out["metrics"].items() if k != "denoising"}, indent=2, default=str))
