import argparse

from digress.config import (BATCH_SIZE, DIFFUSION_STEPS, EDGE_LOSS_WEIGHT, EVAL_INTERVAL, EVAL_SAMPLES,
                            LABEL_PROPERTY, LEARNING_RATE, LOG_INTERVAL, MAX_ATOMS, QM9_SUBSET_SIZE, RANDOM_SEED,
                            RUNS_DIR, TRAIN_STEPS)
from digress.data.dataset import get_dataset
from digress.models.registry import MODEL_REGISTRY
from digress.report.logger import RunLogger
from digress.training.loop import VAL_TS, setup_run, train


def run_description(run, data, **settings):
    return {**settings,
            "eval_samples": EVAL_SAMPLES, "log_interval": LOG_INTERVAL, "diffusion_steps": DIFFUSION_STEPS,
            "model": run.model_config, "label_property": LABEL_PROPERTY,
            "max_atoms": MAX_ATOMS, "subset_size": QM9_SUBSET_SIZE, "train_molecules": len(data.train_idx),
            "parameters": sum(p.numel() for p in run.model.parameters()),
            "train_bonds_per_mol": (data.edge_vecs[data.train_idx] > 0).sum(1).float().mean().item(),
            "resumed_from_step": run.start_step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="folder under outputs/runs/; reuse it to resume")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--edge-loss-weight", type=float, default=EDGE_LOSS_WEIGHT)
    parser.add_argument("--hidden-dim", type=int, help="model width (new runs only)")
    parser.add_argument("--model", choices=sorted(MODEL_REGISTRY), help="model type (new runs only)")
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_name
    run = setup_run(run_dir, args.lr, args.seed, args.model, args.hidden_dim)
    if run.start_step:
        print(f"resuming {args.run_name} from step {run.start_step}/{args.steps}", flush=True)
    logger = RunLogger(run_dir, run.start_step, VAL_TS)
    logger.write_config(run_description(
        run, get_dataset(), run_name=args.run_name, steps=args.steps, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, eval_interval=args.eval_interval, edge_loss_weight=args.edge_loss_weight))
    train(run, run_dir, logger, args.steps, args.batch_size, args.eval_interval, args.edge_loss_weight)


if __name__ == "__main__":
    main()
