"""
changes: argparser update, groups
"""
import argparse

from digress.config import (
    CHECKPOINT_INTERVAL,
    EDGE_LOSS_WEIGHT,
    EVAL_INTERVAL,
    EVAL_SAMPLES,
    GRAD_CLIP,
    LOG_INTERVAL,
    RANDOM_SEED,
    TRAIN_STEPS,
    VAL_SIZE,
)
from digress.data.info import DATASETS
from digress.diffusion.discrete import TRANSITIONS
from digress.features import STRUCTURAL, FeatureConfig
from digress.models.registry import MODEL_REGISTRY
from digress.training.api import run_training
from digress.training.loop import TrainConfig


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train (or resume) a denoising diffusion model. Run settings default to the checkpoint's values "
                    "when resuming; model settings only apply to new runs.")
    parser.add_argument("--run-name", required=True, help="folder under outputs/runs/; reuse it to resume")

    run = parser.add_argument_group("run settings (fixed once a run exists)")
    run.add_argument("--dataset", choices=sorted(DATASETS), help="default: qm9")
    run.add_argument("--subset-size", type=int,
                     help="QM9: first N usable molecules (50000 = Milestone-1 data); MOSES: N train molecules")
    run.add_argument("--aromatic", action="store_true", default=None,
                     help="keep aromatic bonds as a 5th edge class (default: kekulize)")
    run.add_argument("--diffusion-steps", type=int, help="T (default 500)")
    run.add_argument("--transition", choices=TRANSITIONS, help="noise limit distribution (default: marginal)")
    run.add_argument("--optimizer", choices=("adam", "adamw"), help="default: adamw for digress, adam otherwise")
    run.add_argument("--weight-decay", type=float)

    model = parser.add_argument_group("model (new runs only)")
    model.add_argument("--model", choices=sorted(MODEL_REGISTRY), help="default: digress")
    model.add_argument("--hidden-dim", type=int, help="width (digress: dx)")
    model.add_argument("--n-layers", type=int, help="digress / simple_graph_transformer depth")
    model.add_argument("--dropout", type=float, help="digress only (default 0.1)")
    model.add_argument("--extra-features", choices=STRUCTURAL, default="none",
                       help="structural/spectral features of the noisy graph (planned, only 'none' works yet)")
    model.add_argument("--molecular-features", action="store_true", help="charge/valence/weight features (planned)")
    model.add_argument("--node-index-features", action="store_true",
                       help="one-hot node position; breaks permutation equivariance (ablation)")

    tr = parser.add_argument_group("training")
    tr.add_argument("--steps", type=int, default=TRAIN_STEPS)
    tr.add_argument("--lr", type=float, help="default: 2e-4 for digress, 1e-3 otherwise")
    tr.add_argument("--batch-size", type=int, help="default: 512 (digress, qm9) / 256 (digress, moses) / 256 (M1)")
    tr.add_argument("--seed", type=int, default=RANDOM_SEED)
    tr.add_argument("--edge-loss-weight", type=float, default=EDGE_LOSS_WEIGHT)
    tr.add_argument("--grad-clip", type=float, default=GRAD_CLIP, help="max gradient norm; 0 = off")
    tr.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    tr.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    tr.add_argument("--checkpoint-interval", type=int, default=CHECKPOINT_INTERVAL)
    tr.add_argument("--eval-samples", type=int, default=EVAL_SAMPLES)
    tr.add_argument("--val-size", type=int, default=VAL_SIZE, help="cap on molecules per validation-loss pass")
    tr.add_argument("--max-minutes", type=float, help="save last.pt and exit after this wall time (SLURM limit)")
    tr.add_argument("--ema-decay", type=float, default=0.0, help="reserved; EMA is not implemented yet")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                      edge_loss_weight=args.edge_loss_weight, grad_clip=args.grad_clip,
                      log_interval=args.log_interval, eval_interval=args.eval_interval,
                      checkpoint_interval=args.checkpoint_interval, eval_samples=args.eval_samples,
                      val_size=args.val_size, max_minutes=args.max_minutes, ema_decay=args.ema_decay)
    features = FeatureConfig(args.extra_features, args.molecular_features, args.node_index_features)
    run_training(args.run_name, cfg, model=args.model, dataset=args.dataset, subset_size=args.subset_size,
                 aromatic=args.aromatic, diffusion_steps=args.diffusion_steps, transition=args.transition,
                 optimizer=args.optimizer, weight_decay=args.weight_decay, hidden_dim=args.hidden_dim,
                 n_layers=args.n_layers, dropout=args.dropout, features=features)


if __name__ == "__main__":
    main()
