import argparse
import json
from pathlib import Path

from digress.config import EDGE_LOSS_WEIGHT, EVAL_SAMPLES
from digress.data.dataset import get_dataset
from digress.training.checkpoint import build_run, load_checkpoint
from digress.training.loop import sample_quality, split_loss


def _edge_weight(run_dir, override):
    if override is not None:
        return override
    config = run_dir / "config.json"
    if config.exists():
        return json.loads(config.read_text()).get("edge_loss_weight", EDGE_LOSS_WEIGHT)
    return EDGE_LOSS_WEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=Path, help="e.g. outputs/runs/my_run/checkpoints/best.pt"
    )
    parser.add_argument(
        "-n",
        type=int,
        default=EVAL_SAMPLES,
        help="molecules to sample for sample quality",
    )
    parser.add_argument(
        "--edge-loss-weight",
        type=float,
        help="default: the value from the run's config.json",
    )
    parser.add_argument(
        "--out", type=Path, help="default: test_results.json in the run folder"
    )
    args = parser.parse_args()

    ckpt = load_checkpoint(args.checkpoint)
    if ckpt is None:
        raise FileNotFoundError(args.checkpoint)
    run_dir = args.checkpoint.resolve().parent.parent
    edge_weight = _edge_weight(run_dir, args.edge_loss_weight)
    run = build_run(ckpt, lr=0.0)
    data = get_dataset()

    (loss, atom, bond), per_t = split_loss(run, edge_weight, data.test_idx, "test")
    samples, _ = sample_quality(run, args.n, data.test_idx)
    results = {
        "checkpoint": str(args.checkpoint),
        "step": ckpt["step"],
        "best_val_loss": ckpt.get("best_val"),
        "n_test": len(data.test_idx),
        "edge_loss_weight": edge_weight,
        "test_loss": loss,
        "test_atom": atom,
        "test_bond": bond,
        **per_t,
        "samples": {"n": args.n, **samples},
    }

    print(
        f"checkpoint step {results['step']}  |  val loss {results['best_val_loss']:.4f}  |  "
        f"test loss {loss:.4f} (atom {atom:.4f}, bond {bond:.4f})  on {len(data.test_idx)} test molecules"
    )
    print(
        "test loss by t: "
        + "  ".join(f"{k.removeprefix('test_')} {v:.3f}" for k, v in per_t.items())
    )
    print(
        "samples: " + "  ".join(f"{k} {v:.3f}" for k, v in samples.items() if k != "n")
    )

    out = args.out or run_dir / "test_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
