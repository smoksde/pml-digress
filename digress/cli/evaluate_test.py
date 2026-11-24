"""
changes/ see cli/commmon.py:; also for cli/sample.py, cli/evaluate_baseline.py cli/export_smiles.py
"""

import argparse
import json
from pathlib import Path

from digress.config import EDGE_LOSS_WEIGHT, EVAL_SAMPLES, REFERENCE_SIZE, TEST_EVAL_SIZE
from digress.evaluation.denoising import denoising_table
from digress.evaluation.test import evaluate_run, save_samples
from digress.report.tables import format_table
from digress.training.checkpoint import load_run


def _edge_weight(run_dir, override):
    if override is not None:
        return override
    config = run_dir / "config.json"
    if config.exists():
        return json.loads(config.read_text()).get("edge_loss_weight", EDGE_LOSS_WEIGHT)
    return EDGE_LOSS_WEIGHT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="e.g. outputs/runs/my_run/checkpoints/best.pt")
    parser.add_argument("-n", type=int, default=EVAL_SAMPLES, help="molecules to sample for sample quality")
    parser.add_argument("--max-test", type=int, default=TEST_EVAL_SIZE,
                        help="cap on test molecules for test loss / denoising metrics")
    parser.add_argument("--reference-size", type=int, default=REFERENCE_SIZE,
                        help="test molecules used as reference for SNN / Frag / Scaf / FCD")
    parser.add_argument("--no-distribution", action="store_true", help="skip SNN / Frag / Scaf / FCD / Filters")
    parser.add_argument("--edge-loss-weight", type=float, help="default: the value from the run's config.json")
    parser.add_argument("--out", type=Path, help="default: test_results.json in the run folder")
    args = parser.parse_args()

    ckpt, run = load_run(args.checkpoint)
    run_dir = args.checkpoint.resolve().parent.parent
    edge_weight = _edge_weight(run_dir, args.edge_loss_weight)
    results, (atoms, adj, mask), smiles = evaluate_run(run, edge_weight, args.n, args.max_test,
                                                       not args.no_distribution, args.reference_size)
    results = {"checkpoint": str(args.checkpoint), "step": ckpt["step"], "best_val_loss": ckpt.get("best_val"),
               **results}

    print(f"checkpoint step {results['step']}  |  val loss {results['best_val_loss']:.4f}  |  test loss "
          f"{results['test_loss']:.4f} (atom {results['test_atom']:.4f}, bond {results['test_bond']:.4f})  on "
          f"{results['n_test_evaluated']} test molecules")
    print("test loss by t: " + "  ".join(f"{k.removeprefix('test_')} {v:.3f}" for k, v in results.items()
                                         if k.startswith("test_t")))
    print("samples: " + "  ".join(f"{k} {v:.3f}" for k, v in results["samples"].items() if k != "n"))
    if "distribution" in results:
        print("distribution: " + "  ".join(f"{k} {'-' if v is None else f'{v:.3f}'}"
                                           for k, v in results["distribution"].items()))
    print("\n".join(format_table(denoising_table(results["denoising"], run.info.bond_labels))))

    out = args.out or run_dir / "test_results.json"
    out.write_text(json.dumps(results, indent=2))
    save_samples(out.with_name("test_samples.pt"), atoms, adj, mask, smiles)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
