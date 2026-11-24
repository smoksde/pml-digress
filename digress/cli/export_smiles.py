import argparse
from pathlib import Path

from digress.cli.common import add_dataset_args, baselines_dir, dataset_from_args
from digress.config import NUM_SAMPLES, RANDOM_SEED
from digress.data.marginals import bootstrap_stats
from digress.evaluation.baselines import BASELINES
from digress.export.smiles import write_smi
from digress.training.checkpoint import MARGINAL_KEYS


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("checkpoint", nargs="?", type=Path, help="e.g. outputs/runs/my_run/checkpoints/best.pt")
    source.add_argument("--baseline", choices=sorted(BASELINES))
    parser.add_argument("-n", type=int, default=NUM_SAMPLES)
    parser.add_argument("--y", type=float, help="target label value (conditional checkpoints only)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, help="default: samples.smi in the run folder or <baseline>.smi")
    add_dataset_args(parser)  # for baselines only,  checkpoints associated to their dataset
    args = parser.parse_args()

    if args.baseline:
        data = dataset_from_args(args)
        info = data.info
        marginals = dict(zip(MARGINAL_KEYS, bootstrap_stats(data)))
        atoms, adj, mask = BASELINES[args.baseline][1](marginals, args.n, info, seed=args.seed)
        out = args.out or baselines_dir(data) / f"{args.baseline}.smi"
    else:
        from digress.evaluation.sample import sample

        atoms, adj, mask, run = sample(args.n, args.checkpoint, args.y)
        atoms, adj, mask, info = atoms.cpu(), adj.cpu(), mask.cpu(), run.info
        out = args.out or args.checkpoint.resolve().parent.parent / "samples.smi"

    out.parent.mkdir(parents=True, exist_ok=True)
    counts = write_smi(atoms, adj, mask, out, info)
    print(f"wrote {sum(counts.values())} molecules to {out}  " + "  ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
