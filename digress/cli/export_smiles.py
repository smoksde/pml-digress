import argparse
from pathlib import Path

from digress.config import BASELINES_DIR, NUM_SAMPLES, RANDOM_SEED
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
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
    args = parser.parse_args()

    if args.baseline:
        marginals = dict(zip(MARGINAL_KEYS, bootstrap_stats(NUM_ATOM_TYPES, NUM_BOND_TYPES)))
        atoms, adj, mask = BASELINES[args.baseline][1](marginals, args.n, seed=args.seed)
        out = args.out or BASELINES_DIR / f"{args.baseline}.smi"
    else:
        from digress.evaluation.sample import sample

        atoms, adj, mask = (x.cpu() for x in sample(args.n, args.checkpoint, args.y))
        out = args.out or args.checkpoint.resolve().parent.parent / "samples.smi"

    out.parent.mkdir(parents=True, exist_ok=True)
    counts = write_smi(atoms, adj, mask, out)
    print(f"wrote {sum(counts.values())} molecules to {out}  " + "  ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
