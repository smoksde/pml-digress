import argparse

from digress.config import NUM_SAMPLES
from digress.evaluation.metrics import evaluate
from digress.evaluation.sample import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("-n", type=int, default=NUM_SAMPLES)
    parser.add_argument("--y", type=float, help="target label value (conditional checkpoints only)")
    parser.add_argument("--image", help="also draw the first 16 samples to this PNG")
    args = parser.parse_args()

    atoms, bonds, mask, run = sample(args.n, args.checkpoint, args.y)
    m = evaluate(atoms.cpu(), bonds.cpu(), mask.cpu(), run.data.train_smiles_flat, run.info)
    print(f"validity: {m['validity']:.2f}  relaxed validity: {m['relaxed_validity']:.2f}  "
          f"uniqueness: {m['uniqueness']:.2f}  novelty: {m['novelty']:.2f}  (n={m['n']})")
    if args.image:
        from digress.report.molecules import show

        show(atoms[:16], bonds[:16], mask[:16], outfile=args.image, info=run.info)


if __name__ == "__main__":
    main()
