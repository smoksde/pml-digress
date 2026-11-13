import argparse

from digress.config import NUM_SAMPLES
from digress.data.dataset import get_dataset
from digress.evaluation.metrics import evaluate
from digress.evaluation.sample import sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("-n", type=int, default=NUM_SAMPLES)
    parser.add_argument("--y", type=float, help="target label value (conditional checkpoints only)")
    parser.add_argument("--image", help="also draw the first 16 samples to this PNG")
    args = parser.parse_args()

    atoms, bonds, mask = sample(args.n, args.checkpoint, args.y)
    m = evaluate(atoms.cpu(), bonds.cpu(), mask.cpu(), get_dataset().train_smiles_flat)
    print(f"validity: {m['validity']:.2f}  uniqueness: {m['uniqueness']:.2f}  novelty: {m['novelty']:.2f}  "
          f"(n={m['n']})")
    if args.image:
        from digress.report.molecules import show

        show(atoms[:16], bonds[:16], mask[:16], outfile=args.image)


if __name__ == "__main__":
    main()
