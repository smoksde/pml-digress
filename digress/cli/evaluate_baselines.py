import argparse
import json
from pathlib import Path

from digress.config import BASELINES_DIR, EVAL_SAMPLES, HEURISTIC_P_RING, HEURISTIC_P_UPGRADE, RANDOM_SEED
from digress.data.chem import NUM_ATOM_TYPES, NUM_BOND_TYPES
from digress.data.dataset import get_dataset
from digress.data.marginals import bootstrap_stats
from digress.evaluation.baselines import BASELINES
from digress.evaluation.metrics import distribution_distances, evaluate, graph_histograms
from digress.report.tables import Table, format_table
from digress.training.checkpoint import MARGINAL_KEYS
from digress.training.loop import reference_histograms

COLUMNS = ("valid_connected", "validity", "uniqueness", "novelty", "vun",
           "tv_n_atoms", "tv_atom_types", "tv_bond_types", "tv_valence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=EVAL_SAMPLES, help="molecules to sample per baseline")
    parser.add_argument("--ref", choices=("test", "val"), default="test", help="reference molecules for the TV distances")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--out", type=Path, default=BASELINES_DIR)
    parser.add_argument("--image", action="store_true", help="also draw 16 samples per baseline to --out")
    args = parser.parse_args()

    data = get_dataset()
    # same statistics a training run uses for its noise, so baselines and model start from identical frequencies
    marginals = dict(zip(MARGINAL_KEYS, bootstrap_stats(NUM_ATOM_TYPES, NUM_BOND_TYPES)))
    reference = reference_histograms(data, data.test_idx if args.ref == "test" else data.val_idx)

    results = {"reference": args.ref, "n": args.n, "seed": args.seed,
               "heuristic_p_upgrade": HEURISTIC_P_UPGRADE, "heuristic_p_ring": HEURISTIC_P_RING, "baselines": {}}
    args.out.mkdir(parents=True, exist_ok=True)
    for key, (name, sample) in BASELINES.items():
        atoms, adj, mask = sample(marginals, args.n, seed=args.seed)
        scores = evaluate(atoms, adj, mask, data.train_smiles_flat)
        scores.update(distribution_distances(graph_histograms(atoms, adj, mask), reference))
        results["baselines"][key] = {"name": name, **scores}
        if args.image:
            from digress.report.molecules import show

            show(atoms[:16], adj[:16], mask[:16], outfile=args.out / f"{key}.png")

    table = Table("baselines", f"Non-learned baselines, {args.n} samples each, TV against {args.ref} molecules.",
                  ("Baseline", *COLUMNS),
                  [(b["name"], *(f"{b[k]:.3f}" for k in COLUMNS)) for b in results["baselines"].values()])
    print("\n".join(format_table(table)))
    path = args.out / "baseline_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
